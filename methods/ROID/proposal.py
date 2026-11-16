import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger as log
from copy import deepcopy
from methods.ROID.my_transforms import get_tta_transforms
__all__ = ['setup']

@torch.no_grad()
def update_model_probs(x_ema, x, momentum=0.9):
    return momentum * x_ema + (1 - momentum) * x

@torch.no_grad()
def ema_update_model(model_to_update, model_to_merge, momentum, update_all=False):
    if momentum < 1.0:
        for (param_to_update, param_to_merge) in zip(model_to_update.parameters(), model_to_merge.parameters()):
            if param_to_update.requires_grad or update_all:
                param_to_update.data = momentum * param_to_update.data + (1 - momentum) * param_to_merge.data.cuda()
    return model_to_update

class Entropy(nn.Module):

    def __init__(self):
        super(Entropy, self).__init__()

    def __call__(self, logits):
        return -(logits.softmax(1) * logits.log_softmax(1)).sum(1)

class SymmetricCrossEntropy(nn.Module):

    def __init__(self, alpha=0.5):
        super(SymmetricCrossEntropy, self).__init__()
        self.alpha = alpha

    def __call__(self, x, x_ema):
        return -(1 - self.alpha) * (x_ema.softmax(1) * x.log_softmax(1)).sum(1) - self.alpha * (x.softmax(1) * x_ema.log_softmax(1)).sum(1)

class AugCrossEntropy(nn.Module):

    def __init__(self, alpha=0.5):
        super(AugCrossEntropy, self).__init__()
        self.alpha = alpha

    def __call__(self, x, x_aug, x_ema):
        return -(1 - self.alpha) * (x.softmax(1) * x_ema.log_softmax(1)).sum(1) - self.alpha * (x_aug.softmax(1) * x_ema.log_softmax(1)).sum(1)

class SoftLikelihoodRatio(nn.Module):

    def __init__(self, clip=0.99, eps=1e-05):
        super(SoftLikelihoodRatio, self).__init__()
        self.eps = eps
        self.clip = clip

    def __call__(self, logits):
        probs = logits.softmax(1)
        probs = torch.clamp(probs, min=0.0, max=self.clip)
        return -(probs * torch.log(probs / (torch.ones_like(probs) - probs) + self.eps)).sum(1)

def copy_model_and_optimizer(model, optimizer):
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return (model_state, optimizer_state)

class ROID(nn.Module):

    def __init__(self, cfg, model, optimizer):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.cfg = cfg
        self.steps = cfg.OPTIM.STEPS
        assert self.steps > 0, 'PR requires >= 1 step(s) to forward and update'
        self.use_weighting = self.cfg.ADAPTER.ROID.USE_WEIGHTING
        self.use_prior_correction = self.cfg.ADAPTER.ROID.USE_PRIOR_CORRECTION
        self.use_consistency = self.cfg.ADAPTER.ROID.USE_CONSISTENCY
        self.momentum_src = self.cfg.ADAPTER.ROID.MOMENTUM_SRC
        self.momentum_probs = self.cfg.ADAPTER.ROID.MOMENTUM_PROBS
        self.temperature = self.cfg.ADAPTER.ROID.TEMPERATURE
        self.batch_size = self.cfg.TEST.BATCH_SIZE
        self.num_classes = self.cfg.CORRUPTION.NUM_CLASS
        self.class_probs_ema = 1 / self.num_classes * torch.ones(self.num_classes).cuda()
        self.tta_transform = get_tta_transforms(cfg, padding_mode='reflect', cotta_augs=False)
        self.slr = SoftLikelihoodRatio()
        self.symmetric_cross_entropy = SymmetricCrossEntropy()
        self.softmax_entropy = Entropy()
        self.src_model = deepcopy(self.model).cpu()
        for param in self.src_model.parameters():
            param.detach_()
        (self.model_states, self.optimizer_state) = copy_model_and_optimizer(self.model, self.optimizer)

    def loss_calculation(self, x):
        imgs_test = x
        outputs = self.model(imgs_test)
        if self.use_weighting:
            with torch.no_grad():
                weights_div = 1 - F.cosine_similarity(self.class_probs_ema.unsqueeze(dim=0), outputs.softmax(1), dim=1)
                weights_div = (weights_div - weights_div.min()) / (weights_div.max() - weights_div.min())
                mask = weights_div < weights_div.mean()
                weights_cert = -self.softmax_entropy(logits=outputs)
                weights_cert = (weights_cert - weights_cert.min()) / (weights_cert.max() - weights_cert.min())
                weights = torch.exp(weights_div * weights_cert / self.temperature)
                weights[mask] = 0.0
                self.class_probs_ema = update_model_probs(x_ema=self.class_probs_ema, x=outputs.softmax(1).mean(0), momentum=self.momentum_probs)
        loss_out = self.slr(logits=outputs)
        if self.use_weighting:
            loss_out = loss_out * weights
            loss_out = loss_out[~mask]
        loss = loss_out.sum() / self.batch_size
        if self.use_consistency:
            outputs_aug = self.model(self.tta_transform(imgs_test[~mask]))
            loss += (self.symmetric_cross_entropy(x=outputs_aug, x_ema=outputs[~mask]) * weights[~mask]).sum() / self.batch_size
        return (outputs, loss)

    def forward(self, x, y=None, adapt=True):
        if not adapt:
            return self.model(x)
        for _ in range(self.steps):
            outputs = self.forward_and_adapt(x)
        return outputs

    @torch.enable_grad()
    def forward_and_adapt(self, x):
        (outputs, loss) = self.loss_calculation(x)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.model = ema_update_model(model_to_update=self.model, model_to_merge=self.src_model, momentum=self.momentum_src)
        with torch.no_grad():
            if self.use_prior_correction:
                prior = outputs.softmax(1).mean(0)
                smooth = max(1 / outputs.shape[0], 1 / outputs.shape[1]) / torch.max(prior)
                smoothed_prior = (prior + smooth) / (1 + smooth * outputs.shape[1])
                outputs *= smoothed_prior
        return outputs

    def reset(self):
        if self.model_states is None or self.optimizer_state is None:
            raise Exception('cannot reset without saved model/optimizer state')
        self.model.load_state_dict(self.model_states, strict=True)
        self.optimizer.load_state_dict(self.optimizer_state)
        self.class_probs_ema = 1 / self.num_classes * torch.ones(self.num_classes).cuda()

def collect_params(model):
    """Collect the affine scale + shift parameters from norm layers.
    Walk the model's modules and collect all normalization parameters.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for (nm, m) in model.named_modules():
        if 'layer4' in nm:
            continue
        if 'blocks.9' in nm:
            continue
        if 'blocks.10' in nm:
            continue
        if 'blocks.11' in nm:
            continue
        if 'norm.' in nm:
            continue
        if nm in ['norm']:
            continue
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            for (np, p) in m.named_parameters():
                if np in ['weight', 'bias']:
                    params.append(p)
                    names.append(f'{nm}.{np}')
    return (params, names)

def configure_model(model):
    """Configure model."""
    model.eval()
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        elif isinstance(m, nn.BatchNorm1d):
            m.train()
            m.requires_grad_(True)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            m.requires_grad_(True)
    return model

def setup(model, cfg):
    model = configure_model(model)
    (params, parsam_names) = collect_params(model)
    if cfg.OPTIM.METHOD == 'SGD':
        optimizer = torch.optim.SGD(params, lr=float(cfg.OPTIM.LR), momentum=float(cfg.OPTIM.MOMENTUM), weight_decay=float(cfg.OPTIM.WD))
    elif cfg.OPTIM.METHOD == 'Adam':
        optimizer = torch.optim.Adam(params, lr=float(cfg.OPTIM.LR), weight_decay=float(cfg.OPTIM.WD))
    TTA_model = ROID(cfg, model, optimizer)
    return TTA_model
