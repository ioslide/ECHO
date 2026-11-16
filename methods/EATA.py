"""
Builds upon: https://github.com/mr-eggplant/EATA
Corresponding paper: https://arxiv.org/abs/2204.02610
"""
import math
import logging
import os
from copy import deepcopy
import torch
import torch.nn as nn
import torch.jit
import torch.nn.functional as F
from loguru import logger as log
from core.data.data_loading import get_source_loader
from tqdm import tqdm
__all__ = ['setup']

def compute_fishers(model, params, cfg):
    batch_size_src = cfg.TEST.BATCH_SIZE if cfg.TEST.BATCH_SIZE > 1 else cfg.TEST.WINDOW_LENGTH
    (_, fisher_loader) = get_source_loader(dataset_name=cfg.CORRUPTION.DATASET, root_dir=cfg.DATA_DIR, adaptation=cfg.MODEL.ADAPTATION, batch_size=batch_size_src, num_samples=cfg.ADAPTER.EATA.NUM_SAMPLES, train_split=True, ckpt_path=cfg.MODEL.CKPT_PATH, percentage=1.0, workers=2)
    ewc_optimizer = torch.optim.SGD(params, 0.001)
    fishers = {}
    train_loss_fn = nn.CrossEntropyLoss().cuda()
    for (iter_, batch) in tqdm(enumerate(fisher_loader, start=1), total=len(fisher_loader)):
        images = batch[0].cuda(non_blocking=True)
        outputs = model(images)
        (_, targets) = outputs.max(1)
        loss = train_loss_fn(outputs, targets)
        loss.backward()
        for (name, param) in model.named_parameters():
            if param.grad is not None:
                if iter_ > 1:
                    fisher = param.grad.data.clone().detach() ** 2 + fishers[name][0]
                else:
                    fisher = param.grad.data.clone().detach() ** 2
                if iter_ == len(fisher_loader):
                    fisher = fisher / iter_
                fishers.update({name: [fisher, param.data.clone().detach()]})
        ewc_optimizer.zero_grad()
    log.info('compute fisher matrices finished')
    del ewc_optimizer
    return fishers

class EATA_model(nn.Module):
    """EATA adapts a model by entropy minimization during testing.
    Once EATAed, a model adapts itself by updating on every forward.
    """

    def __init__(self, cfg, model):
        super().__init__()
        self.cfg = cfg
        self.model = configure_model(model, eval_mode=False)
        (params, _) = collect_params(model)
        self.params = params
        self.loss_transform = None
        self.entropy_fn = softmax_entropy
        self.optimizer = torch.optim.SGD(params, lr=cfg.OPTIM.LR, momentum=cfg.OPTIM.BETA)
        self.fishers = compute_fishers(model, params, cfg)
        self.num_samples_update_1 = 0
        self.num_samples_update_2 = 0
        margin = 0.4
        self.e_margin = math.log(cfg.CORRUPTION.NUM_CLASS) * margin
        self.d_margin = cfg.ADAPTER.EATA.D_MARGIN
        self.current_model_probs = None
        self.fisher_alpha = cfg.ADAPTER.EATA.FISHER_ALPHA
        self.steps = cfg.OPTIM.STEPS
        assert self.steps > 0, 'eata requires >= 1 step(s) to forward and update'
        (self.model_state, self.optimizer_state) = copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x, y=None, adapt=True):
        if not adapt:
            with torch.no_grad():
                return self.model(x)
        for _ in range(self.steps):
            outputs = self.forward_and_adapt(x)
        return outputs

    @torch.enable_grad()
    def forward_and_adapt(self, x):
        """Forward and adapt model on batch of data.
        Measure entropy of the model prediction, take gradients, and update params.
        Return: model outputs
        """
        outputs = self.model(x)
        entropys = self.entropy_fn(outputs)
        filter_ids_1 = torch.where(entropys < self.e_margin)
        ids1 = filter_ids_1
        ids2 = torch.where(ids1[0] > -0.1)
        entropys = entropys[filter_ids_1]
        if self.current_model_probs is not None:
            cosine_similarities = F.cosine_similarity(self.current_model_probs.unsqueeze(dim=0), outputs[filter_ids_1].softmax(1), dim=1)
            filter_ids_2 = torch.where(torch.abs(cosine_similarities) < self.d_margin)
            entropys = entropys[filter_ids_2]
            ids2 = filter_ids_2
            updated_probs = update_model_probs(self.current_model_probs, outputs[filter_ids_1][filter_ids_2].softmax(1))
        else:
            updated_probs = update_model_probs(self.current_model_probs, outputs[filter_ids_1].softmax(1))
        coeff = 1 / torch.exp(entropys.clone().detach() - self.e_margin)
        entropys = entropys.mul(coeff)
        loss = entropys.mean(0)
        '\n        # implementation version 2, compute loss, forward all batch, forward and backward selected samples again.\n        # if x[ids1][ids2].size(0) != 0:\n        #     loss = softmax_entropy(model(x[ids1][ids2])).mul(coeff).mean(0) # reweight entropy losses for diff. samples\n        '
        if self.fishers is not None:
            ewc_loss = 0
            for (name, param) in self.model.named_parameters():
                if name in self.fishers:
                    ewc_loss += self.fisher_alpha * (self.fishers[name][0] * (param - self.fishers[name][1]) ** 2).sum()
            loss += ewc_loss
        if self.loss_transform is not None:
            loss = self.loss_transform(loss)
        if x[ids1][ids2].size(0) != 0:
            loss.backward()
            self.optimizer.step()
        else:
            outputs = outputs.detach()
        self.optimizer.zero_grad()
        self.num_samples_update_1 += filter_ids_1[0].size(0)
        self.num_samples_update_2 += entropys.size(0)
        self.reset_model_probs(updated_probs)
        return outputs

    def reset_steps(self, new_steps):
        self.steps = new_steps

    def reset_model_probs(self, probs):
        self.current_model_probs = probs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception('cannot reset without saved model/optimizer state')
        load_model_and_optimizer(self.model, self.optimizer, self.model_state, self.optimizer_state)
        self.current_model_probs = None

def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)

def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return (model_state, optimizer_state)

def collect_params(model):
    """Collect the affine scale + shift parameters from batch norms.
    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for (nm, m) in model.named_modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            for (np, p) in m.named_parameters():
                if np in ['weight', 'bias']:
                    params.append(p)
                    names.append(f'{nm}.{np}')
    return (params, names)

def setup_optimizer(params, cfg):
    if cfg.OPTIM.METHOD == 'SGD':
        return torch.optim.SGD(params, lr=cfg.OPTIM.LR, momentum=cfg.OPTIM.MOMENTUM, dampening=cfg.OPTIM.DAMPENING, weight_decay=cfg.OPTIM.WD, nesterov=cfg.OPTIM.NESTEROV)
    if cfg.OPTIM.METHOD in ('Adam', 'AdamW'):
        return getattr(torch.optim, cfg.OPTIM.METHOD)(params, lr=cfg.OPTIM.LR, betas=(cfg.OPTIM.BETA, 0.999), weight_decay=cfg.OPTIM.WD)
    raise ValueError(f'Unsupported optimizer: {cfg.OPTIM.METHOD}')

def configure_model(model, eval_mode=False):
    """Configure model for use with tent."""
    model.train(not eval_mode)
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            m.requires_grad_(True)
            if not eval_mode or isinstance(m, nn.BatchNorm2d):
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            elif isinstance(m, nn.BatchNorm1d):
                m.train()
    return model

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    temprature = 1
    x = x / temprature
    x = -(x.softmax(1) * x.log_softmax(1)).sum(1)
    return x

def update_model_probs(current_model_probs, new_probs):
    if current_model_probs is None:
        if new_probs.size(0) == 0:
            return None
        else:
            with torch.no_grad():
                return new_probs.mean(0)
    elif new_probs.size(0) == 0:
        with torch.no_grad():
            return current_model_probs
    else:
        with torch.no_grad():
            return 0.9 * current_model_probs + (1 - 0.9) * new_probs.mean(0)

def setup(model, cfg):
    log.info('Setup TTA method: EATA')
    model = EATA_model(cfg, model)
    return model
