"""Entropy-minimization adapter with normalization-affine updates."""
from copy import deepcopy
from typing import List, Set
import torch
from torch import nn, optim
from loguru import logger as log

class TENT(nn.Module):
    def __init__(self, cfg, model, optimizer):
        super().__init__()
        self.model, self.optimizer, self.steps = model, optimizer, cfg.OPTIM.STEPS
        self.model_state, self.optimizer_state = copy_model_and_optimizer(model, optimizer)

    def forward(self, x, y=None, adapt=True):
        if not adapt:
            with torch.no_grad():
                return self.model(x)
        for _ in range(self.steps):
            output = self.forward_and_adapt(x)
        return output

    @torch.enable_grad()
    def forward_and_adapt(self, x):
        self.optimizer.zero_grad()
        output = self.model(x)
        softmax_entropy(output).mean().backward()
        self.optimizer.step()
        return output

    def reset(self):
        load_model_and_optimizer(self.model, self.optimizer, self.model_state, self.optimizer_state)

NORM_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)

_CONV_TYPES = (nn.Conv1d, nn.Conv2d, nn.Conv3d)

_ADAPT_ROLES = {'norm', 'attention', 'head', 'conv'}

_ADAPT_ALIASES = {'bn_only': 'norm', 'conv_only': 'conv', 'classifier': 'head', 'bn_attention': 'norm+attention', 'bn_classifier': 'norm+head', 'bn_conv': 'norm+conv', 'ln_attention': 'norm+attention'}

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

def _parse_adapt_params(adapt_params: str) -> Set[str]:
    value = _ADAPT_ALIASES.get(str(adapt_params).strip().lower(), str(adapt_params).strip().lower())
    roles = {part.strip() for part in value.split('+') if part.strip()}
    unsupported = roles - _ADAPT_ROLES
    if not roles or unsupported:
        raise ValueError(f'Unsupported Tent ADAPT_PARAMS: {adapt_params!r}')
    return roles

def _classifier_modules(model: nn.Module) -> List[nn.Module]:
    modules = []
    for root in (model, getattr(model, 'module', None)):
        if not isinstance(root, nn.Module):
            continue
        for name in ('heads', 'head', 'classifier', 'fc'):
            module = getattr(root, name, None)
            if isinstance(module, nn.Module) and all((id(module) != id(item) for item in modules)):
                modules.append(module)
    return modules

def _selected_parameter_ids(model: nn.Module, roles: Set[str]) -> Set[int]:
    selected = set()
    for module in model.modules():
        if 'norm' in roles and isinstance(module, NORM_TYPES):
            selected.update((id(param) for (name, param) in module.named_parameters(recurse=False) if name in {'weight', 'bias'}))
        if 'conv' in roles and isinstance(module, _CONV_TYPES):
            selected.update((id(param) for param in module.parameters(recurse=False)))
        if 'attention' in roles and (isinstance(module, nn.MultiheadAttention) or 'attention' in type(module).__name__.lower()):
            selected.update((id(param) for param in module.parameters()))
    if 'head' in roles:
        for module in _classifier_modules(model):
            selected.update((id(param) for param in module.parameters()))
    return selected

def collect_params(model, adapt_params='bn_only'):
    """Select Tent parameters by role (norm, conv, attention, or head)."""
    selected = _selected_parameter_ids(model, _parse_adapt_params(adapt_params))
    (params, names) = ([], [])
    for (name, p) in model.named_parameters():
        if id(p) in selected and p.requires_grad:
            params.append(p)
            names.append(name)
    return (params, names)

def copy_model_and_optimizer(model, optimizer):
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return (model_state, optimizer_state)

def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)

def configure_model(model, adapt_params='bn_only'):
    model.train()
    model.requires_grad_(False)
    roles = _parse_adapt_params(adapt_params)
    selected = _selected_parameter_ids(model, roles)
    for param in model.parameters():
        if id(param) in selected:
            param.requires_grad_(True)
    for m in model.modules():
        if 'norm' in roles and isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    return model

def setup(model, cfg):
    log.info('Setup TTA method: Tent')
    adapt_params = cfg.ADAPTER.Tent.ADAPT_PARAMS
    model = configure_model(model, adapt_params)
    (params, param_names) = collect_params(model, adapt_params)
    if cfg.OPTIM.METHOD == 'SGD':
        optimizer = optim.SGD(params, lr=float(cfg.OPTIM.LR), dampening=cfg.OPTIM.DAMPENING, momentum=float(cfg.OPTIM.MOMENTUM), weight_decay=float(cfg.OPTIM.WD), nesterov=cfg.OPTIM.NESTEROV)
    elif cfg.OPTIM.METHOD == 'Adam':
        optimizer = optim.Adam(params, lr=float(cfg.OPTIM.LR), betas=(cfg.OPTIM.BETA, 0.999), weight_decay=float(cfg.OPTIM.WD))
    TTA_model = TENT(cfg, model, optimizer)
    return TTA_model
