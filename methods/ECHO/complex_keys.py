"""Alternative shift descriptors using the existing frozen source model.

Plain moments use ordinary penultimate features. Augmented descriptors remain
available to reproduce the earlier, separately tagged experiments.
"""

import math
from copy import deepcopy

import torch
from torch import nn

from .proposal import ShiftKey, _Stop
from .simple_memories import FrozenFeatureKey


class PlainHigherOrderKey(FrozenFeatureKey):
    """Raw mean/std/skewness/excess kurtosis of source penultimate features."""

    @torch.no_grad()
    def __call__(self, x):
        self.net.to(x.device)(x)
        h = self.features.float()
        mean = h.mean(0)
        std = (h.var(0, unbiased=False) + 1e-6).sqrt()
        normalized = (h - mean) / std
        squared = normalized.square()
        skewness = (squared * normalized).mean(0)
        kurtosis = squared.square().mean(0) - 3
        key = torch.cat([mean, std, skewness, kurtosis])
        return key / math.sqrt(key.numel())


class PlainCorrelationKey:
    """Pearson correlations of all channels at the first source convolution."""

    has_source_stats = True

    def __init__(self, source_model):
        self.net = deepcopy(source_model).eval().requires_grad_(False)
        convolution = next(m for m in self.net.modules() if isinstance(m, nn.Conv2d))
        convolution.register_forward_hook(self._record)

    def _record(self, module, inputs, output):
        self.features = output.detach()
        raise _Stop

    @torch.no_grad()
    def __call__(self, x):
        self.net.to(x.device)
        try:
            self.net(x)
        except _Stop:
            pass
        h = self.features.float().movedim(1, 0).flatten(1)
        centered = h - h.mean(1, keepdim=True)
        covariance = centered @ centered.T / centered.shape[1]
        std = covariance.diag().clamp_min(1e-6).sqrt()
        correlation = covariance / std[:, None] / std[None, :]
        upper = torch.triu_indices(h.shape[0], h.shape[0], 1, device=h.device)
        key = correlation[upper[0], upper[1]]
        return key / math.sqrt(key.numel())


class HigherOrderKey(ShiftKey):
    """Default key plus standardized third/fourth moments, equally weighted."""

    def __init__(self, source_model, depth=1 / 3, stats='source'):
        self._higher = {}
        super().__init__(source_model, depth=depth, stats=stats)

    def _hook(self, index):
        base_hook = super()._hook(index)

        def record(module, inputs, output):
            h = inputs[0].float()
            if isinstance(module, nn.LayerNorm):
                h = h.movedim(-1, 1)
            dims = [0] + list(range(2, h.ndim))
            mean = h.mean(dims, keepdim=True)
            std = (h.var(dims, unbiased=False, keepdim=True) + module.eps).sqrt()
            normalized = (h - mean) / std
            squared = normalized.square()
            skewness = (squared * normalized).mean(dims)
            kurtosis = squared.square().mean(dims) - 3
            self._higher[index] = torch.cat([skewness, kurtosis])
            base_hook(module, inputs, output)
        return record

    @torch.no_grad()
    def __call__(self, x):
        self._higher = {}
        base = super().__call__(x)
        extra = torch.cat([self._higher[index] for index in range(len(self.layers))])
        return torch.cat([base, extra / math.sqrt(extra.numel())]) / math.sqrt(2)


class CovarianceKey(ShiftKey):
    """Default Fisher key plus off-diagonal correlations in a fixed 16-D sketch.

    Each layer uses at most 64 uniformly spaced spatial/token positions per
    image. Both blocks have RMS normalization and fixed equal weight.
    """

    def __init__(self, source_model, depth=1 / 3, stats='source', seed=0):
        self.seed, self.rank, self.positions = seed, 16, 64
        self.projections, self._correlations = {}, {}
        super().__init__(source_model, depth=depth, stats=stats)

    def _hook(self, index):
        base_hook = super()._hook(index)

        def record(module, inputs, output):
            h = inputs[0].float()
            if isinstance(module, nn.LayerNorm):
                h = h.movedim(-1, 1)
            h = h.reshape(h.shape[0], h.shape[1], -1)
            positions = torch.linspace(0, h.shape[-1] - 1,
                                       min(self.positions, h.shape[-1]), device=h.device).long()
            sample = h[:, :, positions].permute(1, 0, 2).flatten(1)
            channels = sample.shape[0]
            if index not in self.projections:
                generator = torch.Generator(device='cpu').manual_seed(self.seed + index)
                projection = torch.randn(channels, min(self.rank, channels), generator=generator)
                self.projections[index] = projection.to(h) / math.sqrt(channels)
            projected = self.projections[index].T @ sample
            centered = projected - projected.mean(1, keepdim=True)
            covariance = centered @ centered.T / centered.shape[1]
            std = covariance.diag().clamp_min(1e-6).sqrt()
            correlation = covariance / std[:, None] / std[None, :]
            upper = torch.triu_indices(correlation.shape[0], correlation.shape[1], 1,
                                       device=h.device)
            self._correlations[index] = correlation[upper[0], upper[1]]
            base_hook(module, inputs, output)
        return record

    @torch.no_grad()
    def __call__(self, x):
        self._correlations = {}
        base = super().__call__(x)
        extra = torch.cat([self._correlations[index] for index in range(len(self.layers))])
        return torch.cat([base, extra / math.sqrt(extra.numel())]) / math.sqrt(2)
