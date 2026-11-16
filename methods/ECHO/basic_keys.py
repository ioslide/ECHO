"""Fixed, untuned input descriptors for the key-only ablations in Table 4."""

import math

import torch
import torch.nn.functional as F


def _gray(x):
    x = x.float()
    return (x * x.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)).sum(1)


class RGBMomentsKey:
    """Six raw RGB means and population standard deviations, before normalization."""

    has_source_stats = False

    @torch.no_grad()
    def __call__(self, x):
        x = x.float()
        return torch.cat([x.mean((0, 2, 3)), x.std((0, 2, 3), unbiased=False)])


class RGBHistogramKey:
    """Eight fixed equal-width bins on [0, 1] per channel; unit mass per channel."""

    has_source_stats = False

    @torch.no_grad()
    def __call__(self, x):
        channels = x.float().movedim(1, 0).flatten(1)
        return torch.cat([torch.histc(channel, bins=8, min=0, max=1) / channel.numel()
                          for channel in channels])


class GradientHistogramKey:
    """Eight fixed bins of grayscale forward-difference magnitude on [0, 1].

    Divide magnitude by sqrt(2), its maximum for [0, 1] inputs. Pooling equal-size
    images is equivalent to averaging their unit-mass histograms.
    """

    has_source_stats = False

    @torch.no_grad()
    def __call__(self, x):
        gray = _gray(x)
        dx = gray[:, :-1, 1:] - gray[:, :-1, :-1]
        dy = gray[:, 1:, :-1] - gray[:, :-1, :-1]
        magnitude = (dx.square() + dy.square()).sqrt() / math.sqrt(2)
        return torch.histc(magnitude, bins=8, min=0, max=1) / magnitude.numel()


class FrequencyEnergyKey:
    """Mean grayscale power in eight equal-width radial bands, excluding DC.

    Radii span [0, sqrt(0.5)] cycles/pixel. Band means are L1-normalized per
    image before batch averaging; no source network or fitted weighting.
    """

    has_source_stats = False

    @torch.no_grad()
    def __call__(self, x):
        gray = _gray(x)
        power = torch.fft.fft2(gray, norm='ortho').abs().square()
        power[:, 0, 0] = 0
        fy = torch.fft.fftfreq(gray.shape[-2], device=x.device)
        fx = torch.fft.fftfreq(gray.shape[-1], device=x.device)
        radius = (fy[:, None].square() + fx[None, :].square()).sqrt()
        bands = (radius.flatten() * 8 / math.sqrt(0.5)).long().clamp_max(7)
        counts = torch.bincount(bands, minlength=8).to(power)
        counts[0] -= 1  # DC contributes neither power nor a sample to the mean.
        means = power.new_zeros(gray.shape[0], 8)
        means.scatter_add_(1, bands.expand(gray.shape[0], -1), power.flatten(1))
        means /= counts.clamp_min(1)
        return F.normalize(means, p=1, dim=1).mean(0)
