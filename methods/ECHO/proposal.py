"""Test-time adaptation as an associative memory over distribution shifts.

A test-time adapter changes its parameters by an update dtheta_t on every batch
x_t. Standard continual TTA applies every past update to every future batch:

    theta(x) = theta_0 + sum_t dtheta_t                      (kernel = 1)

which is why it forgets and drifts when domains alternate. A domain reservoir
applies an update only to batches of the same cluster (kernel in {0, 1}).
This wrapper applies each past update in proportion to how similar the
distribution it came from is to the current one:

    theta(x) = theta_0 + sum_t k(x_t, x) dtheta_t

with a Gaussian kernel k on fixed-bin RGB histograms of each input batch.
Network-statistic keys are retained as optional ablations. With random Fourier features phi (k(x, x') ~ phi(x)'
phi(x')) the sum is a fixed-size linear associative memory A (P x m):

    read   theta(x) = theta_0 + A phi(x)
    write  A <- A + dtheta phi(x)'

so adaptation is a write keyed by the shift, inference is a read, a revisited
shift recalls its own accumulated updates, similar shifts share them in
proportion to their similarity, and a shift unlike anything seen reads theta_0.
There are no domains, clusters, thresholds, checkpoints or switching rules.
The base TTA method (loss, sample selection, optimizer) is left unchanged.

Bandwidth h: by default twice the median distance between consecutive batches
over the first CALIBRATION batches (the batch-to-batch noise of the key), so
that batches of one distribution share almost all updates. With decay disabled,
h -> infinity gives continual TTA; h -> 0 gives episodic TTA.

Optional LOCAL_DECAY and GLOBAL_DECAY discount the distribution-keyed and
constant SHARE components of past writes, respectively, once per batch.
Both default to 1, preserving cumulative memory. SHARE=0 disables global decay.
"""

from copy import deepcopy
import json
import math
import os
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.utils import parameters_to_vector

__all__ = ['ECHOWrapper', 'wrap_adapter', 'find_echo', 'configure_from_note']

CALIBRATION = 79      # batches used for the bandwidth's median heuristic
KEY_DEPTH = 1 / 3     # fraction of the source network's BN layers read by the key


class _Stop(Exception):
    pass


NORMS = (_BatchNorm, nn.GroupNorm, nn.LayerNorm)


class ShiftKey:
    """Channel statistics of the batch at the early normalisation layers of
    the frozen source network: per channel (mu / sigma, sqrt(2) log sigma),
    the scale-free coordinates of a channel Gaussian (local Fisher-Rao
    geometry). BatchNorm and GroupNorm inputs are [B, C, ...] (channel dim 1);
    LayerNorm inputs are [B, ..., C] (channel dim -1). BatchNorm layers use the
    source running statistics to propagate the batch when the network still
    has them, batch statistics otherwise."""

    def __init__(self, source_model, depth=KEY_DEPTH, layers='first',
                 coords='fisher', stats='source'):
        if not 0 < depth <= 1:
            raise ValueError('ECHO.KEY_DEPTH must be in (0, 1]')
        if layers not in ('first', 'last', 'all'):
            raise ValueError('ECHO.KEY_LAYERS must be first, last or all')
        if coords not in ('fisher', 'raw') or stats not in ('source', 'batch'):
            raise ValueError('ECHO key coordinates must be fisher/raw and statistics source/batch')
        self.net = deepcopy(source_model).requires_grad_(False)
        norms = [m for m in self.net.modules() if isinstance(m, NORMS)]
        if not norms:
            raise RuntimeError('The key needs a network with BatchNorm, GroupNorm or LayerNorm layers.')
        bns = [m for m in norms if isinstance(m, _BatchNorm)]
        self.has_source_stats = stats == 'source' and all(m.running_mean is not None for m in bns)
        for m in bns:
            m.track_running_stats = self.has_source_stats
        self.net.train(not self.has_source_stats)
        n = len(norms) if layers == 'all' else math.ceil(len(norms) * depth)
        self.layers = norms[-n:] if layers == 'last' else norms[:n]
        self.coords = coords
        self._stats = {}
        for i, m in enumerate(self.layers):
            m.register_forward_hook(self._hook(i))

    def _hook(self, index):
        def record(mod, inputs, output):
            h = inputs[0].float()
            if isinstance(mod, nn.LayerNorm):
                h = h.movedim(-1, 1)
            dims = [0] + list(range(2, h.ndim))
            mean = h.mean(dims)
            std = (h.var(dims, unbiased=False) + mod.eps).sqrt()
            self._stats[index] = (torch.cat([mean, std]) if self.coords == 'raw' else
                                  torch.cat([mean / std, math.sqrt(2.0) * std.log()]))
            if len(self._stats) == len(self.layers):
                raise _Stop  # the rest of the network is not needed
        return record

    @torch.no_grad()
    def __call__(self, x):
        self.net.to(x.device)
        self._stats = {}
        try:
            self.net(x)
        except _Stop:
            pass
        s = torch.cat([self._stats[i] for i in range(len(self.layers))])
        return s / math.sqrt(s.numel() / 2)  # RMS over channels


class PixelKey:
    """Pixel degradation descriptor: channel moments, saturation, gradients
    and residual energy after pooling at scales 2, 4, 8 and 16. Inputs are
    the same image tensors as the source-network key, before normalization.
    Signed means are preserved; positive scales use logarithmic coordinates.
    """

    has_source_stats = False

    @torch.no_grad()
    def __call__(self, x):
        x = x.float()
        gray = x.mean(dim=1, keepdim=True)
        saturation = x.max(dim=1).values - x.min(dim=1).values
        means = x.mean(dim=(0, 2, 3))
        scales = [x.std(dim=(0, 2, 3), unbiased=False),
                  saturation.mean().reshape(1),
                  (gray[..., 1:] - gray[..., :-1]).abs().mean().reshape(1),
                  (gray[:, :, 1:, :] - gray[:, :, :-1, :]).abs().mean().reshape(1)]
        for size in (2, 4, 8, 16):
            low = F.interpolate(F.avg_pool2d(gray, size), size=gray.shape[-2:], mode='nearest')
            scales.append((gray - low).square().mean().sqrt().reshape(1))
        s = torch.cat([means, torch.cat(scales).clamp_min(1e-6).log()])
        return s / math.sqrt(s.numel() / 2)


class AssociativeMemory:
    """Kernel-weighted memory of every state change the base adapter makes.

    The adapted state is the optimized parameters plus the optimizer's first
    moments (momentum / Adam exp_avg), when it keeps them: momentum belongs to
    the shift it was accumulated on as much as the parameters do. Second
    moments and step counters describe the optimizer's scale and stay global.
    """

    def __init__(self, adapter, key, bandwidth=0.0, features=256, share=0.0,
                 params_only=False, seed=0, bandwidth_scale=1.0, calibration=CALIBRATION,
                 prior='source', index='kernel', global_decay=1.0, local_decay=1.0):
        self.adapter, self.key = adapter, key
        groups = adapter.optimizer.param_groups
        self.params = list(dict.fromkeys(p for g in groups for p in g['params']))
        if not self.params:
            raise RuntimeError('The adapter optimizer has no parameters.')
        if not 0.0 <= float(share) <= 1.0:
            raise ValueError('ECHO.SHARE must be in [0, 1]')
        if not 0.0 <= float(global_decay) <= 1.0:
            raise ValueError('ECHO.GLOBAL_DECAY must be in [0, 1]')
        if not 0.0 <= float(local_decay) <= 1.0:
            raise ValueError('ECHO.LOCAL_DECAY must be in [0, 1]')
        if features < 1 or calibration < 2 or bandwidth_scale <= 0 or bandwidth < 0:
            raise ValueError('ECHO requires FEATURES >= 1, CALIBRATION >= 2, '
                             'BANDWIDTH_SCALE > 0 and BANDWIDTH >= 0')
        if prior not in ('source', 'live') or index not in ('kernel', 'oracle', 'gate'):
            raise ValueError('ECHO requires PRIOR=source/live and INDEX=kernel/oracle/gate')
        self.theta0 = parameters_to_vector(self.params).detach().clone()
        self.fixed_bandwidth, self.m, self.seed = float(bandwidth), int(features), seed
        self.share, self.params_only = float(share), bool(params_only)
        self.global_decay = float(global_decay)
        self.local_decay = float(local_decay)
        self.bandwidth_scale, self.calibration = float(bandwidth_scale), int(calibration)
        self.prior, self.index = prior, index
        self.reset()

    # ------------------------------------------------------------------ state
    def _first_moments(self):
        state = self.adapter.optimizer.state
        moments = []
        for p in self.params:
            st = state.get(p, {})
            t = st.get('momentum_buffer', st.get('exp_avg'))
            if not torch.is_tensor(t) or t.shape != p.shape:
                return None
            moments.append(t)
        return moments

    def _tensors(self):
        return self.params + (self.moments or [])

    def _get(self):
        return torch.cat([t.detach().flatten() for t in self._tensors()])

    def _set(self, vec):
        """Copy a flat state vector into the live tensors in place (the
        optimizer keeps its references)."""
        i = 0
        with torch.no_grad():
            for t in self._tensors():
                n = t.numel()
                t.copy_(vec[i:i + n].view_as(t))
                i += n

    def reset(self):
        self.moments = None
        self._set(self.theta0)
        self.A = None                 # state x features memory, built once h is known
        self.v0 = None
        self.h = None
        self.buffer = []              # (key, parameter update) before calibration
        self.prev_key = None
        self.domain = None
        self.mass = None              # sum of write features, for the live-prior intervention
        self.parts = {}               # true domain -> state, for the partition-kernel control
        self.key_stats = {}           # true domain -> online key mean and squared deviations
        self.gate_log = None
        self.gate_batches, self.gate_live = 0, 0
        self.unfamiliar_sum, self.unfamiliar_n = 0.0, 0
        self.steps, self.kappa_sum, self.kappa_min = 0, 0.0, 1.0

    def set_context(self, domain, round_idx=None):
        # Domain labels are used only by the oracle and by noise diagnostics.
        self.domain = domain

    # ------------------------------------------------------------------ kernel
    def _features(self, s):
        phi = torch.cos(s @ self.W + self.b)
        phi = phi / phi.norm()
        if self.share > 0:  # k' = c + (1 - c) k; c = 1, rho = 1 gives continual TTA
            phi = torch.cat([phi * math.sqrt(1 - self.share), phi.new_full((1,), math.sqrt(self.share))])
        return phi.to(self.theta0.dtype)

    def _decay_memory(self):
        # The first m features index the shift; the optional last one is constant.
        if self.local_decay != 1.0:
            self.A[:, :self.m].mul_(self.local_decay)
            if self.mass is not None:
                self.mass[:self.m].mul_(self.local_decay)
        if self.share > 0 and self.global_decay != 1.0:
            self.A[:, -1].mul_(self.global_decay)
            if self.mass is not None:
                self.mass[-1].mul_(self.global_decay)

    def _calibrate(self, keys):
        if self.fixed_bandwidth > 0:
            self.h = self.fixed_bandwidth
        else:
            gaps = torch.stack([(b - a).norm() for a, b in zip(keys[:-1], keys[1:])])
            self.h = 2.0 * gaps.median().item() * self.bandwidth_scale
        d, dev = keys[0].numel(), keys[0].device
        g = torch.Generator(device='cpu').manual_seed(self.seed)
        self.W = (torch.randn(d, self.m, generator=g) / self.h).to(dev)
        self.b = (torch.rand(self.m, generator=g) * 2 * math.pi).to(dev)
        self.moments = None if self.params_only else self._first_moments()
        P = self.theta0.numel()
        self.v0 = torch.cat([self.theta0, torch.zeros(self._get().numel() - P, device=self.theta0.device,
                                                      dtype=self.theta0.dtype)])
        self.A = torch.zeros(self.v0.numel(), self.m + (self.share > 0), device=self.v0.device,
                             dtype=self.v0.dtype)
        if self.prior == 'live':
            self.mass = self.A.new_zeros(self.A.shape[1])
        for s, delta in self.buffer:  # replay the calibration updates under their own keys
            phi = self._features(s)
            self._decay_memory()
            self.A[:P].addr_(delta, phi)
            if self.mass is not None:
                self.mass += phi
        if self.moments is not None:
            self.A[P:].addr_(self._get()[P:], self._features(keys[-1]))
        self.buffer = []

    # ------------------------------------------------------------------ stream
    def read(self, s):
        """State for key s and its features; (None, None) while calibrating."""
        if self.A is None:
            return None, None
        phi = self._features(s)
        return self.v0 + self.A @ phi, phi

    def _track_key(self, s):
        if self.domain is None:
            return
        if self.domain not in self.key_stats:
            self.key_stats[self.domain] = [1, s.clone(), 0.0]
            return
        st = self.key_stats[self.domain]
        st[0] += 1
        delta = s - st[1]
        st[1] += delta / st[0]
        st[2] += delta.dot(s - st[1]).item()

    @torch.no_grad()
    def _logits(self, x):
        predict = getattr(self.adapter, 'predict_with_echo', None)
        return predict(x) if callable(predict) else self.adapter.model(x)

    def _notify_read(self):
        hook = getattr(self.adapter, 'on_echo_read', None)
        if callable(hook):
            hook()

    @torch.no_grad()
    def _entropy_and_pred(self, x):
        logits = self._logits(x)
        entropy = -(logits.float().softmax(1) * logits.float().log_softmax(1)).sum(1).mean()
        return entropy.item(), logits.argmax(1).detach().cpu()

    def _select_state(self, x, v, phi, record=False):
        if self.prior == 'live':
            # Continuous intervention: unfamiliarity is the missing kernel mass.
            # Zero familiarity reads the live state; mass >= 1 keeps the normal read.
            u = (1.0 - self.mass.dot(phi)).clamp(0.0, 1.0).item()
            v = (1.0 - u) * v + u * self._get()
            if record:
                self.unfamiliar_sum += u
                self.unfamiliar_n += 1
        if self.index == 'gate':
            live = self._get()
            ent_live, pred_live = self._entropy_and_pred(x)
            self._set(v)
            ent_rec, pred_rec = self._entropy_and_pred(x)
            chose_live = ent_live <= ent_rec
            if chose_live:
                v = live
            if record:
                self.gate_batches += 1
                self.gate_live += int(chose_live)
                self.gate_log = dict(chose_live=chose_live, pred_live=pred_live,
                                     pred_rec=pred_rec, ent_live=ent_live, ent_rec=ent_rec)
        return v

    def predict(self, x):
        if self.index == 'oracle':
            if self.v0 is not None:
                self._set(self.parts.get(self.domain, self.v0))
            with torch.no_grad():
                return self._logits(x)
        v, phi = self.read(self.key(x))
        if v is not None:
            self._set(self._select_state(x, v, phi))
        # Some base adapters ignore adapt=False; call the network directly.
        with torch.no_grad():
            return self._logits(x)

    def forward(self, x):
        if self.index == 'oracle':
            return self._forward_oracle(x)
        s = self.key(x)
        self._track_key(s)
        self.gate_log = None
        v, phi = self.read(s)
        if v is not None:
            v = self._select_state(x, v, phi, record=True)
            self._set(v)
            self._notify_read()
        before = parameters_to_vector(self.params).detach().clone()
        out = self.adapter(x, y=None, adapt=True)
        if phi is None:  # calibration: plain continual TTA, updates kept for replay
            self.buffer.append((s, parameters_to_vector(self.params).detach() - before))
            if len(self.buffer) >= self.calibration:
                self._calibrate([k for k, _ in self.buffer])
        else:
            self._decay_memory()
            self.A.addr_(self._get() - v, phi)  # write: the change, keyed by the shift
            if self.mass is not None:
                self.mass += phi
            if self.prev_key is not None:
                kappa = torch.dot(phi, self._features(self.prev_key)).item()
                self.kappa_sum += kappa
                self.kappa_min = min(self.kappa_min, kappa)
        self.prev_key = s
        self.steps += 1
        return out

    def _forward_oracle(self, x):
        if self.domain is None:
            raise RuntimeError('ECHO oracle requires set_context(domain) before each domain visit')
        if self.v0 is not None:
            self._set(self.parts.get(self.domain, self.v0))
            self._notify_read()
        out = self.adapter(x, y=None, adapt=True)
        if self.v0 is None:
            self.v0 = self.theta0.clone()
        if self.moments is None and not self.params_only:
            self.moments = self._first_moments()
            if self.moments is not None:
                # EATA may skip its first batches. Extend earlier source states
                # only when the optimizer actually initializes its first moments.
                zeros = self.theta0.new_zeros(sum(t.numel() for t in self.moments))
                self.v0 = torch.cat([self.theta0, zeros])
                self.parts = {dom: torch.cat([v, zeros]) for dom, v in self.parts.items()}
        self.parts[self.domain] = self._get()
        self.steps += 1
        return out

    def diagnostics(self):
        n = max(self.steps - self.calibration, 1)
        variances = [st[2] / (st[0] - 1) for st in self.key_stats.values() if st[0] > 1]
        noise = math.sqrt(sum(variances) / len(variances)) if variances else None
        memories = [self.A] if self.A is not None else list(self.parts.values())
        return {
            'schema': 'ASSOCIATIVE_MEMORY_V2',
            'bandwidth': self.h,
            'features': self.m,
            'share': self.share,
            'global_decay': self.global_decay,
            'local_decay': self.local_decay,
            'moments_in_memory': self.moments is not None,
            'key_dim': None if self.A is None else self.W.shape[0],
            'source_statistics': self.key.has_source_stats,
            'memory_shape': None if self.A is None else list(self.A.shape),
            'batches': self.steps,
            'memory_bytes': sum(t.numel() * t.element_size() for t in memories),
            'key_noise_rms': noise,
            'key_noise_over_h': None if noise is None or self.h is None else noise / self.h,
            'gate_batches': self.gate_batches,
            'gate_live_chosen': self.gate_live,
            'mean_unfamiliarity': self.unfamiliar_sum / max(self.unfamiliar_n, 1),
            'oracle_domains': len(self.parts),
            'mean_kappa_consecutive': self.kappa_sum / n,
            'min_kappa_consecutive': self.kappa_min,
        }


def _note_options(cfg):
    types = dict(FEATURES=int, BANDWIDTH=float, BANDWIDTH_SCALE=float,
                 CALIBRATION=int, SHARE=float, GLOBAL_DECAY=float, LOCAL_DECAY=float,
                 KEY_DEPTH=float, KEY_LAYERS=str,
                 KEY_COORDS=str, KEY_STATS=str, KEY_TYPE=str, PRIOR=str, INDEX=str, SEED=int)
    options = {name: getattr(cfg.ADAPTER.ECHO, name) for name in types}
    options['BATCH_SIZE'] = cfg.TEST.BATCH_SIZE
    tokens = cfg.NOTE.split()
    for token in tokens:
        if token.startswith('ECHO_') and '=' in token:
            name, value = token[5:].split('=', 1)
            if name == 'BANDWIDTH_FROM':
                options['BANDWIDTH'] = float(json.loads(Path(value).read_text())['bandwidth'])
                continue
            if name == 'BATCH_SIZE':
                options[name] = int(value)
                continue
            if name not in types:
                raise ValueError(f'Unknown ECHO NOTE option: {name}')
            options[name] = types[name](value)
    options['PARAMS_ONLY'] = cfg.ADAPTER.ECHO.PARAMS_ONLY or 'PARAMS_ONLY' in tokens
    return options


def configure_from_note(cfg):
    """Apply NOTE overrides to the mutable config before model/data setup."""
    for name, value in _note_options(cfg).items():
        if name == 'BATCH_SIZE':
            cfg.TEST.BATCH_SIZE = value
        else:
            setattr(cfg.ADAPTER.ECHO, name, value)
    if 'ECHO_OFF' in cfg.NOTE.split():
        cfg.ADAPTER.ECHO.ENABLED = False


class ECHOWrapper(nn.Module):
    """Drop-in wrapper around a test-time adapter exposing ``model`` and
    ``optimizer`` and called as ``adapter(x, y=None, adapt=True)``.

    Pass ``source_model`` (the unadapted network with its BN running
    statistics) when the adapter's own model has dropped them; otherwise the
    key propagates batches with batch statistics.
    """

    def __init__(self, cfg, adapter, source_model=None):
        super().__init__()
        self.adapter = adapter
        self.note = cfg.NOTE
        self.options = opt = _note_options(cfg)
        opt['PARAMS_ONLY'] = opt['PARAMS_ONLY'] or bool(getattr(adapter, 'echo_params_only', False))
        if opt['KEY_TYPE'] in ('rgb-moments', 'rgb-histogram'):
            from .basic_keys import RGBMomentsKey, RGBHistogramKey
            key = RGBMomentsKey() if opt['KEY_TYPE'] == 'rgb-moments' else RGBHistogramKey()
        elif opt['KEY_TYPE'] == 'gradient-histogram':
            from .basic_keys import GradientHistogramKey
            key = GradientHistogramKey()
        elif opt['KEY_TYPE'] == 'frequency-energy':
            from .basic_keys import FrequencyEnergyKey
            key = FrequencyEnergyKey()
        elif opt['KEY_TYPE'] == 'feature-mean':
            from .simple_memories import FrozenFeatureKey
            key = FrozenFeatureKey(source_model if source_model is not None else adapter.model)
        elif opt['KEY_TYPE'] == 'higher-order-plain':
            from .complex_keys import PlainHigherOrderKey
            key = PlainHigherOrderKey(source_model if source_model is not None else adapter.model)
        elif opt['KEY_TYPE'] == 'correlation-plain':
            from .complex_keys import PlainCorrelationKey
            key = PlainCorrelationKey(source_model if source_model is not None else adapter.model)
        elif opt['KEY_TYPE'] == 'higher-order':
            from .complex_keys import HigherOrderKey
            key = HigherOrderKey(source_model if source_model is not None else adapter.model,
                                 depth=opt['KEY_DEPTH'], stats=opt['KEY_STATS'])
        elif opt['KEY_TYPE'] == 'covariance':
            from .complex_keys import CovarianceKey
            key = CovarianceKey(source_model if source_model is not None else adapter.model,
                                depth=opt['KEY_DEPTH'], stats=opt['KEY_STATS'], seed=opt['SEED'])
        elif opt['KEY_TYPE'] in ('network', 'pixel'):
            key = (PixelKey() if opt['KEY_TYPE'] == 'pixel' else
                   ShiftKey(source_model if source_model is not None else adapter.model,
                            depth=opt['KEY_DEPTH'], layers=opt['KEY_LAYERS'],
                            coords=opt['KEY_COORDS'], stats=opt['KEY_STATS']))
        else:
            raise ValueError('Unknown ECHO.KEY_TYPE: ' + opt['KEY_TYPE'])
        self.memory = AssociativeMemory(
            adapter, key,
            bandwidth=opt['BANDWIDTH'], features=opt['FEATURES'], share=opt['SHARE'],
            params_only=opt['PARAMS_ONLY'], seed=opt['SEED'],
            bandwidth_scale=opt['BANDWIDTH_SCALE'], calibration=opt['CALIBRATION'],
            prior=opt['PRIOR'], index=opt['INDEX'],
            global_decay=opt['GLOBAL_DECAY'],
            local_decay=opt['LOCAL_DECAY'],
        )
        self.diagnostic_path = cfg.ADAPTER.ECHO.DIAGNOSTIC_PATH

    @property
    def model(self):
        return self.adapter.model

    @property
    def optimizer(self):
        return self.adapter.optimizer

    def set_context(self, domain, round_idx=None):
        self.memory.set_context(domain, round_idx)

    def diagnostics(self):
        return dict(self.memory.diagnostics(), note=self.note, options=self.options)

    def forward(self, x, y=None, adapt=True):
        if not adapt:
            return self.memory.predict(x)
        return self.memory.forward(x)

    def reset(self):
        reset = getattr(self.adapter, 'reset', None)
        if callable(reset):
            reset()
        self.memory.reset()

    def finalize(self):
        finalize = getattr(self.adapter, 'finalize', None)
        if callable(finalize):
            finalize()

    def flush_diagnostics(self):
        flush = getattr(self.adapter, 'flush_diagnostics', None)
        if callable(flush):
            flush()
        configured = os.environ.get('ECHO_DIAGNOSTIC_PATH', self.diagnostic_path)
        if not configured:
            return
        path = Path(configured)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.diagnostics(), indent=2, sort_keys=True) + '\n',
                        encoding='ascii')


def find_echo(model):
    return next((m for m in model.modules() if isinstance(m, ECHOWrapper)), None)


def wrap_adapter(adapter, cfg, source_model=None):
    if not cfg.ADAPTER.ECHO.ENABLED or 'ECHO_OFF' in cfg.NOTE.split():
        return adapter
    return ECHOWrapper(cfg, adapter, source_model)
