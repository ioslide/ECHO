"""Plain leader clustering of source features with independent Tent checkpoints.

No ECHO keys, calibration, replay, shared updates or memory decay. The fixed
Euclidean radius is 0.2 on L2-normalized batch-mean penultimate features.
"""

from copy import deepcopy
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


def _cpu_copy(value):
    if torch.is_tensor(value):
        return value.detach().to('cpu', copy=True)
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    return deepcopy(value)


class FrozenSourceFeatures(nn.Module):
    """Ordinary L2-normalized batch-mean features, without shift statistics."""

    def __init__(self, source_model):
        super().__init__()
        self.net = deepcopy(source_model).eval().requires_grad_(False)
        classifier = [m for m in self.net.modules() if isinstance(m, nn.Linear)][-1]
        classifier.register_forward_pre_hook(self._record)

    def _record(self, module, inputs):
        self.features = inputs[0].detach()

    @torch.no_grad()
    def forward(self, x):
        self.net(x)
        return F.normalize(self.features.float().mean(0), dim=0)


class OnlineClustering(nn.Module):
    def __init__(self, adapter, source_model, diagnostic_path=''):
        super().__init__()
        self.adapter = adapter
        self.source = FrozenSourceFeatures(source_model).eval()
        self.radius = 0.2
        self.diagnostic_path = diagnostic_path
        self.initial = self._checkpoint()
        self.centres, self.counts, self.checkpoints = [], [], []
        self.active = None
        self.batches = 0

    @property
    def model(self):
        return self.adapter.model

    @property
    def optimizer(self):
        return self.adapter.optimizer

    def train(self, mode=True):
        super().train(mode)
        self.source.eval()
        return self

    @torch.no_grad()
    def _key(self, x):
        return self.source(x)

    def _checkpoint(self):
        return (_cpu_copy(self.model.state_dict()),
                _cpu_copy(self.optimizer.state_dict()))

    def _load(self, checkpoint):
        self.model.load_state_dict(checkpoint[0])
        # load_state_dict casts optimizer tensors to each parameter's device.
        self.optimizer.load_state_dict(deepcopy(checkpoint[1]))

    def _activate(self, index):
        if self.active == index:
            return
        if self.active is not None:
            self.checkpoints[self.active] = self._checkpoint()
        self._load(self.initial if index is None else self.checkpoints[index])
        self.active = index

    def forward(self, x, y=None, adapt=True):
        key = self._key(x)
        distances = ((torch.stack(self.centres) - key).square().sum(1)
                     if self.centres else key.new_empty(0))
        new = not self.centres or distances.min().item() > self.radius ** 2
        index = None if new else int(distances.argmin())
        if adapt and new:
            index = len(self.centres)
            self.centres.append(key.clone())
            self.counts.append(0)
            self.checkpoints.append(self.initial)
        self._activate(index)
        if not adapt:
            with torch.no_grad():
                return self.model(x)
        self.counts[index] += 1
        self.centres[index].add_((key - self.centres[index]) / self.counts[index])
        self.batches += 1
        return self.adapter(x, y=None, adapt=True)

    def reset(self):
        self.adapter.reset()
        self._load(self.initial)
        self.centres, self.counts, self.checkpoints = [], [], []
        self.active = None
        self.batches = 0

    def memory_control_diagnostics(self):
        return dict(protocol='plain_online_independent_v2',
                    key_type='l2_normalized_source_penultimate_batch_mean',
                    radius=self.radius, distance='euclidean',
                    n_states=len(self.centres), batches=self.batches,
                    calibration_batches=0, warmup_replay=False,
                    threshold_selection='fixed_before_evaluation',
                    state='full_model_and_optimizer_checkpoint',
                    checkpoint_storage='cpu', shared_state=False)

    def flush_diagnostics(self):
        self.adapter.flush_diagnostics()
        if self.diagnostic_path:
            path = Path(self.diagnostic_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(self.memory_control_diagnostics(), indent=2) + '\n')
