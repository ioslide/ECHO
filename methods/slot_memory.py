"""Plain FIFO slots: cosine top-k checkpoint read, one absolute write per batch.

No ECHO shift key, bandwidth, warm-up replay, shared state or delta writes.
The source backbone is shared; slots store all optimizer-owned parameters and
their original SGD momentum. Slot keys and states count against the budget.
"""

import json
from pathlib import Path

import torch
from torch import nn
from torch.nn.utils import parameters_to_vector

from .online_clustering import FrozenSourceFeatures


class SlotMemory(nn.Module):
    def __init__(self, adapter, source_model, features=4096, diagnostic_path=''):
        super().__init__()
        if not isinstance(adapter.optimizer, torch.optim.SGD):
            raise ValueError('Plain slot memory is configured for Tent with SGD')
        self.adapter = adapter
        self.source = FrozenSourceFeatures(source_model).eval()
        self.params = list(dict.fromkeys(p for group in self.optimizer.param_groups
                                         for p in group['params']))
        self.p = sum(p.numel() for p in self.params)
        self.budget = (features + 1) * 2 * self.p * self.params[0].element_size()
        self.top_k, self.temperature = 8, 1.0
        self.diagnostic_path = diagnostic_path
        self.initial = self._get().clone()
        self.reset()

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

    def _get(self):
        momentum = [self.optimizer.state.get(p, {}).get('momentum_buffer', torch.zeros_like(p))
                    for p in self.params]
        return parameters_to_vector(self.params + momentum).detach()

    @torch.no_grad()
    def _set(self, state):
        position = 0
        for p in self.params:
            count = p.numel()
            p.copy_(state[position:position + count].view_as(p))
            self.optimizer.state[p]['momentum_buffer'] = state[
                self.p + position:self.p + position + count].view_as(p).clone()
            position += count

    def reset(self):
        self.adapter.reset()
        self.keys, self.states = None, None
        self.capacity = None
        self.size = self.next_slot = self.batches = 0

    @torch.no_grad()
    def _read(self, key):
        if not self.size:
            return self.initial
        similarity, indices = (self.keys[:self.size] @ key).topk(min(self.top_k, self.size))
        weights = (similarity / self.temperature).softmax(0)
        states = self.states.index_select(0, indices.cpu()).to(key.device)
        return weights @ states

    def forward(self, x, y=None, adapt=True):
        key = self.source(x)
        if adapt and self.keys is None:
            slot_bytes = key.numel() * key.element_size() + self.initial.numel() * self.initial.element_size()
            self.capacity = self.budget // slot_bytes
            if self.capacity < 1:
                raise ValueError('Memory budget does not fit one key and state')
            self.keys = key.new_empty(self.capacity, key.numel())
            self.states = torch.empty(self.capacity, self.initial.numel(),
                                      dtype=self.initial.dtype, device='cpu')
        # Leave the optimizer pristine on the first adapted batch.
        if self.size or not adapt:
            self._set(self._read(key))
        if not adapt:
            with torch.no_grad():
                return self.model(x)
        output = self.adapter(x, y=None, adapt=True)
        self.keys[self.next_slot].copy_(key)
        self.states[self.next_slot].copy_(self._get().cpu())
        self.next_slot = (self.next_slot + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.batches += 1
        return output

    def memory_control_diagnostics(self):
        allocated = (0 if self.keys is None else
                     self.keys.numel() * self.keys.element_size() +
                     self.states.numel() * self.states.element_size())
        return dict(protocol='plain_slot_independent_v1',
                    key_type='l2_normalized_source_penultimate_batch_mean',
                    read='cosine_softmax_top_k', top_k=self.top_k,
                    temperature=self.temperature, write='append_post_step_absolute_state',
                    replacement='fifo', capacity=self.capacity, n_states=self.size,
                    batches=self.batches, calibration_batches=0, warmup_replay=False,
                    shared_state=False, bandwidth=None,
                    budget_bytes=self.budget, memory_bytes=allocated,
                    state='adapter_parameters_and_sgd_momentum', state_storage='cpu',
                    slot_budget_includes_keys=True)

    def flush_diagnostics(self):
        self.adapter.flush_diagnostics()
        if self.diagnostic_path:
            path = Path(self.diagnostic_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(self.memory_control_diagnostics(), indent=2) + '\n')
