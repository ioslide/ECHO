"""Simple-memory controls for paper Table 4 and the granularity appendix.

Matched-key controls reuse ECHO's keys; plain k-means uses ordinary frozen
penultimate features. All controls keep the base adapter and optimizer.
"""

import math
from copy import deepcopy
import torch
from torch import nn
from torch.nn.utils import parameters_to_vector

from .proposal import AssociativeMemory


class FrozenFeatureKey:
    """Batch mean of ordinary source features at the final linear classifier."""

    def __init__(self, source_model):
        self.net = deepcopy(source_model).eval().requires_grad_(False)
        self.has_source_stats = True
        classifier = [m for m in self.net.modules() if isinstance(m, nn.Linear)][-1]
        classifier.register_forward_pre_hook(self._record)

    def _record(self, module, inputs):
        self.features = inputs[0].detach()

    @torch.no_grad()
    def __call__(self, x):
        self.net.to(x.device)(x)
        s = self.features.float().mean(0)
        self.dim = s.numel()
        return s


class LinearMemory(AssociativeMemory):
    def _features(self, s):
        phi = s @ self.W
        phi = phi / phi.norm()
        if self.share > 0:
            phi = torch.cat([math.sqrt(1 - self.share) * phi,
                             phi.new_tensor([math.sqrt(self.share)])])
        return phi.to(self.theta0.dtype)

    def diagnostics(self):
        return dict(super().diagnostics(), memory_kind='linear')


class DiscreteMemory(AssociativeMemory):
    def reset(self):
        super().reset()
        self.centres, self.counts, self.values, self.touched = [], [], [], []
        self.basis = None
        self.shared = None
        self.capacity = self.n_clusters

    def _calibrate(self, keys):
        gaps = torch.stack([(b - a).norm() for a, b in zip(keys[:-1], keys[1:])])
        self.h = self.fixed_bandwidth or 2 * gaps.median().item() * self.bandwidth_scale
        self.moments = None if self.params_only else self._first_moments()
        p = self.theta0.numel()
        self.v0 = torch.cat([self.theta0, self.theta0.new_zeros(self._get().numel() - p)])
        if self.kind in ('slot', 'pca'):
            self.capacity = self.m + 1
        if self.kind == 'pca':
            self.rank = min(16, self.m + 1, self.v0.numel())
            generator = torch.Generator(device='cpu').manual_seed(self.seed)
            basis = torch.randn(self.v0.numel(), self.rank, generator=generator).to(self.v0)
            self.basis = torch.linalg.qr(basis, mode='reduced')[0]
            budget = (self.m + 1) * self.v0.numel()
            self.capacity = (budget - self.basis.numel()) // (keys[0].numel() + self.rank)
            if self.capacity < 1:
                raise ValueError('PCA basis and one slot exceed the matched memory budget')
        if self.kind == 'kmeans-shared':
            self.shared = torch.zeros_like(self.v0)
        # Same warm-up parameter updates; first moments are seeded at its last key.
        for s, delta in self.buffer:
            v, indices, weights = self._read(s, create=True)
            update = torch.zeros_like(self.v0)
            update[:p] = delta
            self._write(update, indices, weights)
        if self.moments is not None:
            _, indices, weights = self._read(keys[-1], create=True)
            update = torch.zeros_like(self.v0)
            update[p:] = self._get()[p:]
            self._write(update, indices, weights)
        self.buffer = []

    def _decode(self, indices):
        values = torch.stack([self.values[i] for i in indices])
        return values @ self.basis.T if self.basis is not None else values

    def _new(self, s, initial):
        value = initial @ self.basis if self.basis is not None else initial.clone()
        if len(self.centres) < self.capacity or self.kind == 'online':
            self.centres.append(s.detach().clone())
            self.values.append(value)
            self.counts.append(1)
            self.touched.append(self.steps)
            return len(self.centres) - 1
        index = min(range(len(self.touched)), key=self.touched.__getitem__)
        self.centres[index] = s.detach().clone()
        self.values[index] = value
        self.counts[index] = 1
        self.touched[index] = self.steps
        return index

    def _read(self, s, create=False):
        if not self.centres:
            if not create:
                return self.v0, [], s.new_empty(0)
            index = self._new(s, torch.zeros_like(self.v0))
            return self.v0, [index], s.new_ones(1)
        distances = (torch.stack(self.centres) - s).square().sum(1)
        nearest = int(distances.argmin())
        if self.kind in ('online', 'kmeans-hard'):
            threshold = self.threshold if self.kind == 'online' else 1.0
            room = self.kind == 'online' or len(self.centres) < self.capacity
            if create and room and distances[nearest] > (threshold * self.h) ** 2:
                nearest = self._new(s, torch.zeros_like(self.v0))
            elif create:
                self.counts[nearest] += 1
                self.centres[nearest].add_((s - self.centres[nearest]) / self.counts[nearest])
            return self.v0 + self._decode([nearest])[0], [nearest], s.new_ones(1)
        if self.kind.startswith('kmeans'):
            if create and len(self.centres) < self.capacity and distances[nearest] > self.h ** 2:
                self._new(s, torch.zeros_like(self.v0))
                distances = (torch.stack(self.centres) - s).square().sum(1)
            elif create:
                self.counts[nearest] += 1
                self.centres[nearest].add_((s - self.centres[nearest]) / self.counts[nearest])
            indices = list(range(len(self.centres)))
            weights = (-distances / (2 * self.h ** 2)).softmax(0)
            local = weights @ self._decode(indices)
            if self.shared is not None:
                local = (1 - self.share) * local + self.share * self.shared
            return self.v0 + local, indices, weights
        # Slot and PCA: top-8 soft read, a new slot beyond h, LRU replacement.
        chosen = distances.topk(min(8, len(self.centres)), largest=False).indices.tolist()
        weights = (-distances[chosen] / (2 * self.h ** 2)).softmax(0)
        local = weights @ self._decode(chosen)
        if create and distances[nearest] > self.h ** 2:
            index = self._new(s, local)
            return self.v0 + local, [index], s.new_ones(1)
        return self.v0 + local, chosen, weights

    def _write(self, delta, indices, weights):
        if self.basis is not None and delta.norm() > 0:
            # Oja subspace update, reorthogonalized; rotate all stored coefficients.
            coeff = delta @ self.basis
            residual = delta - self.basis @ coeff
            candidate = self.basis + 0.1 * torch.outer(residual, coeff) / delta.square().sum()
            basis = torch.linalg.qr(candidate, mode='reduced')[0]
            rotation = self.basis.T @ basis
            self.values = list((torch.stack(self.values) @ rotation).unbind(0))
            self.basis = basis
        update = delta @ self.basis if self.basis is not None else delta
        for index, weight in zip(indices, weights):
            self.values[index].add_(weight * update)
            self.touched[index] = self.steps
        if self.shared is not None:
            # c is already applied by the read; this gives kernel c+(1-c)<a,a'>.
            self.shared.add_(delta)

    def forward(self, x):
        s = self.key(x)
        self._track_key(s)
        if self.v0 is None:
            before = parameters_to_vector(self.params).detach().clone()
            out = self.adapter(x, y=None, adapt=True)
            self.buffer.append((s, parameters_to_vector(self.params).detach() - before))
            if len(self.buffer) >= self.calibration:
                self._calibrate([k for k, _ in self.buffer])
        else:
            v, indices, weights = self._read(s, create=True)
            self._set(v)
            out = self.adapter(x, y=None, adapt=True)
            self._write(self._get() - v, indices, weights)
        self.steps += 1
        return out

    def predict(self, x):
        if self.v0 is not None:
            v, _, _ = self._read(self.key(x))
            self._set(v)
        with torch.no_grad():
            return self.adapter.model(x)

    def diagnostics(self):
        result = super().diagnostics()
        tensors = self.centres + self.values
        tensors += [t for t in (self.basis, self.shared) if t is not None]
        result = dict(result, memory_kind=self.kind, n_states=len(self.values),
                    n_clusters=self.n_clusters, threshold=self.threshold,
                    capacity=self.capacity,
                    pca_rank=getattr(self, 'rank', None), pca_oja_rate=0.1,
                    memory_bytes=sum(t.numel() * t.element_size() for t in tensors),
                    discrete_decay=1.0)
        return result


class PlainKMeansMemory(DiscreteMemory):
    """Fixed-K online clustering, without shift keys or automatic bandwidth.

    Fit k-means++ to past warm-up batches only. All states start at the same
    final warm-up checkpoint; no keyed replay of warm-up updates is used.
    Soft routing and centre updates use exp(-squared_distance / temperature).
    """

    def _calibrate(self, keys):
        from sklearn.cluster import KMeans

        if self.n_clusters > len(keys):
            raise ValueError('Plain k-means requires at least K warm-up batches')
        data = torch.stack(keys)
        fitted = KMeans(n_clusters=self.n_clusters, init='k-means++', n_init=10,
                        max_iter=300, random_state=self.seed).fit(data.cpu().numpy())
        centres = torch.as_tensor(fitted.cluster_centers_, device=data.device,
                                  dtype=data.dtype)
        if self.kind == 'plain-kmeans-hard':
            counts = torch.bincount(torch.as_tensor(fitted.labels_, device=data.device),
                                    minlength=self.n_clusters).to(data.dtype)
        else:
            weights = (-torch.cdist(data, centres).square() / self.temperature).softmax(1)
            counts = weights.sum(0)
            centres = (weights.T @ data) / counts[:, None]
        self.centres, self.counts = list(centres.unbind()), list(counts.unbind())
        self.moments = None if self.params_only else self._first_moments()
        current = self._get()
        self.v0 = torch.cat([self.theta0, current.new_zeros(current.numel() - self.theta0.numel())])
        self.values = [(current - self.v0).clone() for _ in self.centres]
        self.touched = [self.steps] * self.n_clusters
        if self.kind == 'plain-kmeans-shared':
            self.shared = (current - self.v0).clone()
        self.buffer = []

    def _read(self, s, create=False):
        distances = (torch.stack(self.centres) - s).square().sum(1)
        if self.kind == 'plain-kmeans-hard':
            indices = [int(distances.argmin())]
            weights = s.new_ones(1)
        else:
            indices = list(range(self.n_clusters))
            weights = (-distances / self.temperature).softmax(0)
        # Read and write use the same assignments, computed before centre updates.
        local = weights @ self._decode(indices)
        if self.shared is not None:
            local = (1 - self.share) * local + self.share * self.shared
        if create:
            for index, weight in zip(indices, weights):
                self.counts[index] = self.counts[index] + weight
                self.centres[index].add_(weight * (s - self.centres[index]) / self.counts[index])
        return self.v0 + local, indices, weights

    def diagnostics(self):
        return dict(super().diagnostics(), protocol='plain_kmeans_v1',
                    key_type='source_penultimate_batch_mean',
                    key_dim=getattr(self.key, 'dim', None), bandwidth=None,
                    threshold=None, features=None,
                    share=self.share if self.shared is not None else 0.0,
                    global_decay=1.0, local_decay=1.0,
                    temperature=None if self.kind == 'plain-kmeans-hard' else self.temperature,
                    initialization='k-means++', n_init=10, max_iter=300,
                    warmup_state='cloned_final_checkpoint', warmup_replay=False)


def replace_memory(wrapper, kind, n_clusters=15, threshold=1.0,
                   source_model=None, temperature=1.0):
    memory = wrapper.memory
    if kind in ('plain-kmeans-hard', 'plain-kmeans-soft', 'plain-kmeans-shared'):
        if source_model is None or temperature <= 0:
            raise ValueError('Plain k-means requires a frozen source model and temperature > 0')
        memory.kind, memory.n_clusters, memory.threshold = kind, n_clusters, None
        memory.temperature = float(temperature)
        memory.key = FrozenFeatureKey(source_model)
        memory.__class__ = PlainKMeansMemory
        memory.reset()
    elif kind == 'linear':
        memory.__class__ = LinearMemory
    elif kind in ('online', 'kmeans-hard', 'kmeans-soft', 'kmeans-shared', 'slot', 'pca'):
        memory.kind, memory.n_clusters, memory.threshold = kind, n_clusters, threshold
        memory.__class__ = DiscreteMemory
        memory.reset()
    else:
        raise ValueError('Unknown simple memory: ' + kind)
    return wrapper
