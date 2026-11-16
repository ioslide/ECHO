"""Bounded, verified lossless pixel caching and one-domain CPU prefetch.

Float tensors are retained as uint8 only after EVERY value has been checked
against the float32 lookup table, including its bit representation. Decoding
uses that same table on the GPU, avoiding CPU/GPU division-rounding differences.
No model, random generator, or adaptation state is accessed by the worker.
"""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
import threading
import time
from typing import NamedTuple

from loguru import logger as log
import numpy as np
import torch

from .pt_stream import TensorCacheStream


PIXEL_LUT = np.arange(256, dtype=np.float32) / np.float32(255)


def cache_log(message):
    """Opt-in cache diagnostics; keep normal experiment output concise."""
    if os.environ.get("LCS_CACHE_VERBOSE", "0").lower() in {"1", "true", "yes"}:
        log.info(message)


class CachedDomain(NamedTuple):
    images: torch.Tensor
    labels: torch.Tensor
    decode: str


class MixedDomainCache:
    """Keep a fixed set of compact domains within an explicit byte budget.

    Once full, additional domains use the ordinary reader; cycling an LRU over
    all 37 domains would thrash. A single worker may prepare the next domain.
    Budgets include entries being built, since preparation is serialized.
    """

    def __init__(self, budget_bytes, reserve_bytes, headroom, n_examples=5000,
                 prefetch=True, disk_dir=None, batch_size=64):
        self.budget = int(budget_bytes)
        self.reserve = int(reserve_bytes)
        if self.budget < 0 or self.reserve < 0 or n_examples < 1 or batch_size < 1:
            raise ValueError("invalid mixed-domain cache budget or sample count")
        self.headroom = headroom
        self.n_examples = int(n_examples)
        self.batch_size = int(batch_size)
        self.disk_dir = Path(disk_dir).expanduser() if disk_dir else None
        self.entries = {}
        self.resident_bytes = 0
        self.stats = dict(ram_hits=0, disk_hits=0, encoded=0, fallbacks=0)
        self._next = {}
        self._pending = None
        self._closed = threading.Event()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mix-cache") if prefetch else None

    @staticmethod
    def key(data_path, label_path, external_uint8=False):
        return str(data_path), str(label_path), bool(external_uint8)

    def set_order(self, keys):
        keys = list(keys)
        self._next = {key: keys[(i + 1) % len(keys)] for i, key in enumerate(keys)}

    def _fits(self, size):
        available = self.headroom()
        # Allow an output tensor, its file pages, and bounded conversion scratch.
        return (self.resident_bytes + size <= self.budget and available is not None
                and available >= self.reserve + 2 * size + 256 * 1024**2)

    def get(self, data_path, label_path, external_uint8=False):
        key = self.key(data_path, label_path, external_uint8)
        if self._closed.is_set():
            return None
        prepared = None
        prefetched = False
        if self._pending is not None:
            previous_key, future = self._pending
            self._pending = None
            # Even unexpected order changes must not start a second allocation.
            result = future.result()
            if previous_key == key:
                prepared = result
                prefetched = True
        if key in self.entries:
            self.stats["ram_hits"] += int(prepared is None)
            prepared = self.entries[key]
        elif prepared is None and not prefetched:
            prepared = self._prepare(key)
        next_key = self._next.get(key)
        if (self._executor is not None and next_key is not None
                and next_key != key and next_key not in self.entries and not self._closed.is_set()):
            self._pending = (next_key, self._executor.submit(self._prepare, next_key))
        return prepared

    @staticmethod
    def _identity(data_path, label_path, n):
        data_stat, label_stat = data_path.stat(), label_path.stat()
        return dict(codec="float32_pixel_lut_v1", source=str(data_path),
                    source_size=data_stat.st_size, source_mtime_ns=data_stat.st_mtime_ns,
                    label=str(label_path), label_size=label_stat.st_size,
                    label_mtime_ns=label_stat.st_mtime_ns, samples=n)

    def _disk_path(self, key):
        if self.disk_dir is None:
            return None
        digest = hashlib.sha256(repr((key, self.n_examples)).encode()).hexdigest()
        return self.disk_dir / (digest + ".pt")

    def _read_disk(self, path, identity, shape, decode, size):
        if path is None or not path.exists() or path.stat().st_size > size + 65536:
            return None
        try:
            value = torch.load(path, map_location="cpu", weights_only=True)
            images, labels = value["images"], value["labels"]
            if (value["identity"] != identity or value["decode"] != decode
                    or images.dtype != torch.uint8 or tuple(images.shape) != shape
                    or labels.dtype != torch.int64 or tuple(labels.shape) != (shape[0],)
                    or not images.is_contiguous() or not labels.is_contiguous()):
                return None
            self.stats["disk_hits"] += 1
            return CachedDomain(images, labels, decode)
        except Exception as exc:
            log.warning(f"Ignoring invalid compact cache {path}: {exc}")
            return None
        finally:
            self._drop_file_pages(path)

    @staticmethod
    def _drop_file_pages(path):
        if hasattr(os, "posix_fadvise"):
            try:
                with Path(path).open("rb") as file:
                    os.posix_fadvise(file.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            except OSError:
                pass

    def _write_disk(self, path, entry, identity, size):
        if path is None or self._closed.is_set():
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            if shutil.disk_usage(path.parent).free < size + 256 * 1024**2:
                log.warning("Skipping compact disk cache: insufficient disk space")
                return
            with tempfile.TemporaryDirectory(dir=path.parent) as directory:
                temporary = Path(directory) / "cache.pt"
                torch.save(dict(images=entry.images, labels=entry.labels,
                                decode=entry.decode, identity=identity), temporary)
                with temporary.open("rb") as file:
                    os.fsync(file.fileno())
                # Each file is <1 GiB here; flush and release its file pages
                # before starting another domain, rather than accumulating dirty data.
                self._drop_file_pages(temporary)
                os.replace(temporary, path)
        except OSError as exc:
            log.warning(f"Compact disk cache unavailable ({exc}); retaining RAM cache only")

    def _prepare(self, key):
        if self._closed.is_set():
            return None
        started = time.perf_counter()
        data_path, label_path = map(Path, key[:2])
        try:
            images, labels = TensorCacheStream(data_path), TensorCacheStream(label_path)
            if (len(images.shape) != 4 or len(labels.shape) != 1 or len(images) != len(labels)
                    or len(images) < self.n_examples or labels.dtype.name != "int64"
                    or images.dtype.name != ("uint8" if key[2] else "float32")):
                return None
            shape = (self.n_examples,) + images.shape[1:]
            size = int(np.prod(shape)) + self.n_examples * 8
            if not self._fits(size):
                self.stats["fallbacks"] += 1
                cache_log(f"Mixed cache budget/pressure fallback: resident={self.resident_bytes / 1024**3:.2f} GiB")
                return None
            decode = "divide" if key[2] else "lut"
            identity = self._identity(data_path, label_path, self.n_examples)
            disk_path = self._disk_path(key)
            entry = self._read_disk(disk_path, identity, shape, decode, size)
            if entry is None:
                encoded = np.empty(shape, dtype=np.uint8)
                start = 0
                batches = images.batches(self.batch_size, self.n_examples)
                try:
                    for batch in batches:
                        if self._closed.is_set():
                            return None
                        original = batch.numpy()
                        if key[2]:
                            codes = original
                        else:
                            scaled = original * np.float32(255)
                            np.rint(scaled, out=scaled)
                            if not np.isfinite(scaled).all() or scaled.min() < 0 or scaled.max() > 255:
                                return None
                            codes = scaled.astype(np.uint8)
                            restored = PIXEL_LUT[codes]
                            if not np.array_equal(restored.view(np.uint32), original.view(np.uint32)):
                                cache_log(f"Pixel encoding is not lossless for {data_path}; keeping float reader")
                                return None
                        encoded[start:start + len(batch)] = codes
                        start += len(batch)
                finally:
                    batches.close()
                target_batches = labels.batches(self.n_examples, self.n_examples)
                try:
                    targets = next(target_batches)
                finally:
                    target_batches.close()
                entry = CachedDomain(torch.from_numpy(encoded), targets, decode)
                self._write_disk(disk_path, entry, identity, size)
                self.stats["encoded"] += 1
                self._drop_file_pages(data_path)
            if self._closed.is_set():
                return None
            self.entries[key] = entry
            self.resident_bytes += size
            cache_log(f"Mixed cache ready: {data_path.parent.parent.name}/{data_path.parent.name}, "
                     f"{time.perf_counter() - started:.2f}s, resident={self.resident_bytes / 1024**3:.2f}/"
                     f"{self.budget / 1024**3:.2f} GiB, codec={decode}")
            return entry
        except Exception as exc:
            log.warning(f"Mixed cache unavailable for {data_path}: {exc}; using ordinary reader")
            return None

    def close(self):
        self._closed.set()
        if self._pending is not None:
            self._pending[1].cancel()
        if self._executor is not None:
            self._executor.shutdown(wait=True)
        self._pending = None
        cache_log(f"Mixed cache summary: {self.stats}, resident={self.resident_bytes / 1024**3:.2f} GiB")
        self.entries.clear()
        self.resident_bytes = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
