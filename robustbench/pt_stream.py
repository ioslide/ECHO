"""Read contiguous torch.save tensor caches in bounded batches, including on torch 2.0.

Only the small tensor descriptor is unpickled. Storage bytes stay in the ZIP
file until a batch is requested; no full tensor load or local copy is needed.
Unsupported cache layouts must fall back to the original image dataset.
"""

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import io
import math
import pickle
import sys
from typing import NamedTuple
import zipfile

import numpy as np
import torch


def prefetch_batches(batches):
    """Read one batch ahead; close before closing the underlying iterators."""
    batches = iter(batches)
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="pt-prefetch") as executor:
        pending = executor.submit(next, batches, None)
        while True:
            batch = pending.result()
            if batch is None:
                return
            pending = executor.submit(next, batches, None)
            yield batch
            del batch


class _Storage(NamedTuple):
    dtype: np.dtype
    key: str
    count: int


class _Tensor(NamedTuple):
    storage: _Storage
    offset: int
    shape: tuple
    stride: tuple


def _tensor_descriptor(storage, offset, shape, stride, *unused):
    return _Tensor(storage, offset, shape, stride)


class _DescriptorUnpickler(pickle.Unpickler):
    """Accept tensor metadata only, never arbitrary globals from a cache."""

    def find_class(self, module, name):
        if module == "torch._utils" and name in {"_rebuild_tensor", "_rebuild_tensor_v2"}:
            return _tensor_descriptor
        if module == "collections" and name == "OrderedDict":
            return OrderedDict
        dtypes = {"FloatStorage": "float32", "DoubleStorage": "float64",
                  "HalfStorage": "float16", "ByteStorage": "uint8",
                  "LongStorage": "int64", "IntStorage": "int32",
                  "ShortStorage": "int16", "CharStorage": "int8",
                  "BoolStorage": "bool"}
        if module == "torch" and name in dtypes:
            return np.dtype(dtypes[name])
        raise ValueError(f"unsupported tensor metadata: {module}.{name}")

    def persistent_load(self, value):
        if (not isinstance(value, tuple) or len(value) != 5
                or value[0] != "storage" or not isinstance(value[1], np.dtype)):
            raise ValueError("unsupported storage descriptor")
        _, dtype, key, _, count = value
        return _Storage(dtype, str(key), int(count))


class TensorCacheStream:
    """A single dense CPU tensor stored by torch.save's ZIP serializer.

    Each yielded tensor owns writable storage of at most one batch. Metadata is
    validated before adaptation starts, so unsupported formats can safely use
    the image loader instead. Read errors during adaptation must propagate.
    """

    def __init__(self, path):
        self.path = path
        with zipfile.ZipFile(path) as archive:
            descriptors = [i for i in archive.infolist() if i.filename.endswith("/data.pkl")]
            if len(descriptors) != 1 or descriptors[0].file_size > 65536:
                raise ValueError("expected one small tensor descriptor")
            descriptor = descriptors[0]
            prefix = descriptor.filename[:-len("data.pkl")]
            tensor = _DescriptorUnpickler(io.BytesIO(archive.read(descriptor))).load()
            if not isinstance(tensor, _Tensor):
                raise ValueError("cache must contain a single tensor")
            if (not tensor.shape or any(type(n) is not int or n < 0 for n in tensor.shape)
                    or len(tensor.shape) != len(tensor.stride) or tensor.offset < 0):
                raise ValueError("invalid tensor shape or offset")
            stride = 1
            for size, actual_stride in zip(reversed(tensor.shape), reversed(tensor.stride)):
                if size > 1 and actual_stride != stride:
                    raise ValueError("cache tensor must be contiguous")
                stride *= size
            if tensor.offset + math.prod(tensor.shape) > tensor.storage.count:
                raise ValueError("tensor exceeds storage")
            self.member = prefix + "data/" + tensor.storage.key
            storage = archive.getinfo(self.member)
            if (storage.compress_type != zipfile.ZIP_STORED
                    or storage.file_size != tensor.storage.count * tensor.storage.dtype.itemsize):
                raise ValueError("expected uncompressed tensor storage with matching size")
            byteorder_name = prefix + "byteorder"
            byteorder = archive.read(byteorder_name).decode() if byteorder_name in archive.namelist() else "little"
            if byteorder != sys.byteorder:
                raise ValueError("cache byte order differs from this machine")
        self.shape = tensor.shape
        self.dtype = tensor.storage.dtype
        self.offset = tensor.offset * self.dtype.itemsize
        self.sample_bytes = math.prod(self.shape[1:]) * self.dtype.itemsize

    def __len__(self):
        return self.shape[0]

    def batches(self, batch_size, count):
        if batch_size <= 0 or not 0 <= count <= len(self):
            raise ValueError("invalid batch size or sample count")
        with zipfile.ZipFile(self.path) as archive, archive.open(self.member) as storage:
            storage.seek(self.offset)
            for start in range(0, count, batch_size):
                size = min(batch_size, count - start)
                # bytearray gives numpy/torch writable, independently owned data.
                chunk = bytearray(storage.read(size * self.sample_bytes))
                if len(chunk) != size * self.sample_bytes:
                    raise EOFError(f"truncated tensor cache: {self.path}")
                array = np.frombuffer(chunk, dtype=self.dtype).reshape((size,) + self.shape[1:])
                yield torch.from_numpy(array)
