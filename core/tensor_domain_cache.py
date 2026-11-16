"""Bounded float32 domain caching for the temporal ImageNet loader."""

import os
from pathlib import Path
import struct
import zipfile

from loguru import logger as log
import numpy as np
import torch

from robustbench.pt_stream import TensorCacheStream
from robustbench.cache_io import advise_file_pages, drop_file_pages


GIB = 1024 ** 3
SCRATCH_BYTES = 256 * 1024 ** 2
IO_BLOCK_BYTES = 64 * 1024 ** 2


def memory_headroom():
    """Respect container limits; host MemAvailable alone is insufficient."""
    candidates = []
    try:
        fields = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
        candidates.append(int(fields["MemAvailable"].split()[0]) * 1024)
    except (OSError, ValueError, KeyError):
        pass
    for root, limit_name, usage_name, file_names, excluded_names in [
        (Path("/sys/fs/cgroup/memory"), "memory.limit_in_bytes", "memory.usage_in_bytes",
         ("total_cache", "cache"),
         (("total_shmem", "shmem"), ("total_dirty", "dirty"), ("total_writeback", "writeback"))),
        (Path("/sys/fs/cgroup"), "memory.max", "memory.current", ("file",),
         (("shmem",), ("file_dirty",), ("file_writeback",))),
    ]:
        try:
            limit = int((root / limit_name).read_text())
            usage = int((root / usage_name).read_text())
            stats = dict(line.split() for line in (root / "memory.stat").read_text().splitlines())
            def stat_value(names):
                return next((int(stats[name]) for name in names if name in stats), 0)

            # Active file pages are reclaimable too. Exclude tmpfs/shared
            # memory and pages that still need writeback; retain the reserve
            # in cache_budget to allow for reclaim latency and other jobs.
            if any(name in stats for name in file_names):
                reclaimable = max(0, stat_value(file_names)
                                  - sum(stat_value(names) for names in excluded_names))
            else:
                reclaimable = stat_value(("total_inactive_file", "inactive_file"))
            if limit < 2 ** 60:
                candidates.append(max(0, min(limit, limit - usage + reclaimable)))
        except (OSError, ValueError):
            continue
    return min(candidates) if candidates else 0


def read_float_domain(path, count):
    """Read only the requested prefix, with bounded scratch and no conversion."""
    stream = TensorCacheStream(path)
    if stream.dtype != np.dtype("float32") or len(stream.shape) != 4:
        raise ValueError(f"Expected a float32 NCHW tensor cache: {path}")
    if not 0 <= count <= len(stream):
        raise ValueError(f"Sample count exceeds tensor cache: {path}")
    result = torch.empty((count,) + stream.shape[1:], dtype=torch.float32)
    # The descriptor already verified ZIP_STORED, contiguous storage, dtype,
    # byte order and bounds. Read directly into owned tensor memory instead of
    # allocating ZIP byte strings and copying every block through numpy.
    with zipfile.ZipFile(path) as archive:
        entry = archive.getinfo(stream.member)
    target = memoryview(result.numpy()).cast("B")
    with Path(path).open("rb", buffering=0) as file:
        file.seek(entry.header_offset)
        header = file.read(30)
        if len(header) != 30 or header[:4] != b"PK\x03\x04":
            raise ValueError(f"Invalid tensor storage header: {path}")
        name_length, extra_length = struct.unpack_from("<HH", header, 26)
        file.seek(entry.header_offset + 30 + name_length + extra_length + stream.offset)
        try:
            position = 0
            while position < len(target):
                offset = file.tell()
                amount = file.readinto(target[position:position + IO_BLOCK_BYTES])
                if not amount:
                    raise EOFError(f"Truncated tensor cache: {path}")
                position += amount
                # Bound the transient file-cache copy to one I/O block.
                advise_file_pages(file, offset, amount)
        finally:
            advise_file_pages(file)
    return result


def cache_budget(domain_sizes, prefetch):
    requested = float(os.environ.get("DCF_LS_CACHE_RAM_GIB", "8"))
    reserve = float(os.environ.get("DCF_LS_MEMORY_RESERVE_GIB", "12"))
    if not np.isfinite(requested) or not np.isfinite(reserve) or requested < 0 or reserve < 0:
        raise ValueError("Cache RAM budget and memory reserve must be finite and nonnegative")
    headroom = memory_headroom()
    budget = min(int(requested * GIB), max(0, headroom - int(reserve * GIB)))
    largest = max(domain_sizes.values(), default=0)
    if budget < largest + SCRATCH_BYTES:
        raise MemoryError(f"Insufficient memory for one float32 domain plus scratch: "
                          f"required={(largest + SCRATCH_BYTES) / GIB:.2f} GiB, "
                          f"budget={budget / GIB:.2f} GiB "
                          f"(headroom={headroom / GIB:.2f} GiB, "
                          f"DCF_LS_CACHE_RAM_GIB={requested:g}, DCF_LS_MEMORY_RESERVE_GIB={reserve:g}); "
                          "adjust cache budget/reserve, reduce CORRUPTION.NUM_EX or free container memory")
    prefetch = prefetch and len(domain_sizes) > 1 and budget >= 2 * largest + SCRATCH_BYTES
    # Current domain and one future domain are reserved even after resident
    # slots fill. Fixed residents avoid cyclic-LRU thrashing between rounds.
    resident_limit = max(0, budget - largest * (2 if prefetch else 1) - SCRATCH_BYTES)
    if os.environ.get("DCF_LS_RESIDENT_CACHE", "1").lower() not in {"1", "true", "yes"}:
        resident_limit = 0
    log.info(f"Float32 cache: total_budget={budget / GIB:.2f} GiB, "
             f"resident_budget={resident_limit / GIB:.2f} GiB, domain_prefetch={prefetch}")
    return budget, resident_limit, prefetch
