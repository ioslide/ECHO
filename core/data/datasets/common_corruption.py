import os
import shutil
import uuid
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import time

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
from loguru import logger as log

from core.tensor_domain_cache import (
    GIB, SCRATCH_BYTES, cache_budget, memory_headroom, read_float_domain, drop_file_pages,
)
from robustbench.pt_stream import TensorCacheStream

from .base_dataset import TTADatasetBase, DatumRaw, DatumList, DatumBase
from robustbench.data import PREPROCESSINGS, load_cifar100c, load_cifar10c, load_imagenet3dcc, load_imagenetc, load_imagenet_c_bar
from robustbench.loaders import CustomImageFolder

from tqdm import tqdm, trange

class CorruptionDataset(TTADatasetBase):
    def __init__(self, cfg):
        self.corruptions = cfg.CORRUPTION.TYPE
        self.severity = cfg.CORRUPTION.SEVERITY
        self.corruptions = [self.corruptions] if not isinstance(self.corruptions, list) else self.corruptions
        self.severity = [self.severity] if not isinstance(self.severity, list) else self.severity

        dataset_loaders = {
            "cifar10": load_cifar10c,
            "cifar100": load_cifar100c,
            "imagenet_3dcc": load_imagenet3dcc,
            "imagenet": load_imagenetc,
            "imagenet_c_bar":load_imagenet_c_bar
        }
        self.load_image = dataset_loaders.get(cfg.CORRUPTION.DATASET)

        self.domain_id_to_name = {}
        data_source = []
        for i_s, severity in enumerate(self.severity):
            for i_c, corruption_type in tqdm(enumerate(self.corruptions), desc="Loading Corruption Dataset"):
                d_name = f"{corruption_type}_{severity}"
                d_id = i_s * len(self.corruptions) + i_c
                self.domain_id_to_name[d_id] = d_name
                x_test, y_test = self.load_image(
                    cfg.CORRUPTION.NUM_EX,
                    severity,
                    cfg.DATA_DIR,
                    False,
                    [corruption_type]
                )
                for i in range(len(y_test)):
                    data_item = DatumRaw(x_test[i], y_test[i].item(), d_id)
                    data_source.append(data_item)

        super().__init__(cfg, data_source)


class LazyImageNetCorruptionDataset(TorchDataset):
    """Lazily loads ImageNet corruption samples from disk to avoid materializing all tensors in RAM."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.corruptions = cfg.CORRUPTION.TYPE
        self.severity = cfg.CORRUPTION.SEVERITY
        self.corruptions = [self.corruptions] if not isinstance(self.corruptions, list) else self.corruptions
        self.severity = [self.severity] if not isinstance(self.severity, list) else self.severity
        self.domain_id_to_name = {}
        self.data_source = []
        dataset_roots = {
            "imagenet_3dcc": "ImageNet-3DCC",
            "imagenet_c_bar": "ImageNet-C-Bar",
            "imagenet": "ImageNet-C",
        }
        if cfg.CORRUPTION.DATASET not in dataset_roots:
            raise ValueError(f"LazyImageNetCorruptionDataset does not support {cfg.CORRUPTION.DATASET}")

        self.transform = PREPROCESSINGS["Res224"]

        n_examples = cfg.CORRUPTION.NUM_EX
        setting = getattr(cfg, "SETTING", "")
        dataset_root = os.path.join(cfg.DATA_DIR, dataset_roots[cfg.CORRUPTION.DATASET])

        for i_s, severity in enumerate(self.severity):
            for i_c, corruption_type in tqdm(enumerate(self.corruptions), desc="Loading Corruption Dataset"):
                d_name = f"{corruption_type}_{severity}"
                d_id = i_s * len(self.corruptions) + i_c
                self.domain_id_to_name[d_id] = d_name
                corruption_root = os.path.join(dataset_root, corruption_type, str(severity))
                folder_dataset = CustomImageFolder(
                    root=corruption_root,
                    transform=None,
                    setting=setting,
                )
                samples = folder_dataset.samples[:n_examples] if n_examples is not None else folder_dataset.samples
                for img_path, label in samples:
                    self.data_source.append(DatumList(img=img_path, label=label, domain=d_id))

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, index):
        item = self.data_source[index]
        with Image.open(item.img) as img:
            image = img.convert("RGB")
        image = self.transform(image)
        return {
            "label": item.label,
            "domain": item.domain,
            "image": image,
        }


class CachedImageNetCorruptionDataset(TorchDataset):
    """Use preprocessed tensors while preserving temporal sampler metadata."""

    _DATASET_ROOTS = {
        "imagenet": "ImageNet-C",
        "imagenet_3dcc": "ImageNet-3DCC",
        "imagenet_c_bar": "ImageNet-C-Bar",
    }

    def __init__(self, cfg):
        dataset_name = cfg.CORRUPTION.DATASET
        try:
            dataset_root = Path(cfg.DATA_DIR) / self._DATASET_ROOTS[dataset_name]
        except KeyError as exc:
            raise ValueError(f"Unsupported cached corruption dataset: {dataset_name}") from exc

        corruptions = cfg.CORRUPTION.TYPE
        severities = cfg.CORRUPTION.SEVERITY
        self.corruptions = [corruptions] if not isinstance(corruptions, list) else corruptions
        self.severity = [severities] if not isinstance(severities, list) else severities
        self.n_examples = cfg.CORRUPTION.NUM_EX
        self.domain_id_to_name = {}
        self.data_source = []
        self._domain_data_paths = {}
        self._domain_sizes = {}
        self._domain_bytes = {}
        self._resident_data = {}
        self._resident_bytes = 0
        self._loaded_domain = None
        self._loaded_data = None
        self._pending = None
        self._pending_domain = None
        self._executor = None
        self._closed = False
        self.cache_stats = {"reads": 0, "prefetch_hits": 0, "read_seconds": 0.0,
                            "stage_seconds": 0.0, "domain_wait_seconds": 0.0}

        for severity_idx, severity in enumerate(self.severity):
            for corruption_idx, corruption_type in enumerate(self.corruptions):
                domain_id = severity_idx * len(self.corruptions) + corruption_idx
                cache_dir = dataset_root / corruption_type / str(severity)
                data_path = cache_dir / "data.pt"
                label_path = cache_dir / "label.pt"
                if not (data_path.is_file() and label_path.is_file()):
                    raise FileNotFoundError(
                        f"Missing tensor cache for {corruption_type}_{severity}: "
                        f"{data_path} and/or {label_path}"
                    )

                labels = torch.load(label_path, map_location="cpu")
                n = len(labels) if self.n_examples is None else min(int(self.n_examples), len(labels))
                stream = TensorCacheStream(data_path)
                if stream.dtype != np.dtype("float32") or len(stream.shape) != 4 or not 0 <= n <= len(stream):
                    raise ValueError(f"Invalid float32 NCHW cache or sample count: {data_path}")
                self.domain_id_to_name[domain_id] = f"{corruption_type}_{severity}"
                self._domain_data_paths[domain_id] = data_path
                self._domain_sizes[domain_id] = n
                self._domain_bytes[domain_id] = n * stream.sample_bytes
                for local_idx in range(n):
                    self.data_source.append(
                        DatumBase(
                            img=(domain_id, local_idx),
                            label=int(labels[local_idx]),
                            domain=domain_id,
                        )
                    )

        prefetch = os.environ.get("DCF_LS_DOMAIN_PREFETCH", "0").lower() in {"1", "true", "yes"}
        self._budget, self._resident_limit, self._prefetch = cache_budget(self._domain_bytes, prefetch)
        self._reserve_bytes = int(float(os.environ.get("DCF_LS_MEMORY_RESERVE_GIB", "12")) * GIB)
        self._domains = sorted(self.domain_id_to_name)
        if self._prefetch:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ls-domain")

    def __len__(self):
        return len(self.data_source)

    def _cache_path(self, domain_id):
        source = self._domain_data_paths[domain_id]
        local_root = os.environ.get("DCF_LS_CACHE_LOCAL_DIR", os.environ.get("ROBUSTBENCH_CACHE_LOCAL_DIR", "")).strip()
        if not local_root:
            return source
        relative = source.relative_to(source.parents[3])
        local = Path(local_root) / relative
        source_stat = source.stat()
        if local.is_file():
            local_stat = local.stat()
            if (local_stat.st_size, local_stat.st_mtime_ns) == (source_stat.st_size, source_stat.st_mtime_ns):
                return local
        temporary = local.with_name(f".{local.name}.{uuid.uuid4().hex}.tmp")
        try:
            local.parent.mkdir(parents=True, exist_ok=True)
            if shutil.disk_usage(local.parent).free < source_stat.st_size + GIB:
                log.warning(f"Insufficient local cache disk space; reading {source}")
                return source
            shutil.copy2(source, temporary)
            os.replace(temporary, local)
            drop_file_pages(local)
        except OSError as exc:
            log.warning(f"Local cache staging failed; reading {source}: {exc}")
            return source
        finally:
            drop_file_pages(source)
            temporary.unlink(missing_ok=True)
        return local

    def _read_domain(self, domain_id):
        # Check again at allocation time: other jobs can consume memory after
        # the initial budget was calculated. Fail before allocating a full domain.
        required = self._domain_bytes[domain_id] + SCRATCH_BYTES + self._reserve_bytes
        if memory_headroom() < required:
            raise MemoryError(f"Insufficient container headroom to read {self.domain_id_to_name[domain_id]}; "
                              "free memory or lower DCF_LS_CACHE_RAM_GIB")
        started = time.monotonic()
        path = self._cache_path(domain_id)
        staged = time.monotonic()
        try:
            data = read_float_domain(path, self._domain_sizes[domain_id])
        finally:
            drop_file_pages(path)
        finished = time.monotonic()
        self.cache_stats["reads"] += 1
        self.cache_stats["stage_seconds"] += staged - started
        self.cache_stats["read_seconds"] += finished - staged
        location = "dataset" if path == self._domain_data_paths[domain_id] else "local"
        log.info(f"Loaded {self.domain_id_to_name[domain_id]}: "
                 f"{self._domain_bytes[domain_id] / GIB:.2f} GiB in {finished - started:.2f}s "
                 f"(source={location}, stage={staged - started:.2f}s, read={finished - staged:.2f}s)")
        return data

    def _retain_current(self):
        domain = self._loaded_domain
        if (domain not in self._resident_data and self._loaded_data is not None
                and self._resident_bytes + self._domain_bytes[domain] <= self._resident_limit):
            self._resident_data[domain] = self._loaded_data
            self._resident_bytes += self._domain_bytes[domain]

    def _load_domain(self, domain_id):
        if self._closed:
            raise RuntimeError("Tensor cache dataset is closed")
        if self._loaded_domain == domain_id and self._loaded_data is not None:
            return
        started = time.monotonic()

        # Release the previous streaming tensor BEFORE allocating its successor.
        self._loaded_data = None
        self._loaded_domain = None
        prefetched = None
        if self._pending is not None:
            future, pending_domain = self._pending, self._pending_domain
            self._pending = self._pending_domain = None
            result = future.result()
            if pending_domain == domain_id:
                prefetched = result
                self.cache_stats["prefetch_hits"] += 1
            del result, future

        if domain_id in self._resident_data:
            self._loaded_data = self._resident_data[domain_id]
        elif prefetched is not None:
            self._loaded_data = prefetched
        else:
            self._loaded_data = self._read_domain(domain_id)
        self._loaded_domain = domain_id
        self._retain_current()
        waited = time.monotonic() - started
        self.cache_stats["domain_wait_seconds"] += waited
        log.info(f"Selected {self.domain_id_to_name[domain_id]}: domain_wait={waited:.3f}s, "
                 f"resident={domain_id in self._resident_data}, prefetched={prefetched is not None}")

        if self._executor is not None:
            next_domain = self._domains[(self._domains.index(domain_id) + 1) % len(self._domains)]
            if next_domain != domain_id and next_domain not in self._resident_data:
                self._pending_domain = next_domain
                self._pending = self._executor.submit(self._read_domain, next_domain)

    def __getitem__(self, index):
        item = self.data_source[index]
        domain_id, local_idx = item.img
        self._load_domain(domain_id)
        # A batch can straddle domains. Views in DataLoader's sample list would
        # otherwise keep entire streaming domains alive beyond the budget.
        image = self._loaded_data[local_idx].clone()
        return {
            "label": item.label,
            "domain": item.domain,
            "image": image,
        }

    def close(self):
        self._closed = True
        try:
            if self._executor is not None:
                self._executor.shutdown(wait=True)
        finally:
            self._executor = None
            self._pending = self._pending_domain = None
            self._loaded_data = self._loaded_domain = None
            self._resident_data.clear()
            self._resident_bytes = 0


class CorruptionImageNet(TTADatasetBase):
    def __init__(self, cfg):
        # 确保输入是列表格式
        all_corruption = cfg.CORRUPTION.TYPE
        all_severity = cfg.CORRUPTION.SEVERITY
        all_corruption = [all_corruption] if not isinstance(all_corruption, list) else all_corruption
        all_severity = [all_severity] if not isinstance(all_severity, list) else all_severity

        self.corruptions = all_corruption
        self.severity = all_severity

        self.domain_id_to_name = {}
        data_source = []

        class_to_idx = None

        for i_s, severity in enumerate(self.severity):
            for i_c, corruption in tqdm(enumerate(self.corruptions), desc="Loading ImageNet-C Dataset"):
                d_name = f"{corruption}_{severity}"
                d_id = i_s * len(self.corruptions) + i_c
                self.domain_id_to_name[d_id] = d_name

                target_root = os.path.join(cfg.DATA_DIR, 'ImageNet-C', corruption, str(severity))

                if not os.path.exists(target_root):
                    print(f"Warning: Path not found {target_root}")
                    continue

                if class_to_idx is None:
                    classes = sorted([d for d in os.listdir(target_root) if os.path.isdir(os.path.join(target_root, d))])
                    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

                # 遍历所有类别文件夹
                for class_name, label in class_to_idx.items():
                    class_dir = os.path.join(target_root, class_name)
                    if not os.path.isdir(class_dir):
                        continue

                    # 遍历文件夹内的图片
                    # sorted() 保证读取顺序确定性
                    for fname in sorted(os.listdir(class_dir)):
                        if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                            fpath = os.path.join(class_dir, fname)

                            # 使用 DatumList 仅存储路径，不加载图片到内存
                            data_item = DatumList(img=fpath, label=label, domain=d_id, classname=class_name)
                            data_source.append(data_item)

        # 初始化基类
        super().__init__(cfg, data_source)
