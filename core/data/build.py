import os, zipfile
import torch
from torch.utils.data import DataLoader
from loguru import logger as log
from prefetch_generator import BackgroundGenerator
from .datasets.common_corruption import CorruptionDataset, LazyImageNetCorruptionDataset, CachedImageNetCorruptionDataset
from .ttasampler import build_sampler
from core.utils.result_precess import AvgResultProcessor

class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

def build_loader(cfg):
    if cfg.CORRUPTION.DATASET == 'cifar10':
        dataset_name = 'CIFAR-10-C'
    elif cfg.CORRUPTION.DATASET == 'cifar100':
        dataset_name = 'CIFAR-100-C'
    elif cfg.CORRUPTION.DATASET == 'imagenet':
        dataset_name = 'ImageNet-C'
    elif cfg.CORRUPTION.DATASET == 'imagenet_3dcc':
        dataset_name = 'ImageNet-3DCC'
    elif cfg.CORRUPTION.DATASET == 'imagenet_c_bar':
        dataset_name = 'ImageNet-C-Bar'
    if cfg.CORRUPTION.DATASET in ['imagenet_3dcc', 'imagenet', 'imagenet_c_bar']:
        dataset_class = LazyImageNetCorruptionDataset
    elif cfg.CORRUPTION.DATASET in ['cifar10', 'cifar100']:
        dataset_class = CorruptionDataset
    else:
        raise NotImplementedError(f'Not Implement for dataset: {cfg.CORRUPTION.DATASET}')
    if cfg.LOADER.SAMPLER.TYPE == 'temporal':
        file = f'./data/{dataset_name}/temporal_{cfg.LOADER.SAMPLER.GAMMA}.pth'
        use_pt_cache = os.environ.get('DCF_LS_USE_PT_CACHE', '1').lower() in {'1', 'true', 'yes'}
        if use_pt_cache and cfg.CORRUPTION.DATASET in CachedImageNetCorruptionDataset._DATASET_ROOTS:
            try:
                ds = CachedImageNetCorruptionDataset(cfg)
                log.info(f'Using cached tensor backend for {dataset_name} (dtype=float32, samples={len(ds)})')
            except (MemoryError, FileNotFoundError, OSError, EOFError, RuntimeError, ValueError, zipfile.BadZipFile) as exc:
                log.warning(f'Tensor cache unavailable; falling back to raw image loader: {exc}')
                ds = dataset_class(cfg)
        else:
            ds = dataset_class(cfg)
        sampler = build_sampler(cfg, ds.data_source)
        default_workers = 0 if isinstance(ds, CachedImageNetCorruptionDataset) else int(cfg.LOADER.NUM_WORKS)
        num_workers = int(os.environ.get('DCF_LS_NUM_WORKERS', str(default_workers)))
        if isinstance(ds, CachedImageNetCorruptionDataset) and num_workers != 0:
            raise ValueError('Float32 tensor caching requires DCF_LS_NUM_WORKERS=0; domain and batch prefetch use background threads')
        pin_default = '1' if isinstance(ds, CachedImageNetCorruptionDataset) else '0'
        pin_memory = os.environ.get('DCF_LS_PIN_MEMORY', pin_default).lower() in {'1', 'true', 'yes'}
        use_prefetch = os.environ.get('DCF_LS_USE_PREFETCH', '1').lower() in {'1', 'true', 'yes'}
        loader_cls = DataLoaderX if use_prefetch else DataLoader
        loader_kwargs = {'batch_size': cfg.TEST.BATCH_SIZE, 'sampler': sampler, 'num_workers': num_workers, 'pin_memory': pin_memory}
        if num_workers > 0:
            loader_kwargs['persistent_workers'] = os.environ.get('DCF_LS_PERSISTENT_WORKERS', '1').lower() in {'1', 'true', 'yes'}
            loader_kwargs['prefetch_factor'] = max(2, int(os.environ.get('DCF_LS_PREFETCH_FACTOR', '2')))
        loader = loader_cls(ds, **loader_kwargs)
        log.info(f"L-CS-LS loader: workers={num_workers}, pin_memory={pin_memory}, prefetch={use_prefetch}, persistent={loader_kwargs.get('persistent_workers', False)}")
    elif cfg.LOADER.SAMPLER.TYPE == 'mix':
        file = f'./data/{dataset_name}/mix.pth'
        if os.path.exists(file):
            log.info(f'Load mixed dataset from ./data/{dataset_name}/mix.pth')
            ds = torch.load(f'./data/{dataset_name}/mix.pth')
        else:
            ds = dataset_class(cfg)
            torch.save(ds, f'./data/{dataset_name}/mix.pth')
        num_workers = int(os.environ.get('DCF_LS_NUM_WORKERS', str(cfg.LOADER.NUM_WORKS)))
        loader = DataLoaderX(ds, cfg.TEST.BATCH_SIZE, num_workers=num_workers, pin_memory=os.environ.get('DCF_LS_PIN_MEMORY', '0').lower() in {'1', 'true', 'yes'}, shuffle=True, persistent_workers=num_workers > 0)
    result_processor = AvgResultProcessor(ds.domain_id_to_name)
    return (loader, result_processor)
