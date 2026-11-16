import os
import copy
import pickle
import shutil
import tempfile
import argparse
from contextvars import ContextVar
import threading
import time
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')
import methods
import echo_log
from conf import cfg
from core.model import build_model
from loguru import logger as log
from setproctitle import setproctitle
from core.utils import seed_everything, save_df, set_logger
from robustbench.data import CORRUPTIONS_DIR_NAMES, PREPROCESSINGS, BenchmarkDataset, ThreatModel, get_preprocessing
from robustbench.loaders import CustomImageFolder
from robustbench.pt_stream import TensorCacheStream, prefetch_batches
from robustbench.mix_cache import MixedDomainCache, PIXEL_LUT, cache_log
import torch
from torch.utils.data import DataLoader, Subset
_THREAT_MODELS = {'imagenet': ThreatModel.corruptions, 'imagenet_3dcc': ThreatModel.corruptions_3d, 'imagenet_c_bar': ThreatModel.corruptions_c_bar}
_ACTIVE_MIX_CACHE = ContextVar('active_mix_cache', default=None)

def _maybe_stage_cache_file(src_path, data_dir):
    local_root = os.environ.get('ROBUSTBENCH_CACHE_LOCAL_DIR', '').strip()
    if not local_root:
        return src_path
    dst_path = Path(local_root) / src_path.relative_to(data_dir)
    if src_path.resolve() == dst_path.resolve():
        return src_path
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if not dst_path.exists() or dst_path.stat().st_size != src_path.stat().st_size:
        cache_log(f'Copying cache to local disk: {src_path} -> {dst_path}')
        with tempfile.TemporaryDirectory(dir=dst_path.parent) as tmp_dir:
            tmp_path = Path(tmp_dir) / src_path.name
            shutil.copy2(src_path, tmp_path)
            os.replace(tmp_path, dst_path)
    return dst_path

def _batch_accuracy(model, batches, device='cuda', uint8_cache=False):
    correct = 0
    total = 0
    pixel_lut = torch.from_numpy(PIXEL_LUT).to(device) if uint8_cache == 'lut' else None
    for (x_batch, y_batch) in batches:
        x_batch = x_batch.to(device, non_blocking=True)
        if pixel_lut is not None:
            x_batch = pixel_lut[x_batch.long()]
        elif uint8_cache:
            x_batch = x_batch.float().div_(255.0)
        y_batch = y_batch.to(device, non_blocking=True)
        output = model(x_batch, y=y_batch)
        echo_log.batch(model, output, y_batch)
        correct += (output.argmax(dim=1) == y_batch).sum().item()
        total += y_batch.numel()
    return correct / max(1, total)

def _cache_memory_headroom():
    """Estimate allocatable RAM using the cgroup limit and clean file pages.

    tmpfs/shmem and dirty/writeback pages are not counted as reclaimable. The
    caller also leaves a reserve for the model and concurrent allocations.
    Unknown container limits disable eager loading instead of using host RAM.
    """
    root = Path('/sys/fs/cgroup')
    try:
        if (root / 'memory.current').exists():
            usage = int((root / 'memory.current').read_text())
            limit = int((root / 'memory.max').read_text())
            (cache_key, dirty_key, writeback_key) = ('file', 'file_dirty', 'file_writeback')
        else:
            root /= 'memory'
            usage = int((root / 'memory.usage_in_bytes').read_text())
            limit = int((root / 'memory.limit_in_bytes').read_text())
            (cache_key, dirty_key, writeback_key) = ('cache', 'dirty', 'writeback')
        if limit >= 1 << 60:
            return None
        stats = dict((line.split() for line in (root / 'memory.stat').read_text().splitlines()))
        reclaimable = max(0, int(stats.get(cache_key, 0)) - int(stats.get('shmem', 0)) - int(stats.get(dirty_key, 0)) - int(stats.get(writeback_key, 0)))
        headroom = max(0, min(limit, limit - usage + reclaimable))
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith('MemAvailable:'):
                headroom = min(headroom, int(line.split()[1]) * 1024)
                break
        return headroom
    except (OSError, ValueError):
        return None

def _load_domain_cache(data_path, label_path):
    """Keep only the current domain in RAM; return None for bounded streaming."""
    mode = os.environ.get('ROBUSTBENCH_PT_LOAD_MODE', 'auto').lower()
    if mode not in {'auto', 'batch'}:
        raise ValueError('ROBUSTBENCH_PT_LOAD_MODE must be auto or batch')
    if mode == 'batch':
        return None
    max_bytes = float(os.environ.get('ROBUSTBENCH_DOMAIN_RAM_GIB', '4')) * 1024 ** 3
    reserve = float(os.environ.get('ROBUSTBENCH_MEMORY_RESERVE_GIB', '8')) * 1024 ** 3
    if max_bytes < 0 or reserve < 0:
        raise ValueError('cache RAM budget and memory reserve must be nonnegative')
    size = data_path.stat().st_size + label_path.stat().st_size
    headroom = _cache_memory_headroom()
    if size > max_bytes or headroom is None or headroom < 2 * size + reserve:
        cache_log(f"Using batch reads: domain={size / 1024 ** 3:.2f} GiB, RAM budget={max_bytes / 1024 ** 3:.2f} GiB, headroom={('unknown' if headroom is None else f'{headroom / 1024 ** 3:.2f} GiB')}")
        return None
    started = time.perf_counter()
    try:
        images = torch.load(data_path, map_location='cpu', weights_only=True)
        labels = torch.load(label_path, map_location='cpu', weights_only=True)
    except (OSError, RuntimeError, MemoryError, pickle.UnpicklingError) as exc:
        log.warning(f'Domain RAM load failed for {data_path} ({exc}); using batch reads.')
        return None
    cache_log(f'Loaded current domain into RAM: {size / 1024 ** 3:.2f} GiB in {time.perf_counter() - started:.2f}s; released after this domain')
    return (images, labels)

def _cache_paths(cfg, dataset_name, corruption_type, severity):
    threat_model = _THREAT_MODELS[dataset_name]
    folder = Path(cfg.DATA_DIR) / CORRUPTIONS_DIR_NAMES[BenchmarkDataset.imagenet][threat_model] / corruption_type / str(severity)
    use_uint8 = dataset_name == 'imagenet' and os.environ.get('ROBUSTBENCH_UINT8_PT_STREAM', '0').lower() in {'1', 'true', 'yes'} and (folder / 'data_uint8.pt').exists()
    data_path = folder / ('data_uint8.pt' if use_uint8 else 'data.pt')
    label_path = folder / 'label.pt'
    return (folder, data_path, label_path, use_uint8)

def _corruption_accuracy(model, cfg, dataset_name, corruption_type, severity):
    (folder, data_path, label_path, use_uint8) = _cache_paths(cfg, dataset_name, corruption_type, severity)
    ignore_cache = os.environ.get('ROBUSTBENCH_IGNORE_PT_CACHE', '0').lower() in {'1', 'true', 'yes'}
    fast_stream = os.environ.get('ROBUSTBENCH_FAST_PT_STREAM', '1').lower() in {'1', 'true', 'yes'}
    if fast_stream and (not ignore_cache):
        cache = _ACTIVE_MIX_CACHE.get()
        if cache is not None:
            entry = cache.get(data_path, label_path, use_uint8)
            if entry is not None:
                n = len(entry.images)
                batches = ((entry.images[start:start + cfg.TEST.BATCH_SIZE], entry.labels[start:start + cfg.TEST.BATCH_SIZE]) for start in range(0, n, cfg.TEST.BATCH_SIZE))
                started = time.perf_counter()
                try:
                    return _batch_accuracy(model, batches, uint8_cache='lut' if entry.decode == 'lut' else True)
                finally:
                    batches.close()
                    cache_log(f'Finished compact RAM domain: {dataset_name}/{corruption_type}/{severity}, samples={n}, adaptation={time.perf_counter() - started:.2f}s')
    if fast_stream and (not ignore_cache) and data_path.exists() and label_path.exists():
        try:
            data_path = _maybe_stage_cache_file(data_path, cfg.DATA_DIR)
            label_path = _maybe_stage_cache_file(label_path, cfg.DATA_DIR)
            x_test = TensorCacheStream(data_path)
            y_test = TensorCacheStream(label_path)
            if len(x_test.shape) != 4 or len(y_test.shape) != 1:
                raise ValueError('expected NCHW images and one-dimensional labels')
            if x_test.dtype.name != ('uint8' if use_uint8 else 'float32') or y_test.dtype.name != 'int64':
                raise ValueError('unexpected image or label dtype')
            if len(x_test) != len(y_test):
                raise ValueError('image and label counts differ')
            if dataset_name != 'imagenet' and len(x_test) < int(cfg.CORRUPTION.NUM_EX):
                raise ValueError('cache contains fewer samples than requested')
        except Exception as exc:
            log.warning(f'Cached stream unavailable for {folder} ({exc}); loading images instead.')
        else:
            n = min(int(cfg.CORRUPTION.NUM_EX), len(x_test))
            domain = _load_domain_cache(data_path, label_path)
            if domain is not None:
                (x_domain, y_domain) = domain
                if not isinstance(x_domain, torch.Tensor) or not isinstance(y_domain, torch.Tensor) or tuple(x_domain.shape) != x_test.shape or (tuple(y_domain.shape) != y_test.shape):
                    raise ValueError('cache changed after metadata validation')
                (x_domain, y_domain) = (x_domain[:n], y_domain[:n])
                batches = ((x_domain[start:start + cfg.TEST.BATCH_SIZE], y_domain[start:start + cfg.TEST.BATCH_SIZE]) for start in range(0, n, cfg.TEST.BATCH_SIZE))
                started = time.perf_counter()
                try:
                    return _batch_accuracy(model, batches, uint8_cache=use_uint8)
                finally:
                    batches.close()
                    del domain, x_domain, y_domain
                    cache_log(f'Finished RAM-cached domain: {dataset_name}/{corruption_type}/{severity}, samples={n}, adaptation={time.perf_counter() - started:.2f}s')
            cache_log(f'Streaming PT cache: {dataset_name}/{corruption_type}/{severity}, samples={n}, CPU image batch <= {cfg.TEST.BATCH_SIZE * x_test.sample_bytes / 1024 ** 2:.1f} MiB')
            images = x_test.batches(cfg.TEST.BATCH_SIZE, n)
            labels = y_test.batches(cfg.TEST.BATCH_SIZE, n)
            prefetch = os.environ.get('ROBUSTBENCH_PT_PREFETCH', '1').lower() in {'1', 'true', 'yes'}
            batches = prefetch_batches(zip(images, labels)) if prefetch else zip(images, labels)
            cache_log(f'PT batch prefetch: {int(prefetch)} batch ahead')
            try:
                return _batch_accuracy(model, batches, uint8_cache=use_uint8)
            finally:
                if prefetch:
                    batches.close()
                images.close()
                labels.close()
    if dataset_name == 'imagenet':
        transform = PREPROCESSINGS[None]
    else:
        transform = get_preprocessing(dataset_name, 'Linf', 'Standard_R50', 'Res224')
    dataset = CustomImageFolder(root=folder, transform=transform, setting='continual')
    dataset = Subset(dataset, range(min(int(cfg.CORRUPTION.NUM_EX), len(dataset))))
    cache_log(f'Streaming raw images: {dataset_name}/{corruption_type}/{severity}, samples={len(dataset)}')
    loader = DataLoader(dataset, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False, num_workers=int(os.environ.get('ROBUSTBENCH_STREAM_NUM_WORKERS', '0')), pin_memory=os.environ.get('ROBUSTBENCH_STREAM_PIN_MEMORY', '0').lower() in {'1', 'true', 'yes'})
    return _batch_accuracy(model, ((batch[0], batch[1]) for batch in loader))

def _log_memory(stage):
    """Report container usage, including reclaimable cache, rather than host RAM."""

    def read(path):
        try:
            return Path(path).read_text().strip()
        except OSError:
            return 'unavailable'

    def gib(value):
        return f'{int(value) / 1024 ** 3:.2f}GiB' if value.isdigit() else value
    root = Path('/sys/fs/cgroup')
    if (root / 'memory.current').exists():
        usage = read(root / 'memory.current')
        limit = read(root / 'memory.max')
        peak = read(root / 'memory.peak')
        events = read(root / 'memory.events').replace('\n', ',')
        stats = read(root / 'memory.stat')
        cache_key = 'file'
    else:
        root /= 'memory'
        usage = read(root / 'memory.usage_in_bytes')
        limit = read(root / 'memory.limit_in_bytes')
        peak = read(root / 'memory.max_usage_in_bytes')
        events = f"failcnt={read(root / 'memory.failcnt')}," + read(root / 'memory.oom_control').replace('\n', ',')
        stats = read(root / 'memory.stat')
        cache_key = 'cache'
    cache = dict((line.split() for line in stats.splitlines() if len(line.split()) == 2)).get(cache_key, 'unavailable')
    rss = next((line.split(':', 1)[1].strip() for line in read('/proc/self/status').splitlines() if line.startswith('VmRSS:')), 'unavailable')
    log.info(f'[memory:{stage}] pid={os.getpid()} RSS={rss}, cgroup={gib(usage)}/{gib(limit)}, peak={gib(peak)}, file_cache={gib(cache)}, {events}')

def _memory_heartbeat(stop, interval):
    while not stop.wait(interval):
        _log_memory('heartbeat')

def eval(cfg):
    budget = float(os.environ.get('ROBUSTBENCH_MIX_RAM_GIB', '0')) * 1024 ** 3
    if budget <= 0 or os.environ.get('ROBUSTBENCH_PT_LOAD_MODE', 'auto').lower() == 'batch':
        return _eval(cfg)
    reserve = float(os.environ.get('ROBUSTBENCH_MEMORY_RESERVE_GIB', '8')) * 1024 ** 3
    prefetch = os.environ.get('ROBUSTBENCH_DOMAIN_PREFETCH', '1').lower() in {'1', 'true', 'yes'}
    with MixedDomainCache(budget, reserve, _cache_memory_headroom, n_examples=int(cfg.CORRUPTION.NUM_EX), prefetch=prefetch, disk_dir=os.environ.get('ROBUSTBENCH_LOSSLESS_CACHE_DIR', '')) as cache:
        cache_log(f'Mixed-domain lossless RAM cache: budget={budget / 1024 ** 3:.1f} GiB, reserve={reserve / 1024 ** 3:.1f} GiB, next-domain prefetch={prefetch}')
        token = _ACTIVE_MIX_CACHE.set(cache)
        try:
            return _eval(cfg)
        finally:
            _ACTIVE_MIX_CACHE.reset(token)

def _eval(cfg):
    results_root = Path(os.environ.get('DCF_RESULTS_DIR', './results'))
    base_results = {'method': cfg.ADAPTER.NAME, 'dataset': cfg.CORRUPTION.DATASET, 'model': cfg.MODEL.ARCH, 'batch_size': cfg.TEST.BATCH_SIZE, 'seed': cfg.SEED, 'note': cfg.NOTE, 'order': cfg.CORRUPTION.ORDER_NUM, 'Avg': 0}
    log.info(f'base_results = {base_results}')
    model = build_model(cfg).cuda()
    source_model = copy.deepcopy(model) if cfg.ADAPTER.ECHO.ENABLED else None
    tta_model = methods.setup_adapter(model, cfg)
    tta_model.cuda()
    if cfg.ADAPTER.ECHO.ENABLED:
        from methods.ECHO import wrap_adapter
        tta_model = wrap_adapter(tta_model, cfg, source_model=source_model)
    summary_results = base_results.copy()
    rounds = cfg.TEST.ROUNDS
    imagenet_c_types = ['motion_blur', 'snow', 'fog', 'shot_noise', 'defocus_blur', 'contrast', 'zoom_blur', 'brightness', 'frost', 'elastic_transform', 'glass_blur', 'gaussian_noise', 'pixelate', 'jpeg_compression', 'impulse_noise']
    imagenet_3dcc_types = ['flash', 'h265_abr', 'far_focus', 'bit_error', 'low_light', 'z_motion_blur', 'fog_3d', 'near_focus', 'iso_noise', 'h265_crf', 'xy_motion_blur', 'color_quant']
    imagenet_c_bar_types = ['blue_noise', 'brownish_noise', 'caustic_refraction', 'checkerboard_cutout', 'cocentric_sine_waves', 'inverse_sparkles', 'perlin_noise', 'plasma_noise', 'single_frequency_greyscale', 'sparkles']
    cache = _ACTIVE_MIX_CACHE.get()
    if cache is not None:
        domains = [('imagenet', name) for name in imagenet_c_types] + [('imagenet_3dcc', name) for name in imagenet_3dcc_types] + [('imagenet_c_bar', name) for name in imagenet_c_bar_types]
        cache.set_order([cache.key(*_cache_paths(cfg, dataset, name, severity)[1:]) for severity in cfg.CORRUPTION.SEVERITY for (dataset, name) in domains])
    low_acc_counter = 0
    LOW_ACC_THRESHOLD = 5.0
    PATIENCE_LIMIT = 2
    real_rounds = 0
    stop_flag = False
    for _round in range(rounds):
        real_rounds += 1
        for severity in cfg.CORRUPTION.SEVERITY:
            base_results[f'round_{_round}'] = 0
            round_corruption_results = base_results.copy()
            for (i_c, corruption_type) in enumerate(imagenet_c_types):
                echo_log.begin_domain(tta_model, f'imagenet:{corruption_type}', _round)
                acc = _corruption_accuracy(tta_model, cfg, 'imagenet', corruption_type, severity)
                base_results[f'round_{_round}'] += acc * 100
                round_corruption_results[corruption_type] = acc * 100
                round_corruption_results['Avg'] += acc * 100
            for (i_c, corruption_type) in enumerate(imagenet_3dcc_types):
                echo_log.begin_domain(tta_model, f'imagenet_3dcc:{corruption_type}', _round)
                acc = _corruption_accuracy(tta_model, cfg, 'imagenet_3dcc', corruption_type, severity)
                base_results[f'round_{_round}'] += acc * 100
                round_corruption_results[corruption_type] = acc * 100
                round_corruption_results['Avg'] += acc * 100
            for (i_c, corruption_type) in enumerate(imagenet_c_bar_types):
                echo_log.begin_domain(tta_model, f'imagenet_c_bar:{corruption_type}', _round)
                acc = _corruption_accuracy(tta_model, cfg, 'imagenet_c_bar', corruption_type, severity)
                base_results[f'round_{_round}'] += acc * 100
                round_corruption_results[corruption_type] = acc * 100
                round_corruption_results['Avg'] += acc * 100
            total_corruptions = len(imagenet_c_types) + len(imagenet_3dcc_types) + len(imagenet_c_bar_types)
            base_results[f'round_{_round}'] = base_results[f'round_{_round}'] / total_corruptions
            round_corruption_results['Avg'] = round_corruption_results['Avg'] / total_corruptions
            save_df(round_corruption_results, results_root / f'detail/L-CS-MultiBenchmark{_round}_severity_{severity}.csv')
            round_acc = base_results[f'round_{_round}']
            round_error = 100 - round_acc
            log.info(f'[round_{_round}]: Acc {round_acc:.2f} || Error {round_error:.2f}')
            if round_acc < LOW_ACC_THRESHOLD:
                low_acc_counter += 1
                log.warning(f'Low accuracy detected ({round_acc:.2f}%). Consecutive low rounds: {low_acc_counter}/{PATIENCE_LIMIT}')
            else:
                low_acc_counter = 0
            if low_acc_counter >= PATIENCE_LIMIT and os.environ.get('ROBUSTBENCH_DISABLE_LOW_ACC_EARLY_STOP', '0') != '1':
                log.info(f'Accuracy has been below {LOW_ACC_THRESHOLD}% for {PATIENCE_LIMIT} consecutive rounds. Skipping remaining rounds.')
                stop_flag = True
                break
        if stop_flag:
            break
    total_acc_sum = sum([base_results.get(f'round_{r}', 0) for r in range(rounds)])
    divisor = real_rounds if real_rounds > 0 else 1
    base_results['Avg'] = total_acc_sum / divisor
    log.info(f"[Avg {cfg.CORRUPTION.SEVERITY}]: Acc {base_results['Avg']:.2f} || Error {100 - base_results['Avg']:.2f}")
    save_df(base_results, results_root / 'L-CS-MultiBenchmark.csv')
    echo_log.finish()
    flush_diagnostics = getattr(tta_model, 'flush_diagnostics', None)
    if callable(flush_diagnostics):
        flush_diagnostics()

def main():
    log.info(f'Running file: {os.path.basename(__file__)}')
    parser = argparse.ArgumentParser('Pytorch Implementation for Continual Test Time Adaptation!')
    parser.add_argument('-acfg', '--adapter-config-file', metavar='FILE', default='', help='path to adapter config file', type=str)
    parser.add_argument('-dcfg', '--dataset-config-file', metavar='FILE', default='', help='path to dataset config file', type=str)
    parser.add_argument('-ocfg', '--order-config-file', metavar='FILE', default='', help='path to order config file', type=str)
    parser.add_argument('-mcfg', '--model-config-file', metavar='FILE', default='', help='path to model config file', type=str)
    parser.add_argument('opts', help='modify the configuration by command line', nargs=argparse.REMAINDER, default=None)
    args = parser.parse_args()
    if len(args.opts) > 0:
        args.opts[-1] = args.opts[-1].strip('\r\n')
    cfg.merge_from_file(args.adapter_config_file)
    cfg.merge_from_file(args.dataset_config_file)
    if args.order_config_file != '':
        cfg.merge_from_file(args.order_config_file)
    if args.model_config_file != '':
        cfg.merge_from_file(args.model_config_file)
    cfg.merge_from_list(args.opts)
    if cfg.ADAPTER.ECHO.ENABLED:
        from methods.ECHO import configure_from_note
        configure_from_note(cfg)
    cfg.freeze()
    seed_everything(cfg.SEED)
    set_logger(cfg.LOG_DIR, cfg.ADAPTER.NAME)
    current_file_name = os.path.basename(__file__)
    setproctitle(f'{current_file_name}:{cfg.CORRUPTION.DATASET}:{cfg.ADAPTER.NAME}')
    log.info(f'Loaded configuration file: \n\tadapter: {args.adapter_config_file}\n\tdataset: {args.dataset_config_file}\n\torder: {args.order_config_file}\n\tmodel: {args.model_config_file}')
    log.info(f'LOADER config:\n{cfg.LOADER}')
    log.info(f'OPTIM config:\n{cfg.OPTIM}')
    log.info(f'MODEL config:\n{cfg.MODEL}')
    try:
        log.info(f'METHOD config:\n{cfg.ADAPTER[cfg.ADAPTER.NAME]}')
    except:
        pass
    if cfg.ADAPTER.ECHO.ENABLED:
        log.info(f'ECHO settings: {cfg.ADAPTER.ECHO}')
    interval = float(os.environ.get('LCS_MEMORY_LOG_INTERVAL', '0'))
    stop = threading.Event()
    heartbeat = None
    if interval > 0:
        _log_memory('startup')
        heartbeat = threading.Thread(target=_memory_heartbeat, args=(stop, interval), daemon=True)
        heartbeat.start()
    try:
        eval(cfg)
    finally:
        stop.set()
        if heartbeat is not None:
            heartbeat.join(timeout=1)
        if interval > 0:
            _log_memory('shutdown')
if __name__ == '__main__':
    main()
