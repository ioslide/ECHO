import torch
import torch
import copy
import sys
import gc
import random
import shutil
import os
import json
import warnings
warnings.filterwarnings('ignore')
import argparse
import subprocess
from pathlib import Path
import methods
from conf import cfg
from core.model import build_model
from core.utils import seed_everything, save_df, set_logger
from robustbench.utils import clean_accuracy as accuracy, clean_accuracy_loader
from robustbench.data import CORRUPTIONS_DIR_NAMES, BenchmarkDataset, ThreatModel, load_imagenet3dcc, load_imagenetc, load_imagenetc_dataset, load_imagenet_c_bar, load_cifar100c, load_cifar10c
from setproctitle import setproctitle
from loguru import logger as log
from tqdm import tqdm, trange
from torch.utils.data import DataLoader
from robustbench.cache_io import drop_file_pages, load_tensor_cache

def custom_accuracy(model, x, y, batch_size, device='cuda'):
    acc = 0.0
    n = x.shape[0]
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        x_batch = x[start:end].to(device)
        y_batch = y[start:end].to(device)
        output = model(x_batch, y=y_batch)
        acc += (output.max(1)[1] == y_batch).float().sum().item()
    return acc / max(1, n)

def custom_clean_accuracy_loader(model, loader, device='cuda'):
    acc = 0.0
    n = 0
    for batch in loader:
        if len(batch) < 2:
            raise ValueError(f'Expected loader batch with at least image and label, got {len(batch)} fields.')
        (x_batch, y_batch) = (batch[0], batch[1])
        (x_batch, y_batch) = (x_batch.to(device), y_batch.to(device))
        output = model(x_batch, y=y_batch)
        acc += (output.max(1)[1] == y_batch).float().sum().item()
        n += x_batch.shape[0]
    return acc / max(1, n)

def _collect_adapt_params_for_diagnostic(tta_model):
    if hasattr(tta_model, 'params') and hasattr(tta_model, 'param_names'):
        return (list(tta_model.params), list(tta_model.param_names))
    base_model = getattr(tta_model, 'model', tta_model)
    from methods.Tent import collect_params
    return collect_params(base_model)

def _imagenetc_cache_paths(data_dir, corruption_type, severity):
    data_folder_path = Path(data_dir) / CORRUPTIONS_DIR_NAMES[BenchmarkDataset.imagenet][ThreatModel.corruptions] / corruption_type / str(severity)
    return (data_folder_path / 'data.pt', data_folder_path / 'data_uint8.pt', data_folder_path / 'label.pt')

def _maybe_stage_cache_file(src_path, cfg, corruption_type, severity):
    local_root = os.environ.get('ROBUSTBENCH_CACHE_LOCAL_DIR', '').strip()
    if not local_root:
        return src_path
    src_path = Path(src_path)
    if not src_path.exists():
        return src_path
    local_dir = Path(local_root) / 'ImageNet-C' / str(corruption_type) / str(severity)
    local_dir.mkdir(parents=True, exist_ok=True)
    dst_path = local_dir / src_path.name
    try:
        if dst_path.exists() and dst_path.stat().st_size == src_path.stat().st_size:
            return dst_path
        tmp_path = dst_path.with_suffix(dst_path.suffix + '.tmp')
        shutil.copy2(src_path, tmp_path)
        os.replace(tmp_path, dst_path)
        return dst_path
    except Exception as exc:
        log.warning(f'Local cache staging failed for {src_path} ({exc}); using source path.')
        return src_path
    finally:
        drop_file_pages(src_path)

def _cached_tensor_accuracy(model, x, y, batch_size, device='cuda', uint8_cache=False):
    acc = 0.0
    n = int(x.shape[0])
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        x_curr = x[start:end].to(device)
        if uint8_cache:
            x_curr = x_curr.float().div_(255.0)
        y_curr = y[start:end].to(device)
        output = model(x_curr, y=y_curr)
        acc += (output.max(1)[1] == y_curr).float().sum().item()
    return acc / max(1, n)

def _try_cached_tensor_stream_accuracy(model, cfg, corruption_type, severity):
    fast_stream = os.environ.get('ROBUSTBENCH_FAST_PT_STREAM', '1').lower() in {'1', 'true', 'yes'}
    if not fast_stream:
        return None
    (data_path, uint8_path, label_path) = _imagenetc_cache_paths(cfg.DATA_DIR, corruption_type, severity)
    allow_uint8_cache = os.environ.get('ROBUSTBENCH_UINT8_PT_STREAM', '0').lower() in {'1', 'true', 'yes'}
    use_uint8_cache = allow_uint8_cache and uint8_path.exists()
    selected_data_path = uint8_path if use_uint8_cache else data_path
    if not (selected_data_path.exists() and label_path.exists()):
        return None
    try:
        selected_data_path = _maybe_stage_cache_file(selected_data_path, cfg, corruption_type, severity)
        label_path = _maybe_stage_cache_file(label_path, cfg, corruption_type, severity)
        x_test = load_tensor_cache(selected_data_path, map_location='cpu')[:cfg.CORRUPTION.NUM_EX]
        y_test = load_tensor_cache(label_path, map_location='cpu')[:cfg.CORRUPTION.NUM_EX]
    except Exception as exc:
        log.warning(f'Fast cached ImageNet-C stream unavailable ({exc}); falling back to DataLoader path.')
        return None
    return _cached_tensor_accuracy(model=model, x=x_test, y=y_test, batch_size=cfg.TEST.BATCH_SIZE, device='cuda', uint8_cache=use_uint8_cache)

def eval(cfg, print_every: int=1, disable_progress_bar: bool=False):
    base_results = {'method': cfg.ADAPTER.NAME, 'dataset': cfg.CORRUPTION.DATASET, 'model': cfg.MODEL.ARCH, 'batch_size': cfg.TEST.BATCH_SIZE, 'seed': cfg.SEED, 'note': cfg.NOTE, 'order': cfg.CORRUPTION.ORDER_NUM, 'Avg': 0}
    log.info(f'base_results = {base_results}')
    model = build_model(cfg).cuda()
    source_model = copy.deepcopy(model) if cfg.ADAPTER.ECHO.ENABLED else None
    results_root = Path(os.environ.get('DCF_RESULTS_DIR', './results'))
    requested_alignment = False
    tta_model = methods.setup_adapter(model, cfg)
    tta_model.cuda()
    if bool(cfg.ADAPTER.ECHO.ENABLED):
        from methods.ECHO import wrap_adapter
        tta_model = wrap_adapter(tta_model, cfg, source_model=source_model)
    dataset_loaders = {'cifar10': load_cifar10c, 'cifar100': load_cifar100c, 'imagenet_3dcc': load_imagenet3dcc, 'imagenet': load_imagenetc, 'imagenet_c_bar': load_imagenet_c_bar}
    load_image = dataset_loaders.get(cfg.CORRUPTION.DATASET)
    summary_results = base_results.copy()
    rounds = cfg.TEST.ROUNDS
    low_acc_counter = 0
    LOW_ACC_THRESHOLD = 5.0
    PATIENCE_LIMIT = 3
    disable_low_acc_early_stop = os.environ.get('ROBUSTBENCH_DISABLE_LOW_ACC_EARLY_STOP', '0').lower() in {'1', 'true', 'yes'}
    real_rounds = 0
    stop_experiment = False
    round_iter = trange(rounds, disable=disable_progress_bar)
    for _round in round_iter:
        real_rounds += 1
        round_corruption_types = list(cfg.CORRUPTION.TYPE)
        random.shuffle(round_corruption_types)
        log.info(f'[round_{_round}] Random target-domain order: {round_corruption_types}')
        for severity in cfg.CORRUPTION.SEVERITY:
            summary_results[f'round_{_round}'] = 0
            round_corruption_results = base_results.copy()
            for (i_c, corruption_type) in enumerate(round_corruption_types):
                stream_eval = cfg.INPUT.ROBUSTBENCH_STREAM_EVAL and cfg.CORRUPTION.DATASET == 'imagenet'
                if stream_eval:
                    acc = _try_cached_tensor_stream_accuracy(model=tta_model, cfg=cfg, corruption_type=corruption_type, severity=severity)
                    if acc is None:
                        dataset = load_imagenetc_dataset(cfg.CORRUPTION.NUM_EX, severity, cfg.DATA_DIR, False, [corruption_type])
                        stream_workers = int(os.environ.get('ROBUSTBENCH_STREAM_NUM_WORKERS', '0'))
                        stream_pin_memory = os.environ.get('ROBUSTBENCH_STREAM_PIN_MEMORY', '0').lower() in {'1', 'true', 'yes'}
                        test_loader = DataLoader(dataset, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False, num_workers=stream_workers, pin_memory=stream_pin_memory)
                        acc = custom_clean_accuracy_loader(model=tta_model, loader=test_loader, device='cuda')
                else:
                    (x_test, y_test) = load_image(cfg.CORRUPTION.NUM_EX, severity, cfg.DATA_DIR, False, [corruption_type])
                    acc = custom_accuracy(model=tta_model, x=x_test, y=y_test, batch_size=cfg.TEST.BATCH_SIZE, device='cuda')
                err = 1.0 - acc
                summary_results[f'round_{_round}'] += acc * 100
                round_corruption_results[corruption_type] = acc * 100
                round_corruption_results['Avg'] += acc * 100
                gc.collect()
                torch.cuda.empty_cache()
                log.info(f'[round_{_round}] {i_c + 1}/{len(round_corruption_types)} {corruption_type}: Acc={acc * 100:.2f}')
            round_corruption_results['Avg'] = round_corruption_results['Avg'] / len(round_corruption_types)
            summary_results[f'round_{_round}'] = summary_results[f'round_{_round}'] / len(round_corruption_types)
            save_df(round_corruption_results, results_root / 'detail' / f'L-CS-Shuffle_{_round}_{cfg.CORRUPTION.DATASET}_{cfg.CORRUPTION.ORDER_NUM}_severity_{severity}.csv')
            round_acc = summary_results[f'round_{_round}']
            round_error = 100 - round_acc
            log.info(f'[round_{_round}]: Acc {round_acc:.2f} || Error {round_error:.2f}')
            if round_acc < LOW_ACC_THRESHOLD:
                low_acc_counter += 1
                log.warning(f'Low accuracy detected ({round_acc:.2f}%). Consecutive low rounds: {low_acc_counter}/{PATIENCE_LIMIT}')
            else:
                low_acc_counter = 0
            if low_acc_counter >= PATIENCE_LIMIT:
                if disable_low_acc_early_stop:
                    log.info('Low-accuracy early stop is disabled by the explicit execution override; continuing the registered rounds.')
                else:
                    log.info(f'Accuracy has been below {LOW_ACC_THRESHOLD}% for {PATIENCE_LIMIT} consecutive rounds. Skipping remaining rounds.')
                    stop_experiment = True
                    break
        if stop_experiment:
            break
    total_acc_sum = sum([summary_results.get(f'round_{r}', 0) for r in range(real_rounds)])
    summary_results['Avg'] = total_acc_sum / real_rounds if real_rounds > 0 else 0
    current_severity = severity if 'severity' in locals() else cfg.CORRUPTION.SEVERITY[0]
    log.info(f"[Avg {current_severity}]: Acc {summary_results['Avg']:.2f} || Error {100 - summary_results['Avg']:.2f}")
    save_df(summary_results, results_root / f'L-CS-Shuffle_{cfg.CORRUPTION.DATASET}_{cfg.CORRUPTION.ORDER_NUM}.csv')
    flush_diagnostics = getattr(tta_model, 'flush_diagnostics', None)
    if callable(flush_diagnostics):
        flush_diagnostics()

def _split_runtime_opts(opts):
    runtime_values = {'print_every': None, 'disable_progress_bar': False}
    filtered = []
    i = 0
    while i < len(opts):
        token = opts[i]
        if token == '--print-every':
            if i + 1 < len(opts):
                runtime_values['print_every'] = opts[i + 1]
                i += 2
                continue
        if token == '--disable-progress-bar':
            runtime_values['disable_progress_bar'] = True
            i += 1
            continue
        filtered.append(token)
        i += 1
    return (filtered, runtime_values)

def main():
    log.info(f'Running file: {os.path.basename(__file__)}')
    parser = argparse.ArgumentParser('Pytorch Implementation for Continual Test Time Adaptation!')
    parser.add_argument('--print-every', type=int, default=1, help='Print a progress line every N corruption batches inside each round; 0 disables batch-level prints.')
    parser.add_argument('--disable-progress-bar', action='store_true', help='Disable tqdm progress bars for cleaner logs.')
    parser.add_argument('-acfg', '--adapter-config-file', metavar='FILE', default='', help='path to adapter config file', type=str)
    parser.add_argument('-dcfg', '--dataset-config-file', metavar='FILE', default='', help='path to dataset config file', type=str)
    parser.add_argument('-ocfg', '--order-config-file', metavar='FILE', default='', help='path to order config file', type=str)
    parser.add_argument('-mcfg', '--model-config-file', metavar='FILE', default='', help='path to model config file', type=str)
    parser.add_argument('opts', help='modify the configuration by command line', nargs=argparse.REMAINDER, default=None)
    parser.add_argument('--buffer-corruption', type=str, choices=['gaussian_noise', 'random_labels', None], default=None, help="Type of buffer corruption: 'gaussian_noise' or 'random_labels'")
    parser.add_argument('--corruption-batches', type=int, default=0.0, help='Number of initial batches to corrupt')
    parser.add_argument('--corruption-scale', type=float, default=0.0, help='Scale factor for corruption noise')
    args = parser.parse_args()
    if len(args.opts) > 0:
        args.opts[-1] = args.opts[-1].strip('\r\n')
        (args.opts, runtime_values) = _split_runtime_opts(args.opts)
    else:
        runtime_values = {'print_every': None, 'disable_progress_bar': False}
    cfg.merge_from_file(args.adapter_config_file)
    cfg.merge_from_file(args.dataset_config_file)
    if args.order_config_file != '':
        cfg.merge_from_file(args.order_config_file)
    if args.model_config_file != '':
        cfg.merge_from_file(args.model_config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    seed_everything(cfg.SEED)
    set_logger(cfg.LOG_DIR, cfg.ADAPTER.NAME)
    current_file_name = os.path.basename(__file__)
    setproctitle(f'{current_file_name}:{cfg.CORRUPTION.DATASET}:{cfg.ADAPTER.NAME}')
    log.info(cfg.OPTIM)
    log.info(f'Loaded configuration file: \n\tadapter: {args.adapter_config_file}\n\tdataset: {args.dataset_config_file}\n\torder: {args.order_config_file}\n\tmodel: {args.model_config_file}')
    log.remove()
    log_level = 'DEBUG' if cfg.DEBUG else 'INFO'
    log.add(sys.stderr, level=log_level)
    try:
        log.info(f'METHOD config:\n{cfg.ADAPTER[cfg.ADAPTER.NAME]}')
    except:
        pass
    try:
        print_every = int(runtime_values['print_every']) if runtime_values['print_every'] is not None else int(args.print_every)
        disable_progress_bar = bool(runtime_values['disable_progress_bar'] or args.disable_progress_bar)
        eval(cfg, print_every=print_every, disable_progress_bar=disable_progress_bar)
    except Exception as e:
        log.info(f'Error in TTA {e} \n {cfg.SEED} TTA: {cfg.ADAPTER.NAME} DATASET: {cfg.CORRUPTION.DATASET} BS: {cfg.TEST.BATCH_SIZE} MODEL: {cfg.MODEL.ARCH}  ORDER: {cfg.CORRUPTION.ORDER_NUM} SEVERITY: {cfg.CORRUPTION.SEVERITY} \n {cfg.NOTE}')
        raise
if __name__ == '__main__':
    main()
