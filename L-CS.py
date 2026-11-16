import torch
import copy
import echo_log
import sys
import gc
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
_IMAGE_DATASET_STREAMS = {'imagenet_k', 'imagenet_r', 'imagenet_a', 'imagenet_o', 'imagenet_v2'}
_IMAGENET_D_STREAMS = {'imagenet_d_background', 'imagenet_d_material', 'imagenet_d_texture'}

def custom_accuracy(model, x, y, batch_size, device='cuda', anomaly_state=None):
    acc = 0.0
    n = x.shape[0]
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        x_batch = x[start:end].to(device)
        poison = False
        y_batch = y[start:end].to(device)
        output = model(x_batch, y=y_batch)
        echo_log.batch(model, output, y_batch)
        acc += (output.max(1)[1] == y_batch).float().sum().item()
    return acc / max(1, n)

def custom_clean_accuracy_loader(model, loader, device='cuda', anomaly_state=None):
    acc = 0.0
    n = 0
    for batch in loader:
        if len(batch) < 2:
            raise ValueError(f'Expected loader batch with at least image and label, got {len(batch)} fields.')
        (x_batch, y_batch) = (batch[0], batch[1])
        (x_batch, y_batch) = (x_batch.to(device), y_batch.to(device))
        poison = False
        output = model(x_batch, y=y_batch)
        echo_log.batch(model, output, y_batch)
        acc += (output.max(1)[1] == y_batch).float().sum().item()
        n += x_batch.shape[0]
    return acc / max(1, n)

def custom_dataset_accuracy_loader(model, loader, device='cuda', anomaly_state=None):
    acc = 0.0
    n = 0
    for batch in loader:
        if isinstance(batch, dict):
            x_batch = batch['images']
            labels = batch['labels']
            if isinstance(labels, (list, tuple)):
                y_batch = labels[0]
            elif labels.ndim > 1:
                y_batch = labels[:, 0]
            else:
                y_batch = labels
        elif len(batch) >= 2:
            (x_batch, y_batch) = (batch[0], batch[1])
        else:
            raise ValueError('Expected dataset loader batch with image and label fields.')
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device).long().view(-1)
        poison = False
        output = model(x_batch, y=y_batch)
        echo_log.batch(model, output, y_batch)
        acc += (output.max(1)[1] == y_batch).float().sum().item()
        n += y_batch.numel()
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

def _cached_tensor_accuracy(model, x, y, batch_size, device='cuda', uint8_cache=False, anomaly_state=None):
    acc = 0.0
    n = int(x.shape[0])
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        x_curr = x[start:end].to(device)
        if uint8_cache:
            x_curr = x_curr.float().div_(255.0)
        poison = False
        y_curr = y[start:end].to(device)
        output = model(x_curr, y=y_curr)
        echo_log.batch(model, output, y_curr)
        acc += (output.max(1)[1] == y_curr).float().sum().item()
    return acc / max(1, n)

def _stream_rng(cfg, round_index, domain_index, salt=0):
    """Return a deterministic generator for registered stream variants."""
    seed = int(cfg.SEED) + 1000003 * (int(round_index) + 1) + 9176 * (int(domain_index) + 1) + int(salt)
    return torch.Generator(device='cpu').manual_seed(seed)

def _label_shift_indices(labels, cfg, round_index, domain_index, num_samples=None):
    alpha = float(os.environ.get('DCF_LABEL_SHIFT_ALPHA', '0.1'))
    n = int(num_samples if num_samples is not None else labels.shape[0])
    gen = _stream_rng(cfg, round_index, domain_index, salt=271828)
    labels_cpu = labels.detach().cpu().long()
    present_classes = torch.unique(labels_cpu, sorted=True)
    concentration = torch.full((len(present_classes),), alpha, dtype=torch.float32)
    probs = torch._standard_gamma(concentration, generator=gen)
    probs = probs / probs.sum()
    target = torch.multinomial(probs, n, replacement=True, generator=gen)
    buckets = {}
    for (idx, value) in enumerate(labels_cpu.tolist()):
        buckets.setdefault(int(value), []).append(idx)
    selected = []
    for cls_idx in target.tolist():
        cls = int(present_classes[int(cls_idx)])
        pool = buckets.get(cls)
        if not pool:
            continue
        draw = torch.randint(len(pool), (1,), generator=gen).item()
        selected.append(pool[draw])
    if len(selected) < n:
        fallback = torch.randint(n, (n - len(selected),), generator=gen).tolist()
        selected.extend(fallback)
    return torch.tensor(selected[:n], dtype=torch.long)

def _apply_stream_sampling(x, y, cfg, round_index, domain_index):
    scenario = os.environ.get('DCF_STREAM_SCENARIO', '').lower()
    if scenario in {'label_shift', 't-cs-ls', 'tcs-ls'}:
        idx = _label_shift_indices(y, cfg, round_index, domain_index)
        return (x[idx], y[idx])
    return (x, y)

def _mixed_severity(cfg, round_index, domain_index):
    gen = _stream_rng(cfg, round_index, domain_index, salt=314159)
    severities = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
    return int(severities[torch.randint(len(severities), (1,), generator=gen)].item())

def _try_cached_tensor_stream_accuracy(model, cfg, corruption_type, severity, round_index=0, domain_index=0, anomaly_state=None):
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
        x_test = torch.load(selected_data_path, map_location='cpu')
        y_test = torch.load(label_path, map_location='cpu')
        if os.environ.get('DCF_STREAM_SCENARIO', '').lower() in {'label_shift', 't-cs-ls', 'tcs-ls'}:
            idx = _label_shift_indices(y_test, cfg, round_index, domain_index, num_samples=int(cfg.CORRUPTION.NUM_EX))
            (x_test, y_test) = (x_test[idx], y_test[idx])
        else:
            x_test = x_test[:cfg.CORRUPTION.NUM_EX]
            y_test = y_test[:cfg.CORRUPTION.NUM_EX]
    except Exception as exc:
        log.warning(f'Fast cached ImageNet-C stream unavailable ({exc}); falling back to DataLoader path.')
        return None
    return _cached_tensor_accuracy(model=model, x=x_test, y=y_test, batch_size=cfg.TEST.BATCH_SIZE, device='cuda', uint8_cache=use_uint8_cache)

def _round_corruptions(corruptions, round_index):
    held_out = set(filter(None, os.environ.get('DCF_HELDOUT_CORRUPTIONS', '').split(',')))
    first_pass = int(os.environ.get('DCF_HELDOUT_FIRST_PASS', '20'))
    if round_index + 1 >= first_pass:
        return corruptions
    return [name for name in corruptions if name not in held_out]

def eval(cfg, print_every: int=1, disable_progress_bar: bool=False):
    base_results = {'method': cfg.ADAPTER.NAME, 'dataset': cfg.CORRUPTION.DATASET, 'model': cfg.MODEL.ARCH, 'batch_size': cfg.TEST.BATCH_SIZE, 'seed': cfg.SEED, 'note': cfg.NOTE, 'order': cfg.CORRUPTION.ORDER_NUM, 'scenario': os.environ.get('DCF_STREAM_SCENARIO', 't-cs') or 't-cs', 'label_shift_alpha': float(os.environ.get('DCF_LABEL_SHIFT_ALPHA', '0.1')), 'Avg': 0}
    log.info(f'base_results = {base_results}')
    model = build_model(cfg).cuda()
    source_model = copy.deepcopy(model) if bool(cfg.ADAPTER.ECHO.ENABLED) else None
    results_root = Path(os.environ.get('DCF_RESULTS_DIR', './results'))
    log.info(f'Results directory: {results_root.resolve()}')
    tta_model = methods.setup_adapter(model, cfg)
    tta_model.cuda()
    if bool(cfg.ADAPTER.ECHO.ENABLED):
        from methods.ECHO import wrap_adapter
        tta_model = wrap_adapter(tta_model, cfg, source_model=source_model)
    dataset_loaders = {'cifar10': load_cifar10c, 'cifar100': load_cifar100c, 'imagenet_3dcc': load_imagenet3dcc, 'imagenet': load_imagenetc, 'imagenet_c_bar': load_imagenet_c_bar}
    dataset_name = cfg.CORRUPTION.DATASET
    is_image_stream = False
    is_imagenet_d_stream = False
    is_natural_stream = False
    natural_loader = None
    load_image = dataset_loaders.get(dataset_name)
    if load_image is None:
        raise ValueError(f'Unsupported corruption dataset: {dataset_name}')
    evaluation_severities = cfg.CORRUPTION.SEVERITY
    evaluation_types = cfg.CORRUPTION.TYPE
    summary_results = base_results.copy()
    rounds = cfg.TEST.ROUNDS
    held_out = set(filter(None, os.environ.get('DCF_HELDOUT_CORRUPTIONS', '').split(',')))
    if held_out and (dataset_name != 'imagenet' or not held_out < set(evaluation_types) or (not 1 <= int(os.environ.get('DCF_HELDOUT_FIRST_PASS', '20')) <= rounds)):
        raise ValueError('Held-out corruptions require an ImageNet-C subset and a first pass within TEST.ROUNDS')
    low_acc_counter = 0
    LOW_ACC_THRESHOLD = 5.0
    PATIENCE_LIMIT = 3
    disable_low_acc_early_stop = os.environ.get('ROBUSTBENCH_DISABLE_LOW_ACC_EARLY_STOP', '0').lower() in {'1', 'true', 'yes'}
    real_rounds = 0
    stop_experiment = False
    round_iter = trange(rounds, disable=disable_progress_bar)
    for _round in round_iter:
        real_rounds += 1
        round_types = _round_corruptions(evaluation_types, _round)
        for severity in evaluation_severities:
            summary_results[f'round_{_round}'] = 0
            round_corruption_results = base_results.copy()
            for (i_c, corruption_type) in enumerate(round_types):
                echo_log.begin_domain(tta_model, corruption_type, _round)
                if os.environ.get('DCF_STREAM_SCENARIO', '').lower() in {'mixed_severity', 't-cs-msl', 'tcs-msl'}:
                    severity = _mixed_severity(cfg, _round, i_c)
                if hasattr(tta_model, 'set_domain_info'):
                    tta_model.set_domain_info(corruption_type, i_c)
                stream_eval = cfg.INPUT.ROBUSTBENCH_STREAM_EVAL and cfg.CORRUPTION.DATASET == 'imagenet'
                if stream_eval:
                    acc = _try_cached_tensor_stream_accuracy(model=tta_model, cfg=cfg, corruption_type=corruption_type, severity=severity, round_index=_round, domain_index=i_c)
                    if acc is None:
                        dataset = load_imagenetc_dataset(cfg.CORRUPTION.NUM_EX, severity, cfg.DATA_DIR, False, [corruption_type])
                        stream_workers = int(os.environ.get('ROBUSTBENCH_STREAM_NUM_WORKERS', '0'))
                        stream_pin_memory = os.environ.get('ROBUSTBENCH_STREAM_PIN_MEMORY', '0').lower() in {'1', 'true', 'yes'}
                        test_loader = DataLoader(dataset, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False, num_workers=stream_workers, pin_memory=stream_pin_memory)
                        acc = custom_clean_accuracy_loader(model=tta_model, loader=test_loader, device='cuda')
                else:
                    (x_test, y_test) = load_image(cfg.CORRUPTION.NUM_EX, severity, cfg.DATA_DIR, False, [corruption_type])
                    (x_test, y_test) = _apply_stream_sampling(x_test, y_test, cfg, _round, i_c)
                    acc = custom_accuracy(model=tta_model, x=x_test, y=y_test, batch_size=cfg.TEST.BATCH_SIZE, device='cuda')
                if hasattr(tta_model, 'report_domain_acc'):
                    tta_model.report_domain_acc(corruption_type, i_c, acc)
                err = 1.0 - acc
                summary_results[f'round_{_round}'] += acc * 100
                round_corruption_results[corruption_type] = acc * 100
                round_corruption_results['Avg'] += acc * 100
                gc.collect()
                torch.cuda.empty_cache()
                log.info(f'[round_{_round}] {i_c + 1}/{len(round_types)} {corruption_type}: Acc={acc * 100:.2f}')
            round_corruption_results['Avg'] = round_corruption_results['Avg'] / len(round_types)
            summary_results[f'round_{_round}'] = summary_results[f'round_{_round}'] / len(round_types)
            save_df(round_corruption_results, results_root / 'detail' / f'L-CS_{_round}_{cfg.CORRUPTION.DATASET}_{cfg.CORRUPTION.ORDER_NUM}_severity_{severity}.csv')
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
    save_df(summary_results, results_root / f'L-CS_{cfg.CORRUPTION.DATASET}_{cfg.CORRUPTION.ORDER_NUM}.csv')
    log.info(f"Saved summary: {(results_root / f'L-CS_{cfg.CORRUPTION.DATASET}_{cfg.CORRUPTION.ORDER_NUM}.csv').resolve()}")
    echo_log.finish()
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
    if cfg.ADAPTER.ECHO.ENABLED:
        from methods.ECHO import configure_from_note
        configure_from_note(cfg)
    cfg.freeze()
    seed_everything(cfg.SEED)
    set_logger(cfg.LOG_DIR, cfg.ADAPTER.NAME)
    current_file_name = os.path.basename(__file__)
    setproctitle(f'{current_file_name}:{cfg.CORRUPTION.DATASET}:{cfg.ADAPTER.NAME}')
    log.info(f'Loaded configuration file: \n\tadapter: {args.adapter_config_file}\n\tdataset: {args.dataset_config_file}\n\torder: {args.order_config_file}\n\tmodel: {args.model_config_file}')
    log.remove()
    log_level = 'DEBUG' if cfg.DEBUG else 'INFO'
    log.add(sys.stderr, level=log_level)
    log.info(f'Run settings: method={cfg.ADAPTER.NAME}, dataset={cfg.CORRUPTION.DATASET}, model={cfg.MODEL.ARCH}, seed={cfg.SEED}, batch_size={cfg.TEST.BATCH_SIZE}, rounds={cfg.TEST.ROUNDS}')
    log.info(f'OPTIM settings:\n{cfg.OPTIM}')
    if cfg.ADAPTER.NAME in cfg.ADAPTER:
        log.info(f'{cfg.ADAPTER.NAME} settings:\n{cfg.ADAPTER[cfg.ADAPTER.NAME]}')
    if cfg.ADAPTER.ECHO.ENABLED:
        log.info(f'ECHO settings:\n{cfg.ADAPTER.ECHO}')
    try:
        print_every = int(runtime_values['print_every']) if runtime_values['print_every'] is not None else int(args.print_every)
        disable_progress_bar = bool(runtime_values['disable_progress_bar'] or args.disable_progress_bar)
        eval(cfg, print_every=print_every, disable_progress_bar=disable_progress_bar)
    except Exception as e:
        log.info(f'Error in TTA {e} \n {cfg.SEED} TTA: {cfg.ADAPTER.NAME} DATASET: {cfg.CORRUPTION.DATASET} BS: {cfg.TEST.BATCH_SIZE} MODEL: {cfg.MODEL.ARCH}  ORDER: {cfg.CORRUPTION.ORDER_NUM} SEVERITY: {cfg.CORRUPTION.SEVERITY} \n {cfg.NOTE}')
        raise
if __name__ == '__main__':
    main()
