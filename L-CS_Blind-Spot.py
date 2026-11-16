import os
import sys
import shutil
import warnings
import argparse
import torch
import echo_log
from pathlib import Path
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
import pickle
import math
import methods
import numpy as np
warnings.filterwarnings('ignore')
from copy import deepcopy
from einops import rearrange
from conf import cfg
from core.model import build_model
from core.utils import seed_everything, save_df, set_logger
from robustbench.utils import clean_accuracy as accuracy
from robustbench.data import load_imagenet3dcc, load_imagenetc, load_imagenet_c_bar, load_cifar100c, load_cifar10c
from setproctitle import setproctitle
from loguru import logger as log
from tqdm import tqdm, trange
from prefetch_generator import BackgroundGenerator

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())
_DATASET_ROOT_MAP = {'imagenet': 'ImageNet-C', 'imagenet_3dcc': 'ImageNet-3DCC', 'imagenet_c_bar': 'ImageNet-C-Bar', 'cifar10': 'CIFAR-10-C', 'cifar100': 'CIFAR-100-C', 'imagenet_k': 'ImageNet-K', 'imagenet_r': 'ImageNet-R', 'imagenet_a': 'ImageNet-A', 'imagenet_o': 'ImageNet-O', 'imagenet_v2': 'ImageNet-V2', 'waterbirds': 'waterbirds', 'coloredMNIST': 'ColoredMNIST', 'camelyon17_v1': 'camelyon17_v1'}
SPECIAL_DATASETS = ['imagenet_k', 'imagenet_r', 'imagenet_a', 'imagenet_o', 'imagenet_v2', 'imagenet_d_background', 'imagenet_d_material', 'imagenet_d_texture', 'waterbirds', 'coloredMNIST', 'camelyon17_v1']

def save_blind_spot_cache(path, x_full, y_full, indices, source_path=None):
    labels = y_full[indices]
    if source_path is None:
        cache = {'data': (x_full[indices], labels)}
    else:
        source_path = Path(source_path).resolve()
        source_stat = source_path.stat()
        cache = {'indices': indices, 'labels': labels, 'source': str(source_path), 'source_stamp': (source_stat.st_size, source_stat.st_mtime_ns)}
    torch.save(cache, path)

def load_blind_spot_cache(path):
    cache = torch.load(path, map_location='cpu')
    if 'data' in cache:
        return cache['data']
    source_stat = Path(cache['source']).stat()
    if (source_stat.st_size, source_stat.st_mtime_ns) != cache['source_stamp']:
        raise ValueError(f'Source tensor cache changed; delete {path} and regenerate it.')
    images = torch.load(cache['source'], map_location='cpu')[cache['indices']]
    return (images, cache['labels'])

def prepare_blind_spot_dataset(cfg, source_model):
    dataset_loaders = {'cifar10': load_cifar10c, 'cifar100': load_cifar100c, 'imagenet_3dcc': load_imagenet3dcc, 'imagenet': load_imagenetc, 'imagenet_c_bar': load_imagenet_c_bar}
    is_special = False
    load_image = dataset_loaders.get(cfg.CORRUPTION.DATASET)
    loop_severities = cfg.CORRUPTION.SEVERITY
    loop_types = cfg.CORRUPTION.TYPE
    dataset_folder = _DATASET_ROOT_MAP.get(cfg.CORRUPTION.DATASET, 'ImageNet-C')
    log.info('>>> Stage 1: Filtering Blind-Spot samples (Source model errors only)...')
    source_model.eval()
    data_path = os.path.join(cfg.DATA_DIR, f'{dataset_folder}/source_error_data_{cfg.MODEL.ARCH}')
    os.makedirs(data_path, exist_ok=True)
    for severity in loop_severities:
        for corruption_type in loop_types:
            save_path = os.path.join(data_path, f'source_error_{cfg.CORRUPTION.DATASET}_S{severity}_{corruption_type}.pt')
            if os.path.exists(save_path):
                log.info(f'Source-Error data for {corruption_type} (Sev: {severity}) already exists. Skipping.')
                continue
            (x_full, y_full) = load_image(cfg.CORRUPTION.NUM_EX, severity, cfg.DATA_DIR, False, [corruption_type])
            blind_indices = []
            with torch.no_grad():
                for i in trange(0, len(x_full), cfg.TEST.BATCH_SIZE, desc=f'Filtering {corruption_type}'):
                    batch_x = x_full[i:i + cfg.TEST.BATCH_SIZE].cuda()
                    batch_y = y_full[i:i + cfg.TEST.BATCH_SIZE].cuda()
                    output = source_model(batch_x)
                    pred = output.argmax(dim=1)
                    incorrect_mask = (pred != batch_y).cpu()
                    idx = torch.where(incorrect_mask)[0] + i
                    blind_indices.append(idx)
            blind_indices = torch.cat(blind_indices)
            if len(blind_indices) > 0:
                source_path = Path(cfg.DATA_DIR) / dataset_folder / corruption_type / str(severity) / 'data.pt'
                use_indices = cfg.CORRUPTION.DATASET in ('imagenet', 'imagenet_3dcc', 'imagenet_c_bar') and source_path.is_file() and (not (cfg.CORRUPTION.DATASET == 'imagenet' and os.environ.get('ROBUSTBENCH_IGNORE_PT_CACHE', '0').lower() in {'1', 'true', 'yes'}))
                log.info(f'Captured {len(blind_indices)} Source-Error samples for {corruption_type} (Sev: {severity})')
                save_blind_spot_cache(save_path, x_full, y_full, blind_indices, source_path=source_path if use_indices else None)
            else:
                log.warning(f"No errors found for {corruption_type}! This won't contribute to stress test.")

def eval(cfg):
    results_root = Path(os.environ.get('DCF_RESULTS_DIR', './results'))
    source_model = build_model(cfg).cuda()
    echo_source_model = deepcopy(source_model) if cfg.ADAPTER.ECHO.ENABLED else None
    mode = cfg.INPUT.BLIND_SPOT_MODE
    if mode == 'SOURCE_ERROR':
        prepare_blind_spot_dataset(cfg, source_model)
        data_folder_name = f'source_error_data_{cfg.MODEL.ARCH}'
        data_prefix = 'source_error'
    else:
        raise ValueError(f'Unknown BLIND_SPOT_MODE: {mode}')
    dataset_root = _DATASET_ROOT_MAP.get(cfg.CORRUPTION.DATASET, cfg.CORRUPTION.DATASET)
    loop_severities = cfg.CORRUPTION.SEVERITY
    loop_types = cfg.CORRUPTION.TYPE
    data_base_path = os.path.join(cfg.DATA_DIR, f'{dataset_root}/{data_folder_name}')
    tta_model = methods.setup_adapter(source_model, cfg)
    tta_model.cuda()
    if bool(cfg.ADAPTER.ECHO.ENABLED):
        from methods.ECHO import wrap_adapter
        tta_model = wrap_adapter(tta_model, cfg, source_model=echo_source_model)
    base_results = {'method': cfg.ADAPTER.NAME, 'dataset': cfg.CORRUPTION.DATASET, 'model': cfg.MODEL.ARCH, 'batch_size': cfg.TEST.BATCH_SIZE, 'seed': cfg.SEED, 'note': f'BlindSpot_{mode}_{cfg.NOTE}', 'order': cfg.CORRUPTION.ORDER_NUM, 'Avg': 0}
    log.info(f'Base Results Dict: {base_results}')
    summary_results = base_results.copy()
    rounds = cfg.TEST.ROUNDS
    if cfg.ADAPTER.NAME == 'Source':
        rounds = 1
    log.info(f">>> Stage 2: Starting {rounds}-round Stress Test on {cfg.ADAPTER.NAME} with '{mode}' data...")
    low_acc_counter = 0
    LOW_ACC_THRESHOLD = 5.0
    PATIENCE_LIMIT = 2
    real_rounds = 0
    stop_experiment = False
    for _round in trange(rounds, desc='Stress Test Rounds'):
        avg_acc_round = 0.0
        real_rounds += 1
        round_key = f'round_{_round}'
        summary_results[round_key] = 0
        round_corruption_results = base_results.copy()
        for corruption_type in loop_types:
            for severity in loop_severities:
                db_path = os.path.join(data_base_path, f'{data_prefix}_{cfg.CORRUPTION.DATASET}_S{severity}_{corruption_type}.pt')
                if not os.path.exists(db_path):
                    log.warning(f'Data file not found, skipping: {db_path}')
                    continue
                (x_adapt, y_adapt) = load_blind_spot_cache(db_path)
                echo_log.begin_domain(tta_model, corruption_type, _round)
                correct = 0
                total = 0
                ds = TensorDataset(x_adapt, y_adapt)
                dl = DataLoaderX(ds, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=False)
                tta_model.eval()
                with torch.no_grad():
                    for (xb, yb) in dl:
                        xb = xb.cuda()
                        yb = yb.cuda()
                        out = tta_model(xb, y=yb, adapt=True)
                        echo_log.batch(tta_model, out, yb)
                        pred = out.argmax(1)
                        correct += (pred == yb).sum().item()
                        total += yb.size(0)
                acc = correct / total
                acc_pct = acc * 100
                log.debug(f'Round {_round} | {corruption_type} | Sev {severity} | Acc: {acc_pct:.2f}% | Samples: {len(x_adapt)}')
                round_corruption_results[corruption_type] = acc_pct
        valid_corruptions = [k for k in loop_types if k in round_corruption_results]
        if len(valid_corruptions) > 0:
            avg_acc_round = sum((round_corruption_results[c] for c in valid_corruptions)) / len(valid_corruptions)
            summary_results[round_key] = avg_acc_round
            log.info(f'Round {_round} | Avg Acc: {avg_acc_round:.2f}%')
        else:
            summary_results[round_key] = 0
            log.warning(f'Round {_round} | No valid results to average.')
        if avg_acc_round < LOW_ACC_THRESHOLD:
            low_acc_counter += 1
            log.warning(f'Low accuracy detected ({avg_acc_round:.2f}%). Consecutive low rounds: {low_acc_counter}/{PATIENCE_LIMIT}')
        else:
            low_acc_counter = 0
        if low_acc_counter >= PATIENCE_LIMIT and os.environ.get('ROBUSTBENCH_DISABLE_LOW_ACC_EARLY_STOP', '0') != '1':
            log.info(f'Accuracy has been below {LOW_ACC_THRESHOLD}% for {PATIENCE_LIMIT} consecutive rounds. Skipping remaining rounds.')
            stop_experiment = True
            break
        if stop_experiment:
            break
    avg_total = sum([summary_results[f'round_{r}'] for r in range(real_rounds)])
    summary_results['Avg'] = avg_total / real_rounds if real_rounds > 0 else 0
    log.info(f"Final Protocol IV Avg Acc ({rounds} rounds): {summary_results['Avg']:.2f}%")
    save_df(summary_results, results_root / f'BlindSpot_Summary_{mode}_{cfg.CORRUPTION.DATASET}.csv')
    echo_log.finish()

def main():
    parser = argparse.ArgumentParser('Blind-Spot Stress Test (Protocol IV)')
    parser.add_argument('-acfg', '--adapter-config-file', metavar='FILE', default='', type=str)
    parser.add_argument('-dcfg', '--dataset-config-file', metavar='FILE', default='', type=str)
    parser.add_argument('-ocfg', '--order-config-file', metavar='FILE', default='', type=str)
    parser.add_argument('-mcfg', '--model-config-file', metavar='FILE', default='', type=str)
    parser.add_argument('opts', help='modify the configuration by command line', nargs=argparse.REMAINDER, default=None)
    args = parser.parse_args()
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
    setproctitle(f'StressTest:{cfg.ADAPTER.NAME}')
    log.remove()
    log.add(sys.stderr, level='DEBUG' if cfg.DEBUG else 'INFO')
    try:
        log.info(f'METHOD config:\n{cfg.ADAPTER[cfg.ADAPTER.NAME]}')
    except:
        pass
    try:
        eval(cfg)
    except Exception as e:
        log.exception(f'Stress Test Failed: {e}')
        raise
if __name__ == '__main__':
    main()
