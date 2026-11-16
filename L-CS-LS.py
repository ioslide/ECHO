import sys
import os
import gc
import warnings
warnings.filterwarnings('ignore')
import os
import torch
import methods
import argparse
import pandas as pd
from conf import cfg
from core.model import build_model
from core.data import build_loader
from core.utils import seed_everything, save_df, set_logger
from tqdm import tqdm
from setproctitle import setproctitle
from loguru import logger as log
import copy
import echo_log
from pathlib import Path

def eval(cfg):
    results_root = Path(os.environ.get('DCF_RESULTS_DIR', './results'))
    base_results = {'method': cfg.ADAPTER.NAME, 'dataset': cfg.CORRUPTION.DATASET, 'model': cfg.MODEL.ARCH, 'batch_size': cfg.TEST.BATCH_SIZE, 'seed': cfg.SEED, 'note': cfg.NOTE, 'order': cfg.CORRUPTION.ORDER_NUM, 'Avg': 0}
    log.info(f'base_results = {base_results}')
    model = build_model(cfg).cuda()
    source_model = copy.deepcopy(model) if bool(cfg.ADAPTER.ECHO.ENABLED) else None
    tta_model = methods.setup_adapter(model, cfg)
    tta_model.cuda()
    if bool(cfg.ADAPTER.ECHO.ENABLED):
        from methods.ECHO import wrap_adapter
        tta_model = wrap_adapter(tta_model, cfg, source_model=source_model)
    (loader, processor) = build_loader(cfg)
    summary_results = base_results.copy()
    rounds = cfg.TEST.ROUNDS
    low_acc_counter = 0
    LOW_ACC_THRESHOLD = 3.0
    PATIENCE_LIMIT = 2
    real_rounds = 0
    try:
        for _round in range(rounds):
            real_rounds += 1
            summary_results[f'round_{_round}'] = 0
            round_corruption_results = base_results.copy()
            for (batch_id, data_package) in enumerate(loader):
                (x_test, y_test, domain) = (data_package['image'], data_package['label'], data_package['domain'])
                x_test = x_test.cuda(non_blocking=True)
                y_test = y_test.cuda(non_blocking=True)
                output = tta_model(x_test)
                predict = torch.argmax(output, dim=1)
                accurate = predict == y_test
                processor.process(accurate, domain)
                if echo_log.enabled:
                    masks = {loader.dataset.domain_id_to_name[int(d)]: (domain == d).to(output.device) for d in domain.unique()}
                    echo_log.batch(tta_model, output, y_test, round_idx=_round, domains=masks)
            processor.calculate()
            _results = processor.info()
            acc = _results['Avg']
            for (i_c, corruption_type) in enumerate(cfg.CORRUPTION.TYPE):
                round_corruption_results[corruption_type] = _results[f'{corruption_type}_5']
            round_corruption_results['Avg'] += acc * 100
            save_df(round_corruption_results, results_root / f'detail/L-CS-LS_{_round}_{cfg.CORRUPTION.DATASET}_{cfg.CORRUPTION.ORDER_NUM}_severity_5.csv')
            summary_results[f'round_{_round}'] += acc
            _error = 100 - acc
            log.info(f'[round_{_round}]: Acc {acc:.2f} || Error {_error:.2f}')
            if acc < LOW_ACC_THRESHOLD:
                low_acc_counter += 1
                log.warning(f'Low accuracy detected ({acc:.2f}%). Consecutive low rounds: {low_acc_counter}/{PATIENCE_LIMIT}')
            else:
                low_acc_counter = 0
            if low_acc_counter >= PATIENCE_LIMIT and os.environ.get('ROBUSTBENCH_DISABLE_LOW_ACC_EARLY_STOP', '0').lower() not in {'1', 'true', 'yes'}:
                log.info(f'Accuracy has been below {LOW_ACC_THRESHOLD}% for {PATIENCE_LIMIT} consecutive rounds. Skipping remaining rounds.')
                break
            gc.collect()
            torch.cuda.empty_cache()
        total_acc_sum = sum([summary_results.get(f'round_{_round}', 0) for _round in range(rounds)])
        summary_results['Avg'] = total_acc_sum / real_rounds if real_rounds > 0 else 0
        log.info(f"[Avg ]: Acc {summary_results['Avg']:.2f} || Error {100 - summary_results['Avg']:.2f}")
        save_df(summary_results, results_root / f'L-CS-LS_{cfg.CORRUPTION.DATASET}_{cfg.CORRUPTION.ORDER_NUM}.csv')
        flush_diagnostics = getattr(tta_model, 'flush_diagnostics', None)
        if callable(flush_diagnostics):
            flush_diagnostics()
        if real_rounds == rounds:
            echo_log.finish()
    finally:
        close = getattr(loader.dataset, 'close', None)
        if callable(close):
            log.info(f'Float32 cache statistics: {loader.dataset.cache_stats}')
            close()

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
    cfg.freeze()
    seed_everything(cfg.SEED)
    set_logger(cfg.LOG_DIR, cfg.ADAPTER.NAME)
    current_file_name = os.path.basename(__file__)
    setproctitle(f'{current_file_name}:{cfg.CORRUPTION.DATASET}:{cfg.ADAPTER.NAME}')
    log.remove()
    log_level = 'DEBUG' if cfg.DEBUG else 'INFO'
    log.add(sys.stderr, level=log_level)
    log.info(f'Loaded configuration file: \n\tadapter: {args.adapter_config_file}\n\tdataset: {args.dataset_config_file}\n\torder: {args.order_config_file}')
    eval(cfg)
if __name__ == '__main__':
    main()
