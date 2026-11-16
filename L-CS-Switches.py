'ImageNet-C stream with controlled domain-segment lengths.'
import argparse
import copy
import os
from pathlib import Path
import torch
from loguru import logger as log
from setproctitle import setproctitle
import echo_log
import methods
from conf import cfg
from core.model import build_model
from core.utils import save_df, seed_everything, set_logger
from robustbench.data import load_imagenetc

def _load_domains(cfg):
    (xs, ys) = ([], [])
    for corruption in cfg.CORRUPTION.TYPE:
        (x, y) = load_imagenetc(cfg.CORRUPTION.NUM_EX, cfg.CORRUPTION.SEVERITY[0], cfg.DATA_DIR, False, [corruption])
        x255 = x * 255
        if (x255 - x255.round()).abs().max() < 0.001:
            x = x255.round().to(torch.uint8)
        else:
            x = x.half()
        xs.append(x)
        ys.append(y.long())
    return (xs, ys)

def _run(cfg):
    model = build_model(cfg).cuda()
    source_model = copy.deepcopy(model) if cfg.ADAPTER.ECHO.ENABLED else None
    adapter = methods.setup_adapter(model, cfg).cuda()
    if cfg.ADAPTER.ECHO.ENABLED:
        from methods.ECHO import wrap_adapter
        adapter = wrap_adapter(adapter, cfg, source_model=source_model)
    (xs, ys) = _load_domains(cfg)
    batch_size = int(cfg.TEST.BATCH_SIZE)
    segment = int(os.environ['DCF_SWITCH_SEGMENT_BATCHES'])
    full_batches = int(cfg.CORRUPTION.NUM_EX) // batch_size
    if full_batches < 1 or segment < 1:
        raise ValueError('segment and batch size must leave at least one full batch')
    usable = full_batches * batch_size
    log.info(f'segment={segment} batches, usable images/domain={usable}')
    results_root = Path(os.environ.get('DCF_RESULTS_DIR', './results'))
    summary = {'method': cfg.ADAPTER.NAME, 'dataset': cfg.CORRUPTION.DATASET, 'model': cfg.MODEL.ARCH, 'batch_size': batch_size, 'seed': cfg.SEED, 'note': cfg.NOTE, 'segment_batches': segment, 'Avg': 0.0}
    rounds = int(cfg.TEST.ROUNDS)
    for r in range(rounds):
        correct = torch.zeros(len(xs))
        count = torch.zeros(len(xs))
        for first in range(0, full_batches, segment):
            last = min(first + segment, full_batches)
            for (domain, (x, y)) in enumerate(zip(xs, ys)):
                name = cfg.CORRUPTION.TYPE[domain]
                echo_log.begin_domain(adapter, name, r)
                if hasattr(adapter, 'set_domain_info'):
                    adapter.set_domain_info(name, domain)
                (start, stop) = (first * batch_size, last * batch_size)
                (x_chunk, y_chunk) = (x[start:stop], y[start:stop])
                for offset in range(0, len(x_chunk), batch_size):
                    xb = x_chunk[offset:offset + batch_size]
                    xb = xb.float().div_(255.0) if xb.dtype == torch.uint8 else xb.float()
                    yb = y_chunk[offset:offset + batch_size]
                    output = adapter(xb.cuda(), y=yb.cuda())
                    echo_log.batch(adapter, output, yb.cuda())
                    correct[domain] += (output.argmax(1).cpu() == yb).sum().item()
                    count[domain] += len(yb)
        domain_acc = correct / count.clamp_min(1) * 100
        summary[f'round_{r}'] = domain_acc.mean().item()
        summary['Avg'] += summary[f'round_{r}']
        detail = {'method': cfg.ADAPTER.NAME, 'round': r, 'segment_batches': segment, 'Avg': summary[f'round_{r}']}
        detail.update({name: value.item() for (name, value) in zip(cfg.CORRUPTION.TYPE, domain_acc)})
        save_df(detail, results_root / 'detail' / f'switches_{segment}_round_{r}.csv')
        log.info(f"[round_{r}] segment={segment}: Acc {summary[f'round_{r}']:.2f}")
    summary['Avg'] /= max(rounds, 1)
    save_df(summary, results_root / f'switches_{segment}.csv')
    echo_log.finish()
    flush = getattr(adapter, 'flush_diagnostics', None)
    if callable(flush):
        flush()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-acfg', required=True)
    parser.add_argument('-dcfg', required=True)
    parser.add_argument('-ocfg', default='')
    parser.add_argument('-mcfg', default='')
    parser.add_argument('opts', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cfg.merge_from_file(args.acfg)
    cfg.merge_from_file(args.dcfg)
    if args.ocfg:
        cfg.merge_from_file(args.ocfg)
    if args.mcfg:
        cfg.merge_from_file(args.mcfg)
    cfg.merge_from_list(args.opts)
    if cfg.ADAPTER.ECHO.ENABLED:
        from methods.ECHO import configure_from_note
        configure_from_note(cfg)
    cfg.freeze()
    seed_everything(cfg.SEED)
    set_logger(cfg.LOG_DIR, cfg.ADAPTER.NAME)
    setproctitle(f"L-CS-Switches.py:{cfg.ADAPTER.NAME}:segment{os.environ.get('DCF_SWITCH_SEGMENT_BATCHES')}")
    _run(cfg)
if __name__ == '__main__':
    main()
