"""One-domain FLOP measurements for ECHO paper Tables 1 and 3.

Count convolution and matrix arithmetic in the actual forward/backward/update
path, including ECHO's mv/addr operations (one MAC = two FLOPs). SVD, QR and
eigh and FFTs use conventional algorithm estimates, reported separately. Tensor
indexing, copies, elementwise operations and normalization are not counted.
Setup, a same-domain warm-up, and the measured pass are recorded separately.
"""

import argparse
from collections import Counter
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils._python_dispatch import TorchDispatchMode


class Flops(TorchDispatchMode):
    def __init__(self):
        self.arithmetic = Counter()
        self.decompositions = Counter()
        self.transforms = Counter()
        self.unhandled_compute = Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        name = str(func).split('.')[1]
        n = 0
        if name in ('mm', 'bmm', 'addmm', 'addmm_', 'baddbmm', 'baddbmm_', 'addbmm'):
            a, b = args[:2] if name in ('mm', 'bmm') else args[1:3]
            n = 2 * a.numel() * b.shape[-1]
        elif name in ('mv', 'addmv'):
            n = 2 * args[0 if name == 'mv' else 1].numel()
        elif name in ('addr', 'addr_'):
            n = 2 * args[1].numel() * args[2].numel()
        elif name == 'dot':
            n = 2 * args[0].numel()
        elif name in ('convolution', '_convolution', 'cudnn_convolution'):
            x, w = args[:2]
            transposed = name != 'cudnn_convolution' and args[6]
            n = 2 * (x.numel() if transposed else out.numel()) * w.shape[1] * math.prod(w.shape[2:])
        elif name == 'convolution_backward':
            grad, x, w = args[:3]
            n = 2 * (x.numel() if args[7] else grad.numel()) * w.shape[1] * math.prod(w.shape[2:])
            n *= sum(args[10][:2])
        elif name == '_native_multi_head_attention':
            q, k = args[:2]
            batch, length, dim = q.shape
            source = k.shape[1]
            n = 4 * batch * (length + source) * dim ** 2 + 4 * batch * length * source * dim
        elif name in ('_linalg_svd', 'linalg_svd', 'linalg_qr', '_linalg_eigh', 'linalg_eigh'):
            shape = args[0].shape
            batch = math.prod(shape[:-2])
            m, k = max(shape[-2:]), min(shape[-2:])
            if 'svd' in name:
                estimate = batch * (4 * m * k ** 2 + 8 * k ** 3)
            elif 'qr' in name:
                estimate = batch * (2 * m * k ** 2 - 2 * k ** 3 / 3)
            else:
                estimate = batch * 9 * k ** 3
            self.decompositions[name] += round(estimate)
        elif name == '_fft_c2c':
            x, dims = args[:2]
            self.transforms[name] += round(5 * x.numel() * sum(math.log2(x.shape[d]) for d in dims))
        elif ((name == 'linear' or any(key in name for key in ('attention', 'convolution', 'matmul', 'linalg', 'fft')))
              and name not in ('_linalg_check_errors', 'linalg_vector_norm', 'linalg_matrix_norm',
                               'linalg_norm', 'fft_fftfreq', 'fft_rfftfreq')):
            self.unhandled_compute[name] += 1
        self.arithmetic[name] += n
        return out

    def report(self):
        return dict(mac_flops=sum(self.arithmetic.values()),
                    dense_linalg_estimated_flops=sum(self.decompositions.values()),
                    fft_estimated_flops=sum(self.transforms.values()),
                    operators={k: v for k, v in self.arithmetic.items() if v},
                    dense_linalg=dict(self.decompositions),
                    fft=dict(self.transforms),
                    unhandled_compute=dict(self.unhandled_compute))


def measure(args):
    import methods
    from conf import cfg as defaults
    from core.model import build_model
    from core.utils import seed_everything
    from methods.ECHO import wrap_adapter
    from robustbench.pt_stream import TensorCacheStream

    torch.set_num_threads(4)
    cfg = defaults.clone()
    folder = 'imagenet_d109_asr' if args.method in ('VPT', 'DPCore') else 'imagenet'
    cfg.merge_from_file(str(ROOT / f'configs/adapter/{folder}/{args.method}.yaml'))
    cfg.merge_from_file(str(ROOT / 'configs/dataset/imagenet.yaml'))
    cfg.merge_from_file(str(ROOT / f'configs/models/{args.arch}.yaml'))
    cfg.SEED = 0
    cfg.TEST.BATCH_SIZE, cfg.TEST.ROUNDS = args.batch_size, 1
    cfg.CORRUPTION.TYPE, cfg.CORRUPTION.SEVERITY = [args.domain], [5]
    cfg.CORRUPTION.NUM_EX = args.samples
    if args.arch == 'resnet50':
        cfg.OPTIM.LR = 0.00025
    if args.method in ('VPT', 'DPCore'):
        cfg.OPTIM.METHOD, cfg.OPTIM.LR = 'AdamW', 0.01
    if args.method == 'VPT':
        cfg.ADAPTER.VPT.ADAPTATION = 'source_alignment' if args.echo else 'none'
    cfg.ADAPTER.ECHO.merge_from_list([
        'ENABLED', args.echo, 'FEATURES', args.features, 'BANDWIDTH', 0.0,
        'BANDWIDTH_SCALE', 1.0, 'SHARE', args.share, 'CALIBRATION', 79,
        'GLOBAL_DECAY', 0.999, 'LOCAL_DECAY', 0.999,
        'KEY_TYPE', args.key_type, 'PARAMS_ONLY', False,
        'KEY_STATS', 'source', 'KEY_COORDS', 'fisher', 'KEY_LAYERS', 'first',
        'KEY_DEPTH', 1 / 3, 'PRIOR', 'source', 'INDEX', 'kernel', 'SEED', 0,
    ])
    cfg.NOTE = 'paper_flops_one_domain'
    cfg.freeze()
    seed_everything(cfg.SEED)
    print(cfg, flush=True)
    cache = Path(cfg.DATA_DIR) / 'ImageNet-C' / args.domain / '5' / 'data.pt'
    data = TensorCacheStream(cache)

    def batches():
        for x in data.batches(args.batch_size, args.samples):
            x = x.cuda().float()
            if data.dtype.name == 'uint8':
                x.div_(255)
            yield x

    # Math attention exposes both matrix products to the counter on torch 2.0.
    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
        setup = Flops()
        with setup:
            source = build_model(cfg).cuda()
            frozen = copy.deepcopy(source) if args.echo else None
            adapter = methods.setup_adapter(source, cfg).cuda()
            if args.echo:
                adapter = wrap_adapter(adapter, cfg, source_model=frozen)
            del frozen
        reports = {}
        for phase in ('warmup', 'measured'):
            counter = Flops()
            count = 0
            started = time.monotonic()
            with counter:
                for i, x in enumerate(batches(), 1):
                    if args.method == 'CurvatureTTA_plus':
                        adapter(x, None)  # Its required y argument is unused.
                    else:
                        adapter(x)
                    count += len(x)
                    if i == 1 or i % 10 == 0:
                        print(f'[{phase}] batch={i} images={count}', flush=True)
            torch.cuda.synchronize()
            reports[phase] = dict(counter.report(), images=count, batches=i,
                                  wall_seconds=time.monotonic() - started)
            print(f'[{phase}-done] {json.dumps(reports[phase])}', flush=True)
    measured = reports['measured']
    complete = measured['images'] == args.samples and not measured['unhandled_compute']
    return dict(complete=complete, table=1 if args.arch == 'resnet50' else 3,
                method=args.method, echo=args.echo, arch=args.arch, domain=args.domain,
                seed=0, batch_size=args.batch_size, features=args.features,
                key_type=args.key_type, share=args.share, rounds=1, config=cfg.dump(),
                protocol='one warmup pass, then one measured pass of the same corruption',
                convention='1 MAC = 2 FLOPs; convolution/matrix forward and backward; scalar/index/copy operations excluded',
                setup=setup.report(), **reports,
                gflops_per_image=measured['mac_flops'] / measured['images'] / 1e9,
                dense_linalg_estimated_gflops_per_image=measured['dense_linalg_estimated_flops'] / measured['images'] / 1e9,
                fft_estimated_gflops_per_image=measured['fft_estimated_flops'] / measured['images'] / 1e9,
                fft_convention='complex FFT: 5*N*sum(log2(axis lengths)); conventional estimate, not an exact cuFFT instruction count',
                code_sha256={str(p): hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                             for p in map(Path, ('echo_analysis/flops.py', 'conf.py', 'methods/ECHO/proposal.py',
                                                 'methods/ECHO/basic_keys.py', 'methods/ECHO/simple_memories.py'))})


def summarize(directory):
    rows = []
    for path in sorted(directory.glob('*.json')):
        d = json.loads(path.read_text())
        measured, warmup, setup = d['measured'], d['warmup'], d['setup']
        row = {k: d[k] for k in ('table', 'method', 'echo', 'arch', 'complete')}
        row.update(
            mac_gflops_per_image=measured['mac_flops'] / measured['images'] / 1e9,
            dense_linalg_estimated_gflops_per_image=measured['dense_linalg_estimated_flops'] / measured['images'] / 1e9,
            total_estimated_gflops_per_image=(measured['mac_flops'] + measured['dense_linalg_estimated_flops']) / measured['images'] / 1e9,
            warmup_total_estimated_gflops_per_image=(warmup['mac_flops'] + warmup['dense_linalg_estimated_flops']) / warmup['images'] / 1e9,
            setup_total_estimated_gflops=(setup['mac_flops'] + setup['dense_linalg_estimated_flops']) / 1e9,
        )
        rows.append(row)
    with (directory / 'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=('table', 'method', 'echo', 'arch', 'complete',
                                             'mac_gflops_per_image', 'dense_linalg_estimated_gflops_per_image',
                                             'total_estimated_gflops_per_image', 'warmup_total_estimated_gflops_per_image',
                                             'setup_total_estimated_gflops'))
        writer.writeheader()
        writer.writerows(rows)
    print(f'[summary] {len(rows)} records in {directory / "summary.csv"}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method')
    p.add_argument('--arch', choices=('resnet50', 'vit_b_16'))
    p.add_argument('--echo', action='store_true')
    p.add_argument('--domain', default='motion_blur')
    p.add_argument('--samples', type=int, default=5000)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--features', type=int, default=4096)
    p.add_argument('--share', type=float, default=0.3)
    p.add_argument('--key-type', choices=('feature-mean', 'network', 'gradient-histogram',
                                         'frequency-energy', 'rgb-histogram'), default='rgb-histogram')
    p.add_argument('--output', type=Path)
    p.add_argument('--summarize', type=Path)
    args = p.parse_args()
    if args.summarize:
        summarize(args.summarize)
        return
    result = measure(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print('[result]', json.dumps({k: v for k, v in result.items() if k not in ('config', 'setup', 'warmup', 'measured')}), flush=True)
    if not result['complete']:
        raise RuntimeError('Incomplete measurement; inspect unhandled_compute in the output JSON')


if __name__ == '__main__':
    main()
