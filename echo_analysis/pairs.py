"""Pairwise interference and shift-key geometry on ImageNet-C (Fig. 3b, 3c, 6a).

Stand-alone script (does not use the L-CS runners): torchvision ResNet-50,
the same Tent as the framework (tent.py) and ECHO (echo.py).

For every ordered pair (A, B) of the 15 corruptions and for continual Tent and
Tent + ECHO: start from the source model, adapt on N_ADAPT batches of A,
measure accuracy on held-out images of A, adapt on N_ADAPT batches of B,
measure A again. delta = acc(A after B) - acc(A after A), in points.
kappa(A, B) is computed with ECHO's own random features.

Outputs (in --out):
  pairs_tent.json, pairs_echo.json   one record per ordered pair
  keys.json                           2-D PCA of keys (KEYS_BATCHES per corruption)
  pairs_meta.json                     bandwidth and settings

Usage:
  python echo_analysis/pairs.py --data_dir /path/to/ImageNet-C --out results/echo_pairs \
      [--bandwidth_from results/echo_logs/CSC__imagenet__resnet50__bs64__s0__Tent__echo.json]
"""
import argparse
import copy
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, os.getcwd())
from conf import cfg as defaults  # noqa: E402
import methods.ECHO.proposal as echo  # noqa: E402
import methods.Tent as tent  # noqa: E402

CORRUPTIONS = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur',
               'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate',
               'jpeg_compression']
FAMILY = [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True, help='ImageNet-C root: <root>/<corruption>/<severity>/<wnid>/*.JPEG')
    ap.add_argument('--severity', type=int, default=5)
    ap.add_argument('--out', default='results/echo_pairs')
    ap.add_argument('--cache', default='results/echo_pairs/cache')
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--n_adapt', type=int, default=8, help='adaptation batches per shift')
    ap.add_argument('--n_heldout', type=int, default=512, help='held-out images per shift')
    ap.add_argument('--keys_batches', type=int, default=30, help='batches per corruption for the key plot')
    ap.add_argument('--lr', type=float, default=2.5e-4)
    ap.add_argument('--momentum', type=float, default=0.9)
    ap.add_argument('--features', type=int, default=512)
    ap.add_argument('--share', type=float, default=0.2)
    ap.add_argument('--decay', type=float, default=0.999)
    ap.add_argument('--bandwidth', type=float, default=0.0, help='0 = automatic (see --bandwidth_from)')
    ap.add_argument('--bandwidth_from', default='', help='ECHO run log whose bandwidth is reused')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return ap.parse_args()


# --------------------------------------------------------------------------- data
def load_corruption(args, name, n_total):
    """Fixed subset of n_total images of one corruption, cached as uint8."""
    cache = Path(args.cache) / f'{name}_s{args.severity}_n{n_total}_seed{args.seed}.pt'
    if cache.exists():
        return torch.load(cache)
    from torchvision import datasets, transforms
    ds = datasets.ImageFolder(os.path.join(args.data_dir, name, str(args.severity)),
                              transform=transforms.Compose([transforms.CenterCrop(224), transforms.PILToTensor()]))
    idx = np.random.default_rng(args.seed).permutation(len(ds))[:n_total]
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, idx.tolist()), batch_size=128,
                                         num_workers=args.workers, shuffle=False)
    xs, ys = [], []
    for x, y in loader:
        xs.append(x)
        ys.append(y)
    data = (torch.cat(xs), torch.cat(ys))
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, cache)
    return data


class Normalized(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        return self.net((x - self.mean) / self.std)


def source_model(device):
    from torchvision.models import resnet50, ResNet50_Weights
    return Normalized(resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)).to(device).eval()


def to_float(x, device):
    return x.to(device, non_blocking=True).float().div_(255.0)


def cfg_for(args, enabled, bandwidth):
    cfg = defaults.clone()
    cfg.NOTE = ''
    cfg.TEST.BATCH_SIZE = args.bs
    cfg.ADAPTER.ECHO.merge_from_list([
        'ENABLED', enabled, 'FEATURES', args.features, 'BANDWIDTH', bandwidth,
        'CALIBRATION', 2, 'SEED', args.seed, 'SHARE', args.share,
        'GLOBAL_DECAY', args.decay, 'LOCAL_DECAY', args.decay,
        'KEY_TYPE', 'network',
    ])
    cfg.OPTIM.LR, cfg.OPTIM.MOMENTUM = args.lr, args.momentum
    return cfg


def build(args, src, enabled, bandwidth):
    c = cfg_for(args, enabled, bandwidth)
    adapter = tent.setup(copy.deepcopy(src), c)
    return echo.wrap_adapter(adapter, c, source_model=src)


@torch.no_grad()
def accuracy(model, x_u8, y, args):
    correct = 0
    for i in range(0, len(y), args.bs):
        x = to_float(x_u8[i:i + args.bs], args.device)
        out = model(x, adapt=False)
        correct += (out.argmax(1).cpu() == y[i:i + args.bs]).sum().item()
    return 100.0 * correct / len(y)


def adapt(model, x_u8, args):
    for i in range(args.n_adapt):
        model(to_float(x_u8[i * args.bs:(i + 1) * args.bs], args.device))


def main():
    args = parse()
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n_adapt_img = args.n_adapt * args.bs
    n_total = n_adapt_img + args.n_heldout + args.keys_batches * args.bs
    data = {c: load_corruption(args, c, n_total) for c in CORRUPTIONS}
    src = source_model(args.device)
    key = echo.ShiftKey(src)

    # ---------------------------------------------------------------- keys and bandwidth
    keys = {}
    for c in CORRUPTIONS:
        x_u8 = data[c][0][n_adapt_img + args.n_heldout:]
        keys[c] = torch.stack([key(to_float(x_u8[i * args.bs:(i + 1) * args.bs], args.device))
                               for i in range(args.keys_batches)])
    h = args.bandwidth
    if h <= 0 and args.bandwidth_from:
        h = float(json.load(open(args.bandwidth_from))['echo']['bandwidth'])
    if h <= 0:  # automatic: twice the median distance of consecutive same-shift keys
        gaps = torch.cat([(k[1:] - k[:-1]).norm(dim=1) for k in keys.values()])
        h = 2.0 * gaps.median().item()
    print(f'bandwidth h = {h:.4f}', flush=True)

    allk = torch.cat([keys[c] for c in CORRUPTIONS]).float().cpu()
    centred = allk - allk.mean(0, keepdim=True)
    _, _, V = torch.linalg.svd(centred, full_matrices=False)
    pc = centred @ V[:2].T
    recs = []
    for ci, c in enumerate(CORRUPTIONS):
        for b in range(args.keys_batches):
            p = pc[ci * args.keys_batches + b]
            recs.append(dict(x=float(p[0]), y=float(p[1]), dom=ci, family=FAMILY[ci], name=c))
    json.dump(recs, open(out / 'keys.json', 'w'))

    # ---------------------------------------------------------------- kernel between shifts
    probe = build(args, src, True, h)
    probe.memory._calibrate([keys[CORRUPTIONS[0]][0]])
    feats = {c: torch.stack([probe.memory._features(k) for k in keys[c]]) for c in CORRUPTIONS}
    kappa = {(a, b): float((feats[a] * feats[b]).sum(1).mean()) for a in CORRUPTIONS for b in CORRUPTIONS}
    del probe

    # ---------------------------------------------------------------- pairs
    for method in ('tent', 'echo'):
        path = out / f'pairs_{method}.json'
        if path.exists():
            print(f'[skip] {path}')
            continue
        rows = []
        for a in CORRUPTIONS:
            xa, ya = data[a]
            held_x, held_y = xa[n_adapt_img:n_adapt_img + args.n_heldout], ya[n_adapt_img:n_adapt_img + args.n_heldout]
            for b in CORRUPTIONS:
                if a == b:
                    continue
                torch.manual_seed(args.seed)
                model = build(args, src, method == 'echo', h)
                adapt(model, xa, args)
                acc_a = accuracy(model, held_x, held_y, args)
                adapt(model, data[b][0], args)
                acc_after = accuracy(model, held_x, held_y, args)
                rows.append(dict(A=a, B=b, acc_after_A=acc_a, acc_after_B=acc_after,
                                 delta=acc_after - acc_a, kappa=kappa[(a, b)]))
            print(f'{method}: {a} done', flush=True)
        json.dump(rows, open(path, 'w'))
    json.dump(dict(bandwidth=h, features=args.features, share=args.share, decay=args.decay,
                   calibration=2, n_adapt=args.n_adapt, bs=args.bs,
                   n_heldout=args.n_heldout, lr=args.lr, seed=args.seed), open(out / 'pairs_meta.json', 'w'))


if __name__ == '__main__':
    main()
