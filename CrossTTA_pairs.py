"Protocol-matched continual-Tent transfer evaluation for Fig. 3(b).\n\nFor every ordered pair (A, B), adapt on eight batches of A, evaluate A on\nheld-out images, adapt on eight batches of B, and evaluate A again.  The\nresult is the change on A caused by adapting on B, matching the paper's\npairwise protocol rather than CrossTTA's full-domain protocol.\n"
import argparse
import copy
import json
from pathlib import Path
import torch
from echo_analysis.pairs import CORRUPTIONS, accuracy, adapt, cfg_for, load_corruption, source_model, to_float
import methods.Tent as tent

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--cache', required=True)
    p.add_argument('--start', type=int, required=True)
    p.add_argument('--end', type=int, required=True)
    p.add_argument('--severity', type=int, default=5)
    p.add_argument('--bs', type=int, default=64)
    p.add_argument('--n_adapt', type=int, default=8)
    p.add_argument('--n_heldout', type=int, default=512)
    p.add_argument('--lr', type=float, default=0.00025)
    p.add_argument('--momentum', type=float, default=0.9)
    p.add_argument('--features', type=int, default=512)
    p.add_argument('--share', type=float, default=0.2)
    p.add_argument('--decay', type=float, default=0.999)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--device', default='cuda')
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    args.device = str(device)
    n_total = args.n_adapt * args.bs + args.n_heldout
    data = {c: load_corruption(args, c, n_total) for c in CORRUPTIONS}
    src = source_model(device)
    rows = []
    for ai in range(args.start, args.end):
        a = CORRUPTIONS[ai]
        (xa, ya) = data[a]
        held_x = xa[args.n_adapt * args.bs:args.n_adapt * args.bs + args.n_heldout]
        held_y = ya[args.n_adapt * args.bs:args.n_adapt * args.bs + args.n_heldout]
        for b in CORRUPTIONS:
            if a == b:
                continue
            torch.manual_seed(args.seed)
            cfg = cfg_for(args, False, 0.0)
            model = tent.setup(copy.deepcopy(src), cfg)
            adapt(model, xa, args)
            acc_after_a = accuracy(model, held_x, held_y, args)
            adapt(model, data[b][0], args)
            acc_after_b = accuracy(model, held_x, held_y, args)
            rows.append({'A': a, 'B': b, 'acc_after_A': acc_after_a, 'acc_after_B': acc_after_b, 'delta': acc_after_b - acc_after_a})
        print(f'{ai + 1}/{len(CORRUPTIONS)} {a} done', flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows))
if __name__ == '__main__':
    main()
