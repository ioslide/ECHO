"""Run recorder for the ECHO paper experiments.

Writes one JSON file per run with everything the paper needs: accuracy per
(round, domain), per-batch accuracy (for CCC), wall-clock time per 1,000 images,
peak GPU memory, the ECHO diagnostics and the gate statistics of the
overconfidence-trap experiment.

The run is identified by environment variables set by the experiment scripts:

    ECHO_RUN_TAG   name of the run (no tag -> the recorder does nothing)
    ECHO_LOG_DIR   output directory (default: results/echo_logs)

Integration (see README_ECHO_EXPERIMENTS.md; patch_runner.py inserts these calls):

    import echo_log
    ...
    echo_log.begin_domain(model, domain_name, round_idx)   # before the batches of a domain
    ...
    outputs = model(imgs)
    echo_log.batch(model, outputs, labels)                  # after every forward pass
    ...
    echo_log.finish()                                        # optional (also runs at exit)
"""

import atexit
import json
import os
import time
from pathlib import Path

import torch

__all__ = ['begin_domain', 'batch', 'finish', 'enabled']


def _find_echo(model):
    find_echo = None
    for name in ('methods.ECHO', 'echo'):          # framework location first, then a stand-alone echo.py
        try:
            find_echo = __import__(name, fromlist=['find_echo']).find_echo
            break
        except Exception:  # noqa: BLE001
            continue
    if find_echo is None or model is None:
        return None
    try:
        return find_echo(model)
    except Exception:  # noqa: BLE001
        return None


class _Recorder:
    def __init__(self):
        self.tag = os.environ.get('ECHO_RUN_TAG', '').strip()
        self.dir = Path(os.environ.get('ECHO_LOG_DIR', 'results/echo_logs'))
        self.enabled = bool(self.tag)
        self.round, self.domain = None, None
        self.cells = {}                 # "round|domain" -> [correct, total]
        self.order = []                 # visited (round, domain) in order
        self.step_correct, self.step_total = [], []
        self.gate = {}                  # round -> [batches, live chosen, live better, recalled better]
        self.t_last, self.t_sum, self.n_timed = None, 0.0, 0
        self.n_images = 0
        self.model_ref = None
        self.finished = False
        self.t_start = time.time()
        if self.enabled:
            atexit.register(self.finish)

    # ------------------------------------------------------------------ hooks
    def begin_domain(self, model, domain, round_idx=None):
        echo = _find_echo(model)
        if echo is not None:
            echo.set_context(str(domain), round_idx)
        if not self.enabled:
            return
        self.domain = str(domain)
        self.round = None if round_idx is None else int(round_idx)
        self.order.append([self.round, self.domain])
        self.t_last = None              # loading the next domain is not adaptation time
        if model is not None:
            self.model_ref = model

    def batch(self, model, outputs, labels, domain=None, round_idx=None, domains=None):
        if not self.enabled:
            return
        if domains is not None:
            self.round = int(round_idx)
        elif domain is not None or round_idx is not None:
            if str(domain) != self.domain or round_idx != self.round:
                self.begin_domain(model, domain if domain is not None else self.domain,
                                  round_idx if round_idx is not None else self.round)
        if model is not None:
            self.model_ref = model
        # timing (time between consecutive batches, synchronised)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        now = time.perf_counter()
        if isinstance(outputs, (tuple, list)):
            outputs = outputs[0]
        n = int(outputs.shape[0])
        if self.t_last is not None:
            self.t_sum += now - self.t_last
            self.n_timed += n
        self.t_last = now
        self.n_images += n
        if labels is None:
            return
        labels = labels.detach().to(outputs.device).long().view(-1)
        pred = outputs.detach().argmax(1)
        hits = pred == labels
        correct = int(hits.sum().item())
        self.step_correct.append(correct)
        self.step_total.append(n)
        if domains is None:
            key = f'{self.round}|{self.domain}'
            cell = self.cells.setdefault(key, [0, 0])
            cell[0] += correct
            cell[1] += n
        else:
            # Sample domain IDs are measurement metadata, never adapter inputs.
            for name, mask in domains.items():
                key = f'{self.round}|{name}'
                if key not in self.cells:
                    self.order.append([self.round, str(name)])
                cell = self.cells.setdefault(key, [0, 0])
                cell[0] += int(hits[mask].sum().item())
                cell[1] += int(mask.sum().item())
        # overconfidence trap statistics
        echo = _find_echo(model)
        if echo is not None and echo.memory.gate_log is not None:
            g = echo.memory.gate_log
            lab = labels.cpu()
            acc_live = (g['pred_live'] == lab).float().mean().item()
            acc_rec = (g['pred_rec'] == lab).float().mean().item()
            r = self.gate.setdefault(str(self.round), [0, 0, 0, 0])
            r[0] += 1
            r[1] += int(g['chose_live'])
            r[2] += int(acc_live > acc_rec)
            r[3] += int(acc_rec > acc_live)
            echo.memory.gate_log = None

    def finish(self, complete=False):
        if not self.enabled or self.finished:
            return
        self.finished = True
        echo = _find_echo(self.model_ref) if self.model_ref is not None else None
        out = {
            'schema': 'ECHO_RUN_V1',
            'tag': self.tag,
            # Only the explicit call after successful evaluation marks a run complete.
            # atexit also writes interrupted/failed runs, with complete=False.
            'complete': complete,
            'data_source': 'measured',
            'cells': self.cells,
            'order': self.order,
            'steps': {'correct': self.step_correct, 'total': self.step_total},
            'images': self.n_images,
            'seconds_per_1k_images': (1000.0 * self.t_sum / self.n_timed) if self.n_timed else None,
            'peak_gpu_mb': (torch.cuda.max_memory_allocated() / 2 ** 20) if torch.cuda.is_available() else None,
            'wall_seconds': time.time() - self.t_start,
            'gate': self.gate,
            'echo': echo.diagnostics() if echo is not None else None,
        }
        control = getattr(self.model_ref, 'memory_control_diagnostics', None)
        if callable(control):
            out['memory_control'] = control()
        protocol = os.environ.get('DCF_STREAM_PROTOCOL', '')
        if protocol:
            out['stream_protocol'] = dict(
                name=protocol,
                held_out=list(filter(None, os.environ.get('DCF_HELDOUT_CORRUPTIONS', '').split(','))),
                held_out_first_pass=int(os.environ.get('DCF_HELDOUT_FIRST_PASS', '20')),
            )
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.dir / f'{self.tag}.json.tmp'
        tmp.write_text(json.dumps(out, default=str))
        tmp.replace(self.dir / f'{self.tag}.json')


_REC = _Recorder()
enabled = _REC.enabled


def begin_domain(model, domain, round_idx=None):
    _REC.begin_domain(model, domain, round_idx)


def batch(model, outputs, labels, domain=None, round_idx=None, domains=None):
    _REC.batch(model, outputs, labels, domain, round_idx, domains)


def finish():
    _REC.finish(complete=True)
