'ASR IN-D109: Clipart -> Infograph -> Painting -> Sketch, 20 visits.\n\nUses ImageNet pretrained weights, the official 109-class mask and full lists.\nTEST.ROUNDS controls visits; adapter state persists across domains and visits.\n'
import argparse
import copy
import csv
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from loguru import logger as log
from torch.utils.data import DataLoader, Dataset
import echo_log
from conf import cfg
from core.utils import seed_everything, save_df, set_logger
from setproctitle import setproctitle
ROOT = Path(__file__).resolve().parent
LISTS = ROOT / 'benchmark_assets/asr'
D109_DOMAINS = ('clipart', 'infograph', 'painting', 'sketch')

class ListedImages(Dataset):

    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        (path, label) = self.samples[index]
        with Image.open(path) as image:
            image = self.transform(image.convert('RGB'))
        return (image, label)

def d109_samples(data_root, domain):
    """Official labels are already in the 109-class output-mask order."""
    root = Path(data_root) / 'imagenet-d'
    rows = (LISTS / f'{domain}_list.txt').read_text().splitlines()
    return [(str(root / path), int(label)) for (path, label) in (row.split() for row in rows)]

def prepare_samples(samples, seed, num_examples):
    random.seed(seed)
    np.random.seed(seed)
    samples = list(samples)
    random.shuffle(samples)
    return samples

def make_loader(samples, cfg, transform, workers=4):
    samples = prepare_samples(samples, cfg.SEED, cfg.CORRUPTION.NUM_EX)
    dataset = ListedImages(samples, transform)
    return DataLoader(dataset, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False, drop_last=False, num_workers=workers, pin_memory=True)

def d109_transform(arch):
    """ASR uses the pretrained torchvision model's resize/crop/interpolation."""
    import torchvision.models as models
    preset = models.get_model_weights(arch).IMAGENET1K_V1.transforms()
    return transforms.Compose([transforms.Resize(preset.resize_size, interpolation=preset.interpolation), transforms.CenterCrop(preset.crop_size), transforms.ToTensor()])

def ensure_input_normalization(model, cfg):
    """Normalize raw backbones while preserving project feature-split APIs."""
    from robustbench.model_zoo.architectures.utils_architectures import ImageNormalizer
    if any((isinstance(module, ImageNormalizer) for module in model.modules())):
        return model
    for (name, module) in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            (parent_name, _, attribute) = name.rpartition('.')
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, attribute, torch.nn.Sequential(ImageNormalizer(cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_STD), module))
            return model
    raise ValueError('Backbone has no ImageNormalizer or input convolution')

def setup_adapter(cfg):
    import methods
    from core.model import build_model
    model = ensure_input_normalization(build_model(cfg), cfg).cuda()
    source_model = copy.deepcopy(model) if bool(cfg.ADAPTER.ECHO.ENABLED) else None
    adapter = methods.setup_adapter(model, cfg).cuda()
    if cfg.ADAPTER.ECHO.ENABLED:
        from methods.ECHO import wrap_adapter
        adapter = wrap_adapter(adapter, cfg, source_model=source_model)
    return adapter.cuda()

def evaluate_domain(model, loader, device='cuda', print_every=100):
    correct = torch.zeros((), device=device)
    seen = 0
    with torch.no_grad():
        for (index, (images, labels)) in enumerate(loader, 1):
            labels = labels.to(device, non_blocking=True)
            outputs = model(images.to(device, non_blocking=True), labels)
            echo_log.batch(model, outputs, labels)
            correct += (outputs.argmax(1) == labels).float().sum()
            seen += len(labels)
    count = int(correct.item())
    return {'samples': seen, 'correct': count, 'Acc': 100.0 * count / seen}

def evaluate_sequence(model, domains, loader_factory, visits=1, device='cuda', print_every=100, write_domain=None):
    """Initialize/reset once, then retain model state across all domain visits."""
    reset = getattr(model, 'reset', None)
    if callable(reset):
        reset()
    records = []
    visit_means = []
    low_accuracy_visits = 0
    for visit in range(1, visits + 1):
        for (domain_index, domain) in enumerate(domains):
            echo_log.begin_domain(model, domain, visit)
            stats = evaluate_domain(model, loader_factory(domain), device, print_every)
            record = dict(visit=visit, domain=domain, domain_index=domain_index, **stats)
            records.append(record)
            if write_domain is not None:
                write_domain(record)
            log.info(f"[visit {visit}/{visits}] {domain}: Acc={stats['Acc']:.4f}% ({stats['correct']}/{stats['samples']})")
        visit_accuracy = float(np.mean([r['Acc'] for r in records if r['visit'] == visit]))
        visit_means.append(visit_accuracy)
        log.info(f'[visit {visit}/{visits}] Avg Acc={visit_accuracy:.4f}%')
        low_accuracy_visits = low_accuracy_visits + 1 if visit_accuracy < 5.0 else 0
        if low_accuracy_visits > 2 and os.environ.get('ROBUSTBENCH_DISABLE_LOW_ACC_EARLY_STOP', '0').lower() not in {'1', 'true', 'yes'}:
            log.info(f'Early stopping after {visit} visits: average accuracy below 5% for {low_accuracy_visits} consecutive visits')
            break
    samples = sum((r['samples'] for r in records))
    correct = sum((r['correct'] for r in records))
    summary = {'Avg': float(np.mean([r['Acc'] for r in records])), 'First': visit_means[0], 'Last': visit_means[-1], 'MicroAvg': 100.0 * correct / samples, 'samples': samples, 'correct': correct, 'visits': len(visit_means)}
    summary.update({f'visit_{v}': value for (v, value) in enumerate(visit_means, 1)})
    return summary

def eval(cfg, workers=4, print_every=100):
    if tuple(cfg.CORRUPTION.TYPE) != D109_DOMAINS:
        raise ValueError('ASR IN-D109 order is clipart, infograph, painting, sketch')
    if cfg.CORRUPTION.DATASET != 'imagenet_d109' or cfg.CORRUPTION.NUM_CLASS != 109:
        raise ValueError('IN-D109 requires the ImageNet model with a 109-class output mask')
    base_results = dict(method=cfg.ADAPTER.NAME, model=cfg.MODEL.ARCH, seed=cfg.SEED, batch_size=cfg.TEST.BATCH_SIZE, note=cfg.NOTE, echo=bool(cfg.ADAPTER.ECHO.ENABLED))
    log.info(f'base_results = {base_results}')
    tta_model = setup_adapter(cfg)
    transform = d109_transform(cfg.MODEL.ARCH)

    def loader(domain):
        samples = d109_samples(cfg.DATA_DIR, domain)
        return make_loader(samples, cfg, transform, workers)
    results_root = Path(os.environ.get('DCF_RESULTS_DIR', ROOT / 'results'))
    detail_root = results_root / 'detail'
    detail_root.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    detail_path = detail_root / f'L-IN-D109_{cfg.ADAPTER.NAME}_{cfg.MODEL.ARCH}_seed{cfg.SEED}_{run_id}.csv'
    log.info(f'L-IN-D109: domains={D109_DOMAINS}, visits={cfg.TEST.ROUNDS}, samples/domain={cfg.CORRUPTION.NUM_EX}, seed={cfg.SEED}')
    try:
        with detail_path.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=['visit', 'domain', 'domain_index', 'samples', 'correct', 'Acc'])
            writer.writeheader()

            def write_domain(record):
                writer.writerow(record)
                handle.flush()
            summary_results = evaluate_sequence(tta_model, D109_DOMAINS, loader, visits=cfg.TEST.ROUNDS, print_every=print_every, write_domain=write_domain)
        summary_results.update(base_results)
        save_df(summary_results, results_root / 'L-IN-D109.csv')
        log.info(f"L-IN-D109: First={summary_results['First']:.4f}%, Last={summary_results['Last']:.4f}%, Avg={summary_results['Avg']:.4f}%")
        echo_log.finish()
        return summary_results
    finally:
        flush_diagnostics = getattr(tta_model, 'flush_diagnostics', None)
        if callable(flush_diagnostics):
            flush_diagnostics()

def main():
    log.info(f'Running file: {os.path.basename(__file__)}')
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('-acfg', '--adapter-config-file', metavar='FILE', type=str, default=str(ROOT / 'configs/adapter/imagenet/Tent.yaml'), help='path to adapter config file')
    parser.add_argument('-dcfg', '--dataset-config-file', metavar='FILE', type=str, default=str(ROOT / 'configs/dataset/imagenet_d109_asr.yaml'), help='path to dataset config file')
    parser.add_argument('-mcfg', '--model-config-file', metavar='FILE', type=str, default=str(ROOT / 'configs/models/vit_b_16.yaml'), help='path to model config file')
    parser.add_argument('-ecfg', '--echo-config-file', metavar='FILE', type=str, default='', help='ECHO defaults applied before command-line overrides')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--print-every', type=int, default=100)
    parser.add_argument('opts', nargs='*', help='modify the configuration by command line')
    args = parser.parse_args()
    cfg.CORRUPTION.NUM_EX = -1
    cfg.merge_from_file(args.adapter_config_file)
    cfg.merge_from_file(args.dataset_config_file)
    if args.model_config_file:
        cfg.merge_from_file(args.model_config_file)
    if args.echo_config_file:
        cfg.merge_from_file(args.echo_config_file)
    cfg.merge_from_list(args.opts)
    if cfg.ADAPTER.ECHO.ENABLED:
        from methods.ECHO import configure_from_note
        configure_from_note(cfg)
    cfg.freeze()
    seed_everything(cfg.SEED)
    if os.environ.get('DCF_DETERMINISTIC_ATTENTION', '0') == '1':
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        torch.use_deterministic_algorithms(True)
    log.remove()
    log.add(sys.stderr, level='DEBUG' if cfg.DEBUG else 'INFO')
    set_logger(cfg.LOG_DIR, f'ASR_protocol_{cfg.ADAPTER.NAME}')
    current_file_name = os.path.basename(__file__)
    setproctitle(f'{current_file_name}:{cfg.CORRUPTION.DATASET}:{cfg.ADAPTER.NAME}')
    log.info(f'Run settings: method={cfg.ADAPTER.NAME}, model={cfg.MODEL.ARCH}, dataset={cfg.CORRUPTION.DATASET}, seed={cfg.SEED}, batch_size={cfg.TEST.BATCH_SIZE}, visits={cfg.TEST.ROUNDS}, ECHO={cfg.ADAPTER.ECHO.ENABLED}, NEO={False}\nNOTE: {cfg.NOTE}')
    log.info(f'Model settings:\n{cfg.MODEL}')
    log.info(f'Optimizer settings:\n{cfg.OPTIM}')
    log.info(f"Deterministic attention: {os.environ.get('DCF_DETERMINISTIC_ATTENTION', '0') == '1'}")
    if cfg.ADAPTER.NAME in cfg.ADAPTER:
        log.info(f'{cfg.ADAPTER.NAME} settings:\n{cfg.ADAPTER[cfg.ADAPTER.NAME]}')
    if cfg.ADAPTER.ECHO.ENABLED:
        log.info(f'ECHO settings: {cfg.ADAPTER.ECHO}')
    log.info(f'Loaded configuration file:\n\tadapter: {args.adapter_config_file}\n\tdataset: {args.dataset_config_file}\n\tmodel: {args.model_config_file}')
    if args.echo_config_file:
        log.info(f'ECHO defaults: {args.echo_config_file}')
    eval(cfg, workers=args.workers, print_every=args.print_every)
if __name__ == '__main__':
    main()
