# ECHO

**Static in Shift Space: Test-Time Adaptation that Remembers**

ECHO adds a fixed-size associative memory to a test-time adapter. An incoming
batch is summarized by a shift key, the memory recalls an adaptation state for
that distribution, the base method takes its usual update, and the update is
written back under the same key. The default key is a **24-dimensional RGB
histogram**: eight fixed bins per color channel.

This release contains the ECHO implementation, Tent/EATA/ROID integrations,
ImageNet corruption and ImageNet-D109 stream evaluators, memory/key controls,
and FLOPs and pairwise-transfer analysis tools.

## Installation

The tested environment is Python 3.8, PyTorch 2.0.1, Torchvision 0.15.2, and CUDA
11.8. Benchmark runners require a CUDA GPU; the small demo and unit tests run on CPU.

```bash
conda create -n echo python=3.8 -y
conda activate echo
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

Run a self-contained example without downloading models or datasets:

```bash
python examples/demo.py
python -m unittest discover -s tests -v
```

The toy demo uses fewer random features and calibration batches to keep it small.
Benchmark defaults are listed below.

## How ECHO works

Let `v` contain the adapted parameters and the optimizer's first moments. For
key `s`, ECHO forms normalized random Fourier features and a shared component:

```text
phi_shared(s) = [sqrt(1-c) * phi(s); sqrt(c)]
read:          v = v_source + A @ phi_shared(s)
base step:     v_next = BaseAdapter(v, batch)
write:         A = decay(A) + (v_next - v) outer phi_shared(s)
```

The memory has a fixed number of columns. RGB keys depend on the input pixels,
and the automatic bandwidth is calibrated from consecutive batch keys. The
calibration updates are replayed into the memory. No test labels, domain
identities, boundary detector or per-domain checkpoint are needed by the default method.

| ECHO setting | Default |
|---|---|
| Shift key | RGB histogram, 3 × 8 bins on `[0,1]` |
| Random features `m` | 4096 |
| Shared weight `c` | 0.3 |
| Bandwidth | Twice the median consecutive-key distance over 79 calibration batches |
| Bandwidth multiplier | 1.0 |
| Local / shared decay | 0.999 / 0.999 per batch |
| Memory content | Adapted parameters and optimizer first moments |
| Seed | 0 |

The central implementation is [methods/ECHO/proposal.py](methods/ECHO/proposal.py).
Fixed pixel descriptors are in [basic_keys.py](methods/ECHO/basic_keys.py).
Network-statistic descriptors and independent-memory controls are included for
ablations; they do not replace the default RGB key.

## Data and pretrained models

Set one portable data root:

```bash
export ECHO_DATA_DIR=/path/to/datasets
```

Expected layout:

```text
datasets/
├── ImageNet/                 # torchvision ImageNet train/meta files for EATA Fisher initialization
├── ImageNet-C/
│   └── motion_blur/5/        # ImageNet-C images are already resized/cropped
│       └── n01440764/*.JPEG
├── ImageNet-3DCC/
│   └── flash/5/n01440764/*.JPEG
├── ImageNet-C-Bar/
│   └── blue_noise/5/n01440764/*.JPEG
└── imagenet-d/
    ├── clipart/...
    ├── infograph/...
    ├── painting/...
    └── sketch/...
```

Other corruption folders follow the names in `configs/order/`. D109 uses the
four public sample lists in `benchmark_assets/asr/`, the 109-class output mask,
and the complete sample lists. Its directory is named `imagenet-d`.
EATA estimates its Fisher regularizer from 2,000 source-training images; the
source ImageNet data are therefore also required for EATA.

Corruption streams use the project's legacy ResNet-50 ImageNet checkpoint
(`resnet50-19c8e357.pth`) and `[0,1]` image tensors without an additional input
normalization layer, preserving the original experiment path. D109 uses
Torchvision ViT-B/16 `IMAGENET1K_V1`, its resize/crop preset, and ImageNet input
normalization. Weight files are downloaded through Torchvision/PyTorch's cache.

Optional `data.pt` and `label.pt` files inside a corruption/severity directory
are supported by the streaming cache readers. Raw-image readers retain each
benchmark's original preprocessing. If caches are used, they must contain the
same preprocessed images, labels, and order; cache preparation is not part of
model adaptation. ImageNet-C also supports a losslessly encoded `data_uint8.pt`.

## Main experiments

The common launcher runs one method, scenario and mode:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run.sh Tent inc echo
CUDA_VISIBLE_DEVICES=0 bash scripts/run.sh Tent inc baseline
CUDA_VISIBLE_DEVICES=0 bash scripts/run.sh EATA 3dcc echo
CUDA_VISIBLE_DEVICES=1 bash scripts/run.sh ROID d109 echo
```

Supported base methods: `Tent`, `EATA`, `ROID`; use `Source ... baseline` for the
frozen pretrained model. ECHO uses the base adapter's loss and optimizer unchanged.

| Scenario | Evaluator | Protocol |
|---|---|---|
| `inc` | `L-CS.py` | ImageNet-C, fixed corruption order |
| `3dcc` | `L-CS.py` | ImageNet-3DCC |
| `cbar` | `L-CS.py` | ImageNet-C-Bar |
| `multi` | `L-CS-MultiBenchmark.py` | IN-C → 3DCC → C-Bar, no model reset at dataset boundaries |
| `shuffle` | `L-CS-Shuffle.py` | ImageNet-C, corruption order shuffled in every pass |
| `imbalance` | `L-CS-LS.py` | Temporal sampler, `LOADER.SAMPLER.GAMMA=0.1` |
| `d109` | `L-IN-D109.py` | Clipart → Infograph → Painting → Sketch, ViT-B/16, learning rate 0.001 |
| `source-error` | `L-CS_Blind-Spot.py` | Source-misclassified ImageNet-C samples |

The default is **20 passes, batch size 64, seed 0, corruption severity 5**.
Corruption runs use 5,000 images per corruption and base learning rate 0.00025.
The D109 runner averages domain accuracies rather than weighting by domain size.
Accuracy uses the predictions returned by the base adapter while it adapts.

Two independent background streams, with separate logs:

```bash
CUDA_VISIBLE_DEVICES=0 nohup bash scripts/run.sh Tent inc echo > tent_inc.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup bash scripts/run.sh EATA inc echo > eata_inc.log 2>&1 &
```

Change one setting with environment variables:

```bash
SEED=1 ROUNDS=20 CUDA_VISIBLE_DEVICES=0 bash scripts/run.sh Tent inc echo
ECHO_SHARE=0 ECHO_KEY_TYPE=network bash scripts/run.sh Tent inc echo
ECHO_INDEX=gate ECHO_SHARE=0.3 bash scripts/run.sh Tent inc echo
ECHO_PRIOR=live ECHO_SHARE=0 bash scripts/run.sh EATA inc echo
ECHO_FEATURES=512 ECHO_BANDWIDTH_SCALE=2 bash scripts/run.sh Tent inc echo
ROUNDS=100 bash scripts/run.sh Tent inc echo
```

Use a different `ECHO_RESULTS_ROOT` for each ablation to keep runs separate.
Outputs are written to `outputs/<method>_<mode>/<scenario>/seed<seed>/` by default:
CSV summaries contain `round_0` ... `round_19` and `Avg`; D109 additionally reports
`First`, `Last` and `visit_1` ... `visit_20`. JSON run records and ECHO diagnostics
are stored in the `runs/` and `diagnostics/` subdirectories.

The Python evaluators also retain the original YAML/override interface:

```bash
python L-CS.py \
  -acfg configs/adapter/imagenet/Tent.yaml \
  -dcfg configs/dataset/imagenet.yaml \
  -ocfg configs/order/imagenet/0.yaml \
  -mcfg configs/models/resnet50.yaml \
  DATA_DIR "$ECHO_DATA_DIR" TEST.ROUNDS 20 TEST.BATCH_SIZE 64 \
  ADAPTER.ECHO.ENABLED True ADAPTER.ECHO.FEATURES 4096 ADAPTER.ECHO.SHARE 0.3
```

The launcher disables low-accuracy early stopping so that collapsed methods
still complete all 20 passes.

## Additional analyses

- **Key comparisons:** set `ECHO_SHARE=0` and vary `ECHO_KEY_TYPE` among
  `rgb-histogram`, `network`, `feature-mean`, `gradient-histogram` and `frequency-energy`.
- **Memory controls:** `L-CS-memory.py`, configured with `ECHO_MEMORY_KIND`,
  `ECHO_MEMORY_CLUSTERS` and `ECHO_MEMORY_TEMPERATURE`; see
  [simple_memories.py](methods/ECHO/simple_memories.py).
- **More frequent domain changes:** `L-CS-Switches.py`, controlled by
  `DCF_SWITCH_SEGMENT_BATCHES`.
- **Pairwise transfer:** `CrossTTA_pairs.py --help` describes the ordered-pair
  evaluator. Use `--n_adapt 32` for the 32-batch experiment, `--features 4096`
  and `--share 0.3` for the current memory configuration.
- **FLOPs:** one warm-up domain pass followed by one measured pass:

```bash
python echo_analysis/flops.py --method Tent --arch resnet50 --echo \
  --features 4096 --share 0 --key-type rgb-histogram \
  --domain motion_blur --samples 5000 --batch-size 64 \
  --output outputs/flops/rgb-histogram.json
```

The counter covers convolution/matrix forward, backward and update operations
with **1 MAC = 2 FLOPs**. FFT costs are reported separately as conventional
estimates. Indexing, histogram binning, normalization and other scalar operations
are excluded. Setup, warm-up and measured-pass costs are recorded separately.

## Repository structure

```text
methods/ECHO/        Associative memory, shift keys, component and memory controls
methods/             Tent, EATA, ROID and frozen Source integrations
configs/             Adapter, model, dataset and corruption-order YAML files
core/                Backbones, temporal sampler, source-data preparation and utilities
robustbench/         Curated corruption loaders and streaming cache helpers
benchmark_assets/    Public D109 sample lists and provenance
scripts/run.sh       Common launcher for the eight stream scenarios
examples/demo.py     Dataset-free CPU example
tests/               CPU checks of memory behavior and reset
```

## Acknowledgements and license

This release builds on Tent, EATA, ROID/test-time-adaptation, RobustBench,
Torchvision and ASR's benchmark lists. Source links, licenses and file provenance
are collected in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
ECHO is released under the [MIT license](LICENSE); third-party components retain
their original license terms. Datasets and model weights are obtained separately.
