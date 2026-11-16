<div align="center">

<img src="assets/echo-logo.svg" alt="ECHO" width="440">

### Static in Shift Space: Test-Time Adaptation that Remembers

**Recall the shift. Reuse the update.**

<a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&amp;logo=pytorch&amp;logoColor=white" alt="PyTorch"></a>
<a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&amp;logo=python&amp;logoColor=white" alt="Python"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-2563EB?style=flat-square" alt="MIT license"></a>

[Quick Start](#quick-start) · [Benchmarks](#benchmarks) · [Code Map](#code-map)

</div>

---

ECHO equips test-time adaptation with a **fixed-size associative memory**.
RGB histogram keys describe each incoming batch; the memory recalls an adaptation
state, the base method takes its usual step, and the update is written back under
the same key. Recurring shifts can build on what the adapter learned before.

**RGB histogram keys · Similarity-weighted recall · Fixed memory budget**

## Quick Start

Install the dependencies and try the dataset-free CPU demo:

```bash
pip install -r requirements.txt
python examples/demo.py
```

Benchmark runs require a CUDA GPU. Set the data root and launch an experiment:

```bash
export ECHO_DATA_DIR=/path/to/datasets
bash scripts/run.sh Tent inc echo
```

The launcher supports **Tent**, **EATA**, and **ROID**. Replace `echo` with
`baseline` to run the base adapter. Results and diagnostics are saved in `outputs/`.
Configurations live in [configs/](configs/).

<details>
<summary><b>Dataset layout</b></summary>

```text
datasets/
├── ImageNet/
├── ImageNet-C/
├── ImageNet-3DCC/
├── ImageNet-C-Bar/
└── imagenet-d/
    ├── clipart/
    ├── infograph/
    ├── painting/
    └── sketch/
```

Corruption folders follow the orders in [configs/order/](configs/order/).
EATA also uses ImageNet training data for Fisher initialization. D109 sample lists
and their provenance are included in [benchmark_assets/asr/](benchmark_assets/asr/).
Model construction and preprocessing are defined in [core/model/](core/model/).

</details>

## Benchmarks

Choose a scenario with `bash scripts/run.sh <method> <scenario> <echo|baseline>`.

| Scenario | Stream |
| :--- | :--- |
| `inc` | ImageNet-C |
| `3dcc` | ImageNet-3DCC |
| `cbar` | ImageNet-C-Bar |
| `multi` | IN-C → 3DCC → C-Bar |
| `shuffle` | ImageNet-C with shuffled domain order |
| `imbalance` | ImageNet-C with temporal label imbalance |
| `d109` | ImageNet-D109 natural domain shifts |
| `source-error` | Source-misclassified ImageNet-C samples |

## Code Map

| Location | Contents |
| :--- | :--- |
| [methods/ECHO/](methods/ECHO/) | Associative memory, shift keys, and ablations |
| [methods/](methods/) | Base adapter integrations |
| [scripts/run.sh](scripts/run.sh) | Unified experiment launcher |
| [echo_analysis/](echo_analysis/) | Pairwise-transfer and FLOPs tools |
| [tests/](tests/) | CPU checks of memory behavior and reset |

Memory controls and domain-switch experiments are available through
[L-CS-memory.py](L-CS-memory.py) and [L-CS-Switches.py](L-CS-Switches.py).

## Acknowledgements & License

Built on Tent, EATA, ROID, RobustBench, Torchvision, and ASR benchmark assets.
See [third-party acknowledgements](THIRD_PARTY_NOTICES.md) for attribution and
upstream licenses. ECHO is released under the [MIT license](LICENSE).
