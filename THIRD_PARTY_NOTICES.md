# Third-party acknowledgements

The ECHO implementation is distributed under the root MIT license. The following
integrations and utilities retain their upstream notices and license terms.
The release adapts their interfaces to a common continual-evaluation pipeline.

| Component | Upstream | Included license | Used by |
|---|---|---|---|
| Tent | https://github.com/DequanWang/tent | [MIT](third_party/licenses/tent.txt) | `methods/Tent.py` |
| EATA | https://github.com/mr-eggplant/EATA | [MIT](third_party/licenses/eata.txt) | `methods/EATA.py` |
| ROID / test-time-adaptation | https://github.com/mariodoebler/test-time-adaptation | [MIT](third_party/licenses/roid.txt) | `methods/ROID/`, source-data preparation, ImageNet class mappings |
| RobustBench | https://github.com/RobustBench/robustbench | [Upstream license](third_party/licenses/robustbench.txt) | `robustbench/` corruption readers and normalization utilities |
| Torchvision | https://github.com/pytorch/vision | [BSD-3-Clause](third_party/licenses/torchvision.txt) | `core/model/resnet.py`, model construction and transforms |
| ASR benchmark lists | https://github.com/YonseiML/asr | Upstream benchmark metadata; no root license was provided at retrieval | `benchmark_assets/asr/` |

The ASR file provenance, commit and checksums are recorded in
[benchmark_assets/asr/source.json](benchmark_assets/asr/source.json).
Only filename/class-label lists are included; images remain subject to the
licenses and access terms of ImageNet and DomainNet/ImageNet-D109.

No pretrained model weights or dataset images are distributed in this repository.
Torchvision downloads its own pretrained weights when a benchmark is started.
