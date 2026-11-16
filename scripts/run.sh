#!/usr/bin/env bash
# Single stream; ECHO configuration is shared across the released base methods.
set -euo pipefail
cd "$(dirname "$0")/.."
method="${1:-Tent}"
scenario="${2:-inc}"
mode="${3:-echo}"
case "$method" in Tent|EATA|ROID|Source) ;; *) echo 'Method: Tent/EATA/ROID/Source' >&2; exit 2 ;; esac
case "$mode" in echo) enabled=True ;; baseline) enabled=False ;; *) echo 'Mode: echo/baseline' >&2; exit 2 ;; esac
if [[ "$method" == Source && "$mode" == echo ]]; then echo 'Source uses baseline mode.' >&2; exit 2; fi
runner=L-CS.py
dataset=imagenet
adapter_dataset=imagenet
arch=resnet50
lr=0.00025
extra=()
case "$scenario" in
    inc) ;;
    3dcc) dataset=imagenet_3dcc ;;
    cbar) dataset=imagenet_c_bar ;;
    multi) runner=L-CS-MultiBenchmark.py ;;
    shuffle) runner=L-CS-Shuffle.py ;;
    imbalance) runner=L-CS-LS.py; extra+=(LOADER.SAMPLER.TYPE temporal LOADER.SAMPLER.GAMMA 0.1) ;;
    source-error) runner=L-CS_Blind-Spot.py; extra+=(INPUT.BLIND_SPOT_MODE SOURCE_ERROR) ;;
    d109) runner=L-IN-D109.py; dataset=imagenet_d109_asr; adapter_dataset=imagenet_d109_asr; arch=vit_b_16; lr=0.001; extra+=(--workers 4) ;;
    *) echo 'Scenario: inc/3dcc/cbar/multi/shuffle/imbalance/d109/source-error' >&2; exit 2 ;;
esac
order=()
[[ "$scenario" == d109 ]] || order=(-ocfg "configs/order/${dataset}/0.yaml")
export PYTHONUNBUFFERED=1
export ROBUSTBENCH_DISABLE_LOW_ACC_EARLY_STOP=1
export ROBUSTBENCH_FAST_PT_STREAM=1
export ROBUSTBENCH_UINT8_PT_STREAM=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
seed="${SEED:-0}"
export DCF_RESULTS_DIR="${ECHO_RESULTS_ROOT:-outputs}/${method}_${mode}/${scenario}/seed${seed}"
export ECHO_LOG_DIR="$DCF_RESULTS_DIR/runs"
export ECHO_RUN_TAG="${method}_${mode}_${scenario}_seed${seed}"
cmd=(
    "${PYTHON_BIN:-python}" "$runner"
    -acfg "configs/adapter/${adapter_dataset}/${method}.yaml"
    -dcfg "configs/dataset/${dataset}.yaml"
    "${order[@]}"
    -mcfg "configs/models/${arch}.yaml"
    "${extra[@]}"
    SEED "$seed" OPTIM.LR "$lr"
    TEST.BATCH_SIZE "${BATCH_SIZE:-64}" TEST.ROUNDS "${ROUNDS:-20}"
    ADAPTER.ECHO.ENABLED "$enabled"
    ADAPTER.ECHO.FEATURES "${ECHO_FEATURES:-4096}"
    ADAPTER.ECHO.SHARE "${ECHO_SHARE:-0.3}"
    ADAPTER.ECHO.BANDWIDTH_SCALE "${ECHO_BANDWIDTH_SCALE:-1.0}"
    ADAPTER.ECHO.KEY_TYPE "${ECHO_KEY_TYPE:-rgb-histogram}"
    ADAPTER.ECHO.INDEX "${ECHO_INDEX:-kernel}"
    ADAPTER.ECHO.PRIOR "${ECHO_PRIOR:-source}"
    ADAPTER.ECHO.DIAGNOSTIC_PATH "$DCF_RESULTS_DIR/diagnostics/echo.json"
    NOTE "$ECHO_RUN_TAG" LOG_DIR "$DCF_RESULTS_DIR/logs"
)
if [[ "$scenario" != d109 ]]; then cmd+=(CORRUPTION.NUM_EX 5000 CORRUPTION.SEVERITY '[5]'); fi
if [[ "${DRY_RUN:-0}" == 1 ]]; then printf '%q ' "${cmd[@]}"; printf '\n'; exit 0; fi
mkdir -p "$ECHO_LOG_DIR" "$DCF_RESULTS_DIR/diagnostics"
exec "${cmd[@]}"
