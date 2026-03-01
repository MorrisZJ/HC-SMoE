#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# HC-SMoE  |  Mixtral-8x7B  |  DEBUG mode: random grouping + uniform average merge
#
# Fast pipeline test: no calibration data, no ZipIt. Use to verify
# load -> group -> merge -> save -> (eval) without waiting for full ZipIt.
#
# Usage:
#   bash experiment/mixtral/run_debug.sh [output_dir]
#   NUM_GROUPS=4 bash experiment/mixtral/run_debug.sh exp_output/mixtral/debug_g4
# ---------------------------------------------------------------------------

set -euo pipefail

export NCCL_P2P_DISABLE=0
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export TOKENIZERS_PARALLELISM="false"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p "$HF_HOME"

MODEL_NAME="${MODEL_NAME:-mistralai/Mixtral-8x7B-v0.1}"
OUTPUT_BASE="${1:-${OUTPUT_BASE:-saved_models/mixtral/debug}}"
mkdir -p "$OUTPUT_BASE"
NUM_GROUPS="${NUM_GROUPS:-4}"
PARTITION="${PARTITION:-1}"
ACCEL_CONFIG="${ACCEL_CONFIG:-static/finetune_config.yaml}"
MAIN_PORT="${MAIN_PORT:-29513}"

OUTPUT_DIR="${OUTPUT_BASE}/groups_${NUM_GROUPS}_uniform"
LOG_FILE="${OUTPUT_DIR}/run.log"
mkdir -p "$OUTPUT_DIR"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

echo "=========================================================="
echo " Mixtral DEBUG (uniform merge) |  num_groups=${NUM_GROUPS}  ->  ${OUTPUT_DIR}"
echo "=========================================================="

accelerate launch \
    --config_file "$ACCEL_CONFIG" \
    --main_process_port "$MAIN_PORT" \
    hcsmoe/merging-mixtral.py \
    --task="no" \
    --model_name="$MODEL_NAME" \
    --dominant="random" \
    --similarity_base="expert-output" \
    --cluster="hierarchical" \
    --linkage="average" \
    --merge="uniform" \
    --mode="normal" \
    --ingredient="act" \
    --num_average_groups="$NUM_GROUPS" \
    --n_sentences=4 \
    --train_batch_size=2 \
    --start_layer=0 \
    --partition="$PARTITION" \
    --data_limit=1000 \
    --output_path="$OUTPUT_DIR" \
    --result_path="${OUTPUT_DIR}/eval.txt" \
    2>&1 | tee "$LOG_FILE"

echo "Debug run done: $OUTPUT_DIR"
