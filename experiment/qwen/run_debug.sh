#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# HC-SMoE  |  Qwen  |  DEBUG mode: random grouping + uniform average merge
#
# Fast pipeline test: no calibration data, no ZipIt. Use to verify
# load -> group -> merge -> save -> (eval) without waiting for full ZipIt.
#
# Usage:
#   bash experiment/qwen/run_debug.sh [output_dir]
#   NUM_GROUPS=45 bash experiment/qwen/run_debug.sh exp_output/qwen/debug_g45
# ---------------------------------------------------------------------------

set -euo pipefail

export NCCL_P2P_DISABLE=0
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export TOKENIZERS_PARALLELISM="false"
export HF_HOME="${HF_HOME:-/scratch/mz81/huggingface}"
mkdir -p "$HF_HOME"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen1.5-MoE-A2.7B-Chat}"
OUTPUT_BASE="${1:-${OUTPUT_BASE:-/scratch/mz81/hc_smoe/qwen/debug}}"
mkdir -p "$OUTPUT_BASE"
NUM_GROUPS="${NUM_GROUPS:-45}"
PARTITION="${PARTITION:-1}"
ACCEL_CONFIG="${ACCEL_CONFIG:-static/finetune_config.yaml}"
MAIN_PORT="${MAIN_PORT:-29522}"

OUTPUT_DIR="${OUTPUT_BASE}/groups_${NUM_GROUPS}_uniform"
LOG_FILE="${OUTPUT_DIR}/run.log"
mkdir -p "$OUTPUT_DIR"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

echo "=========================================================="
echo " Qwen DEBUG (uniform merge) |  num_groups=${NUM_GROUPS}  ->  ${OUTPUT_DIR}"
echo "=========================================================="

accelerate launch \
    --config_file "$ACCEL_CONFIG" \
    --main_process_port "$MAIN_PORT" \
    hcsmoe/merging-qwen.py \
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
