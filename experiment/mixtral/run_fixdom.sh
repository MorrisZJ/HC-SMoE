#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# HC-SMoE  |  Mixtral-8x7B  |  Merge = fix-dom-same
#
# Two-stage dominant merging: first merge non-dominant experts in each group,
# then merge the result into the dominant expert.
#
# Usage:
#   bash experiment/mixtral/run_fixdom.sh [output_dir]
#   bash experiment/mixtral/run_fixdom.sh /path/to/output
#   NUM_GROUPS=4 bash experiment/mixtral/run_fixdom.sh /path/to/output
# ---------------------------------------------------------------------------

set -euo pipefail

export NCCL_P2P_DISABLE=0
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export TOKENIZERS_PARALLELISM="false"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"

MODEL_NAME="${MODEL_NAME:-mistralai/Mixtral-8x7B-v0.1}"
OUTPUT_BASE="${1:-${OUTPUT_BASE:-saved_models/mixtral/fixdom}}"
mkdir -p "$OUTPUT_BASE"
N_SENTENCES="${N_SENTENCES:-32}"
TRAIN_BS="${TRAIN_BS:-2}"
START_LAYER="${START_LAYER:-0}"
PARTITION="${PARTITION:-1}"
DATA_LIMIT="${DATA_LIMIT:-50000}"
INGREDIENT="${INGREDIENT:-act}"
ACCEL_CONFIG="${ACCEL_CONFIG:-static/finetune_config.yaml}"
MAIN_PORT="${MAIN_PORT:-29513}"

if [[ -n "${NUM_GROUPS:-}" ]]; then
    GROUP_SIZES=("$NUM_GROUPS")
else
    GROUP_SIZES=(2 3 4 6)
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

for G in "${GROUP_SIZES[@]}"; do
    OUTPUT_DIR="${OUTPUT_BASE}/groups_${G}"
    LOG_FILE="${OUTPUT_DIR}/run.log"
    mkdir -p "$OUTPUT_DIR"

    echo "=========================================================="
    echo " fix-dom-same  |  num_groups=${G}  ->  ${OUTPUT_DIR}"
    echo "=========================================================="

    accelerate launch \
        --config_file "$ACCEL_CONFIG" \
        --main_process_port "$MAIN_PORT" \
        hcsmoe/merging-mixtral.py \
        --task="no" \
        --model_name="$MODEL_NAME" \
        --dominant="no" \
        --similarity_base="expert-output" \
        --cluster="hierarchical" \
        --linkage="average" \
        --merge="fix-dom-same" \
        --mode="normal" \
        --ingredient="$INGREDIENT" \
        --num_average_groups="$G" \
        --n_sentences="$N_SENTENCES" \
        --train_batch_size="$TRAIN_BS" \
        --start_layer="$START_LAYER" \
        --partition="$PARTITION" \
        --data_limit="$DATA_LIMIT" \
        --output_path="$OUTPUT_DIR" \
        --result_path="${OUTPUT_DIR}/eval.txt" \
        2>&1 | tee "$LOG_FILE"
done

echo "All fix-dom-same runs done."
