#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# HC-SMoE  |  Qwen1.5-MoE-A2.7B  |  Merge = fix-dom-same
#
# Saved in HuggingFace format.  Load in benchmark:
#   AutoModelForCausalLM.from_pretrained(output_path, trust_remote_code=True)
#
# Usage:
#   bash experiment/qwen/run_fixdom.sh [output_dir]
#   bash experiment/qwen/run_fixdom.sh /path/to/output
#   NUM_GROUPS=45 bash experiment/qwen/run_fixdom.sh /path/to/output
# ---------------------------------------------------------------------------

set -euo pipefail

export NCCL_P2P_DISABLE=0
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export TOKENIZERS_PARALLELISM="false"
export HF_HOME="${HF_HOME:-/scratch/mz81/huggingface}"
mkdir -p "$HF_HOME"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen1.5-MoE-A2.7B-Chat}"
OUTPUT_BASE="${1:-${OUTPUT_BASE:-/scratch/mz81/hc_smoe/qwen/fixdom}}"
mkdir -p "$OUTPUT_BASE"
N_SENTENCES="${N_SENTENCES:-32}"
TRAIN_BS="${TRAIN_BS:-2}"
START_LAYER="${START_LAYER:-0}"
PARTITION="${PARTITION:-1}"
DATA_LIMIT="${DATA_LIMIT:-1000000}"
INGREDIENT="${INGREDIENT:-act}"
ACCEL_CONFIG="${ACCEL_CONFIG:-static/finetune_config.yaml}"
MAIN_PORT="${MAIN_PORT:-29522}"

if [[ -n "${NUM_GROUPS:-}" ]]; then
    GROUP_SIZES=("$NUM_GROUPS")
else
    GROUP_SIZES=(16 24 32 45)
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

for G in "${GROUP_SIZES[@]}"; do
    OUTPUT_DIR="${OUTPUT_BASE}/groups_${G}"
    LOG_FILE="${OUTPUT_DIR}/run.log"
    mkdir -p "$OUTPUT_DIR"

    echo "=========================================================="
    echo " Qwen fix-dom-same  |  num_groups=${G}  ->  ${OUTPUT_DIR}"
    echo "=========================================================="

    accelerate launch \
        --config_file "$ACCEL_CONFIG" \
        --main_process_port "$MAIN_PORT" \
        hcsmoe/merging-qwen.py \
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

echo "All Qwen fix-dom-same runs done."
