#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# HC-SMoE  |  Mixtral-8x7B  |  Merge = freq (usage-frequency-weighted avg)
#
# Produces one saved model per NUM_GROUPS value.  Each output directory is
# self-contained (HuggingFace format + copied modeling files) and can be
# loaded directly with:
#   AutoModelForCausalLM.from_pretrained(OUTPUT_DIR, trust_remote_code=True)
#
# Usage:
#   bash experiment/mixtral/run_freq.sh [output_dir]
#   bash experiment/mixtral/run_freq.sh /path/to/output
#   NUM_GROUPS=4 bash experiment/mixtral/run_freq.sh /path/to/output
# ---------------------------------------------------------------------------

set -euo pipefail

# ── Environment ─────────────────────────────────────────────────────────────
export NCCL_P2P_DISABLE=0
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export TOKENIZERS_PARALLELISM="false"
export HF_HOME="${HF_HOME:-/scratch/mz81/huggingface}"
mkdir -p "$HF_HOME"

# ── Configurable defaults ────────────────────────────────────────────────────
MODEL_NAME="${MODEL_NAME:-mistralai/Mixtral-8x7B-v0.1}"
# output_dir: positional arg (highest priority) > OUTPUT_BASE env var > default
OUTPUT_BASE="${1:-${OUTPUT_BASE:-/scratch/mz81/hc_smoe/mixtral/freq}}"
mkdir -p "$OUTPUT_BASE"
N_SENTENCES="${N_SENTENCES:-32}"
TRAIN_BS="${TRAIN_BS:-2}"
START_LAYER="${START_LAYER:-0}"
ACCEL_CONFIG="${ACCEL_CONFIG:-static/finetune_config.yaml}"
MAIN_PORT="${MAIN_PORT:-29510}"

# Group sizes to sweep over (override with NUM_GROUPS=<n> env var)
if [[ -n "${NUM_GROUPS:-}" ]]; then
    GROUP_SIZES=("$NUM_GROUPS")
else
    GROUP_SIZES=(2 3 4 6)
fi

# ── Run ──────────────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

for G in "${GROUP_SIZES[@]}"; do
    OUTPUT_DIR="${OUTPUT_BASE}/groups_${G}"
    LOG_FILE="${OUTPUT_DIR}/run.log"
    mkdir -p "$OUTPUT_DIR"

    echo "=========================================================="
    echo " freq merge  |  num_groups=${G}  ->  ${OUTPUT_DIR}"
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
        --merge="freq" \
        --num_average_groups="$G" \
        --n_sentences="$N_SENTENCES" \
        --train_batch_size="$TRAIN_BS" \
        --start_layer="$START_LAYER" \
        --output_path="$OUTPUT_DIR" \
        --result_path="${OUTPUT_DIR}/eval.txt" \
        2>&1 | tee "$LOG_FILE"
done

echo "All freq runs done."
