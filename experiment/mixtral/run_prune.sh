#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# HC-SMoE  |  Mixtral-8x7B  |  Merge = prune  (structural expert removal)
#
# Hard-prunes the model: picks dominant experts per layer, physically removes
# the rest from the architecture.  The saved model has fewer experts per layer
# and a smaller gate matrix — fully reflected in config.json.
#
# Two variants:
#   mode=normal      → structural removal (gate + ModuleList rebuilt)
#   mode=zero-output → soft prune (w2 zeroed, arch unchanged — useful ablation)
#
# Usage:
#   bash experiment/mixtral/run_prune.sh [output_dir]
#   bash experiment/mixtral/run_prune.sh /path/to/output
#   NUM_GROUPS=4 PRUNE_MODE=normal bash experiment/mixtral/run_prune.sh /path/to/output
# ---------------------------------------------------------------------------

set -euo pipefail

export NCCL_P2P_DISABLE=0
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export TOKENIZERS_PARALLELISM="false"
export HF_HOME="${HF_HOME:-/scratch/mz81/huggingface}"
mkdir -p "$HF_HOME"

MODEL_NAME="${MODEL_NAME:-mistralai/Mixtral-8x7B-v0.1}"
OUTPUT_BASE="${1:-${OUTPUT_BASE:-/scratch/mz81/hc_smoe/mixtral/prune}}"
mkdir -p "$OUTPUT_BASE"
N_SENTENCES="${N_SENTENCES:-32}"
TRAIN_BS="${TRAIN_BS:-2}"
START_LAYER="${START_LAYER:-0}"
ACCEL_CONFIG="${ACCEL_CONFIG:-static/finetune_config.yaml}"
MAIN_PORT="${MAIN_PORT:-29512}"

# mode: normal (structural) | zero-output (soft)
PRUNE_MODE="${PRUNE_MODE:-normal}"

if [[ -n "${NUM_GROUPS:-}" ]]; then
    GROUP_SIZES=("$NUM_GROUPS")
else
    GROUP_SIZES=(2 3 4 6)
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

for G in "${GROUP_SIZES[@]}"; do
    OUTPUT_DIR="${OUTPUT_BASE}/groups_${G}_mode_${PRUNE_MODE}"
    LOG_FILE="${OUTPUT_DIR}/run.log"
    mkdir -p "$OUTPUT_DIR"

    echo "=========================================================="
    echo " prune  |  num_groups=${G}  mode=${PRUNE_MODE}  ->  ${OUTPUT_DIR}"
    echo "=========================================================="

    accelerate launch \
        --config_file "$ACCEL_CONFIG" \
        --main_process_port "$MAIN_PORT" \
        hcsmoe/merging-mixtral.py \
        --task="no" \
        --model_name="$MODEL_NAME" \
        --dominant="frequency" \
        --similarity_base="router-logits" \
        --merge="prune" \
        --mode="$PRUNE_MODE" \
        --num_average_groups="$G" \
        --n_sentences="$N_SENTENCES" \
        --train_batch_size="$TRAIN_BS" \
        --start_layer="$START_LAYER" \
        --output_path="$OUTPUT_DIR" \
        --result_path="${OUTPUT_DIR}/eval.txt" \
        2>&1 | tee "$LOG_FILE"
done

echo "All prune runs done."
