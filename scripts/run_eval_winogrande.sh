#!/usr/bin/env bash
# Run 0-shot Winogrande eval on a saved model (merged or baseline).
#
# Usage:
#   bash scripts/run_eval_winogrande.sh <model_dir> [output_file]
#   bash scripts/run_eval_winogrande.sh /scratch/mz81/hc_smoe/qwen/zipit/groups_45_ing_act
#   bash scripts/run_eval_winogrande.sh /path/to/model /path/to/result.txt
#
# Optional: output_file as $2, or env OUTPUT=... (both override default <model_dir>/eval_winogrande.txt)
#   TASK=winogrande NUM_FEWSHOT=0 bash scripts/run_eval_winogrande.sh <model_dir>

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODEL_DIR="${1:-}"
if [[ -z "$MODEL_DIR" ]]; then
  echo "Usage: $0 <model_dir_or_hf_id> [output_file]"
  echo "  Local: $0 /path/to/model"
  echo "  HF:    $0 morriszjm/Mixtral-8x7B-v0.1-sdpa"
  exit 1
fi
# Allow HuggingFace model ID (e.g. user/repo); only require local dir when it looks like a path
if [[ "$MODEL_DIR" == /* || "$MODEL_DIR" == ./* ]]; then
  if [[ ! -d "$MODEL_DIR" ]]; then
    echo "Error: model_dir not found: $MODEL_DIR"
    exit 1
  fi
fi

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
mkdir -p "$HF_HOME"

OUTPUT_FILE="${2:-${OUTPUT:-}}"
python scripts/eval_winogrande.py "$MODEL_DIR" \
  --task="${TASK:-winogrande}" \
  --num_fewshot="${NUM_FEWSHOT:-0}" \
  --batch_size="${BATCH_SIZE:-4}" \
  ${OUTPUT_FILE:+--output "$OUTPUT_FILE"}
