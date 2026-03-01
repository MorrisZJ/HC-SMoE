#!/usr/bin/env bash
# Run 0-shot Winogrande eval on a saved model (merged or baseline).
#
# Usage:
#   bash scripts/run_eval_winogrande.sh <model_dir>
#   bash scripts/run_eval_winogrande.sh /scratch/mz81/hc_smoe/qwen/zipit/groups_45_ing_act
#
# Optional env / args passed through to the Python script:
#   TASK=winogrande NUM_FEWSHOT=0 bash scripts/run_eval_winogrande.sh <model_dir>
#   OUTPUT=/path/to/result.txt bash scripts/run_eval_winogrande.sh <model_dir>

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODEL_DIR="${1:-}"
if [[ -z "$MODEL_DIR" ]]; then
  echo "Usage: $0 <model_dir>"
  echo "  e.g. $0 /scratch/mz81/hc_smoe/qwen/zipit/groups_45_ing_act"
  exit 1
fi

if [[ ! -d "$MODEL_DIR" ]]; then
  echo "Error: model_dir not found: $MODEL_DIR"
  exit 1
fi

export HF_HOME="${HF_HOME:-/scratch/mz81/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/scratch/mz81/huggingface/datasets}"
mkdir -p "$HF_HOME"

python scripts/eval_winogrande.py "$MODEL_DIR" \
  --task="${TASK:-winogrande}" \
  --num_fewshot="${NUM_FEWSHOT:-0}" \
  --batch_size="${BATCH_SIZE:-4}" \
  ${OUTPUT:+--output "$OUTPUT"}
