#!/usr/bin/env bash
# One-click: download Mixtral to model_zoo and convert to SDPA (no flash-attn) version.
# Optionally push to HuggingFace (default: morriszjm). For gated model or push: run `huggingface-cli login` first.
#
# Usage:
#   bash scripts/run_convert_mixtral_sdpa.sh
#   OUTPUT_DIR=/path/to/save bash scripts/run_convert_mixtral_sdpa.sh
#   PUSH_TO_HUB=1 bash scripts/run_convert_mixtral_sdpa.sh
#   PUSH_TO_HUB=1 HF_REPO=Mixtral-8x7B-v0.1-sdpa bash scripts/run_convert_mixtral_sdpa.sh
#   FROM_LOCAL=/path/to/Mixtral-8x7B-v0.1 bash scripts/run_convert_mixtral_sdpa.sh  # convert existing, no download
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OUTPUT_DIR="${OUTPUT_DIR:-/mnt/scratch/model_zoo/Mixtral-8x7B-v0.1-sdpa}"
FROM_LOCAL="${FROM_LOCAL:-}"
PUSH_TO_HUB="${PUSH_TO_HUB:-0}"
HF_USERNAME="${HF_USERNAME:-morriszjm}"
HF_REPO="${HF_REPO:-}"

OPTS=(--output_dir "$OUTPUT_DIR")
[[ -n "${FROM_LOCAL:-}" ]] && OPTS+=(--from_local "$FROM_LOCAL")
if [[ "${PUSH_TO_HUB}" == "1" ]]; then
  OPTS+=(--push_to_hub --hf_username "$HF_USERNAME")
  [[ -n "${HF_REPO:-}" ]] && OPTS+=(--hf_repo "$HF_REPO")
fi

python scripts/convert_mixtral_to_sdpa.py "${OPTS[@]}"
