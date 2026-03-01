#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# HC-SMoE  |  Mixtral-8x7B  |  Run ALL variants sequentially
#
# Produces a complete grid of saved models under saved_models/mixtral/:
#
#   saved_models/mixtral/
#   ├── freq/groups_{2,3,4,6}/
#   ├── zipit/groups_{2,3,4,6}_ing_act/
#   ├── fixdom/groups_{2,3,4,6}/
#   └── prune/groups_{2,3,4,6}_mode_normal/
#
# Each sub-directory is a self-contained HuggingFace model:
#   AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)
#
# Env-var overrides (all optional):
#   MODEL_NAME      HF model id or local path  (default: mistralai/Mixtral-8x7B-v0.1)
#   MODELS_ROOT     root for all saved models  (default: saved_models/mixtral)
#   GROUP_SIZES     space-separated list       (default: "2 3 4 6")
#   N_SENTENCES     calibration sentences      (default: 32)
#   TRAIN_BS        calibration batch size     (default: 2)
#   HF_HOME         HuggingFace cache dir
#   ACCEL_CONFIG    accelerate config yaml     (default: static/finetune_config.yaml)
#
# Usage:
#   bash experiment/run_all_mixtral.sh [output_dir]
#   bash experiment/run_all_mixtral.sh /scratch/hcsmoe/mixtral
#   GROUP_SIZES="4" bash experiment/run_all_mixtral.sh /scratch/hcsmoe/mixtral
# ---------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# NOTE: OUTPUT_BASE is intentionally NOT exported here.  Each sub-script owns
# its own per-variant default; run_variant passes it explicitly as a positional arg.
export MODEL_NAME="${MODEL_NAME:-mistralai/Mixtral-8x7B-v0.1}"
export N_SENTENCES="${N_SENTENCES:-32}"
export TRAIN_BS="${TRAIN_BS:-2}"
export ACCEL_CONFIG="${ACCEL_CONFIG:-static/finetune_config.yaml}"

# Positional arg (highest priority) > MODELS_ROOT env var > default
_MODELS_ROOT="${1:-${MODELS_ROOT:-/scratch/mz81/hc_smoe/mixtral}}"
mkdir -p "$_MODELS_ROOT"

IFS=' ' read -r -a _SIZES <<< "${GROUP_SIZES:-2 3 4 6}"

# run_variant <script> <variant_name> <output_base>
# Passes output_base as the positional arg expected by each sub-script.
run_variant() {
    local script="$1"
    local variant_name="$2"
    local output_base="$3"
    echo ""
    echo "##################################################################"
    echo "##  Starting variant: ${variant_name}  ->  ${output_base}"
    echo "##################################################################"
    for G in "${_SIZES[@]}"; do
        NUM_GROUPS="$G" bash "$script" "$output_base"
    done
}

cd "$REPO_ROOT"

run_variant "$SCRIPT_DIR/mixtral/run_freq.sh"   "freq"              "${_MODELS_ROOT}/freq"
run_variant "$SCRIPT_DIR/mixtral/run_zipit.sh"  "zipit"             "${_MODELS_ROOT}/zipit"
run_variant "$SCRIPT_DIR/mixtral/run_fixdom.sh" "fix-dom-same"      "${_MODELS_ROOT}/fixdom"
run_variant "$SCRIPT_DIR/mixtral/run_prune.sh"  "prune (structural)" "${_MODELS_ROOT}/prune"

echo ""
echo "All variants complete.  Models saved under: ${REPO_ROOT}/${_MODELS_ROOT}"
