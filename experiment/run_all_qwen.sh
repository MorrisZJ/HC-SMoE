#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# HC-SMoE  |  Qwen1.5-MoE-A2.7B  |  Run ALL variants sequentially
#
# Produces saved models under saved_models/qwen/:
#
#   saved_models/qwen/
#   ├── freq/groups_{16,24,32,45}/
#   ├── zipit/groups_{16,24,32,45}_ing_act/
#   └── fixdom/groups_{16,24,32,45}/
#
# Models saved in HuggingFace format.  Load in benchmark:
#   AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)
#
# Env-var overrides:
#   MODEL_NAME     (default: Qwen/Qwen1.5-MoE-A2.7B-Chat)
#   MODELS_ROOT    root for all saved models  (default: saved_models/qwen)
#   GROUP_SIZES    space-separated (default: "16 24 32 45")
#   N_SENTENCES    (default: 32)
#   TRAIN_BS       (default: 2)
#   HF_HOME
#   ACCEL_CONFIG   (default: static/finetune_config.yaml)
#
# Usage:
#   bash experiment/run_all_qwen.sh [output_dir]
#   bash experiment/run_all_qwen.sh /scratch/hcsmoe/qwen
#   GROUP_SIZES="45" bash experiment/run_all_qwen.sh /scratch/hcsmoe/qwen
# ---------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# NOTE: OUTPUT_BASE is intentionally NOT exported here; run_variant passes it
# explicitly as a positional arg to each sub-script.
export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen1.5-MoE-A2.7B-Chat}"
export N_SENTENCES="${N_SENTENCES:-32}"
export TRAIN_BS="${TRAIN_BS:-2}"
export ACCEL_CONFIG="${ACCEL_CONFIG:-static/finetune_config.yaml}"

# Positional arg (highest priority) > MODELS_ROOT env var > default
_MODELS_ROOT="${1:-${MODELS_ROOT:-/scratch/mz81/hc_smoe/qwen}}"
mkdir -p "$_MODELS_ROOT"

IFS=' ' read -r -a _SIZES <<< "${GROUP_SIZES:-16 24 32 45}"

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

run_variant "$SCRIPT_DIR/qwen/run_freq.sh"   "freq"         "${_MODELS_ROOT}/freq"
run_variant "$SCRIPT_DIR/qwen/run_zipit.sh"  "zipit"        "${_MODELS_ROOT}/zipit"
run_variant "$SCRIPT_DIR/qwen/run_fixdom.sh" "fix-dom-same" "${_MODELS_ROOT}/fixdom"

echo ""
echo "All Qwen variants complete.  Models saved under: ${REPO_ROOT}/${_MODELS_ROOT}"
