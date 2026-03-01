# HC-SMoE Experiments

Scripts that produce **standardised HuggingFace checkpoints** from each HC-SMoE variant.
Every saved directory — **both Mixtral and Qwen** — is loadable with the exact same call:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(path)
```

No manual registration, no custom loading code, no model-specific handling in the benchmark.

---

## Directory layout

```
experiment/
├── README.md                  ← this file
├── run_all_mixtral.sh         ← orchestrator: all Mixtral variants
├── run_all_qwen.sh            ← orchestrator: all Qwen variants
├── mixtral/
│   ├── run_freq.sh            ← freq (usage-weighted average)
│   ├── run_zipit.sh           ← zipit (activation-matching, main HC-SMoE)
│   ├── run_fixdom.sh          ← fix-dom-same (dominant-guided two-stage)
│   └── run_prune.sh           ← hard structural pruning
└── qwen/
    ├── run_freq.sh
    ├── run_zipit.sh
    └── run_fixdom.sh
```

After running, saved models appear under `saved_models/`:

```
saved_models/
├── mixtral/
│   ├── freq/groups_{2,3,4,6}/
│   ├── zipit/groups_{2,3,4,6}_ing_act/
│   ├── fixdom/groups_{2,3,4,6}/
│   └── prune/groups_{2,3,4,6}_mode_normal/
└── qwen/
    ├── freq/groups_{16,24,32,45}/
    ├── zipit/groups_{16,24,32,45}_ing_act/
    └── fixdom/groups_{16,24,32,45}/
```

---

## Quick start

### Mixtral — run everything

```bash
bash experiment/run_all_mixtral.sh
# single compression level:
GROUP_SIZES="4" bash experiment/run_all_mixtral.sh
```

### Mixtral — run a single variant

```bash
bash experiment/mixtral/run_zipit.sh                 # all default group sizes: 2 3 4 6
NUM_GROUPS=4 bash experiment/mixtral/run_zipit.sh    # single compression level
MODEL_NAME="mistralai/Mixtral-8x7B-v0.1" \
NUM_GROUPS=4 N_SENTENCES=64 TRAIN_BS=4 \
OUTPUT_BASE="my_output/mixtral" \
bash experiment/mixtral/run_zipit.sh
```

### Mixtral — debug (random grouping + uniform merge)

Fast pipeline test, no ZipIt/calibration:

```bash
bash experiment/mixtral/run_debug.sh
NUM_GROUPS=4 bash experiment/mixtral/run_debug.sh saved_models/mixtral/debug
```

### Qwen — run everything

```bash
bash experiment/run_all_qwen.sh
# single compression level:
GROUP_SIZES="45" bash experiment/run_all_qwen.sh
```

### Qwen — run a single variant

```bash
bash experiment/qwen/run_zipit.sh
NUM_GROUPS=45 bash experiment/qwen/run_zipit.sh
```

---

## Key environment variables

| Variable | Default | Description |
|---|---|---|
| `MODEL_NAME` | `mistralai/Mixtral-8x7B-v0.1` | HF model id or local path |
| `OUTPUT_BASE` | `saved_models/mixtral/<variant>` | Root directory for saved models |
| `NUM_GROUPS` | (all of 2 3 4 6) | Target number of expert groups per layer |
| `GROUP_SIZES` | `"2 3 4 6"` | Space-separated list (run_all only) |
| `N_SENTENCES` | `32` | Calibration sentences from C4 |
| `TRAIN_BS` | `2` | Batch size for calibration forward pass |
| `INGREDIENT` | `act` | Feature for zipit/fixdom: `act`, `weight`, `act+weight` |
| `PRUNE_MODE` | `normal` | `normal` (structural) or `zero-output` (soft) |
| `PARTITION` | `1` | Partition ratio for low-VRAM; increase to 2/4 if OOM |
| `DATA_LIMIT` | `50000` | Max tokens for zipit activation collection |
| `HF_HOME` | `~/.cache/huggingface` | HuggingFace cache directory |
| `ACCEL_CONFIG` | `static/finetune_config.yaml` | Accelerate config (GPU layout) |

---

## Variant descriptions

### `freq` — Frequency-weighted average
Groups experts by output similarity (hierarchical clustering), then merges each
group into a single expert via usage-frequency-weighted parameter averaging.
Fast, no activation collection required for the merge step.

### `zipit` — ZipIt activation-matching *(main HC-SMoE method)*
Uses pairwise neuron correlation across experts to compute a permutation/merge
matrix before averaging.  Requires a calibration forward pass to collect
activations.  `ingredient=act` is the paper default.

### `fix-dom-same` — Dominant-guided two-stage merge
Selects a dominant expert per group, merges non-dominant experts into it in two
stages.  Uses the same ZipIt-style activation matching internally.

### `prune` — Structural expert pruning
Selects dominant experts by global usage frequency and physically removes the
rest: the gate linear layer is rebuilt with fewer outputs and the `ModuleList`
is shortened.  The saved model has a genuinely smaller architecture.  The
`zero-output` mode variant keeps the architecture intact but zeros out the `w2`
projection of pruned experts (useful as a soft-pruning ablation).

---

---

## Qwen-specific notes

### Expert counts
Qwen1.5-MoE-A2.7B has **64 experts** per layer (vs 8 for Mixtral).
Default group sweep: `16 24 32 45` (paper default = 45, i.e. ~30% reduction).

### Save format
Qwen uses the native HF `Qwen2MoeForCausalLM` (no custom modeling class).
`merging-qwen.py` saves with the standard `model.save_pretrained()`.
`trust_remote_code=True` is accepted by HF even when there is no `auto_map` in the config,
so the benchmark can safely use the same call for both Mixtral and Qwen.

### No prune script
Structural pruning (`merge=prune`) changes the gate dimension and is not yet
covered by a unified save/load path for Qwen.  No `run_prune.sh` is provided.

---

## Key environment variables

| Variable | Mixtral default | Qwen default | Description |
|---|---|---|---|
| `MODEL_NAME` | `mistralai/Mixtral-8x7B-v0.1` | `Qwen/Qwen1.5-MoE-A2.7B-Chat` | HF model id or local path |
| `OUTPUT_BASE` | `saved_models/mixtral/<variant>` | `saved_models/qwen/<variant>` | Root for saved models |
| `NUM_GROUPS` | (all of 2 3 4 6) | (all of 16 24 32 45) | Target groups per layer |
| `GROUP_SIZES` | `"2 3 4 6"` | `"16 24 32 45"` | Space-separated (run_all only) |
| `N_SENTENCES` | `32` | `32` | Calibration sentences from C4 |
| `TRAIN_BS` | `2` | `2` | Calibration batch size |
| `INGREDIENT` | `act` | `act` | Feature for zipit/fixdom |
| `PRUNE_MODE` | `normal` | — | Mixtral only |
| `DATA_LIMIT` | `50000` | `1000000` | Max tokens for zipit |
| `PARTITION` | `1` | `1` | Increase to 2/4 if OOM |
| `HF_HOME` | `~/.cache/huggingface` | same | HF cache directory |
| `ACCEL_CONFIG` | `static/finetune_config.yaml` | same | Accelerate config |

---

## Notes

- All scripts use `--task="no"` — evaluation is handled by the benchmark repo,
  not by HC-SMoE inline.
- Each script creates its output directory automatically (`mkdir -p`).
- Logs are written to `<output_dir>/run.log` alongside the model files.
- `--start_layer=0` merges all layers.  Set e.g. `--start_layer=8` to leave
  the first 8 layers untouched (sometimes helps preserve performance).
