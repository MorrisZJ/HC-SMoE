# Scripts: Run & Eval

Bash and Python scripts for **merge/prune** and **evaluation**. Default outputs go under `/scratch/mz81/hc_smoe/` and caches under `/scratch/mz81/huggingface/`.

---

## 1. General — Bash examples

All commands assume you are in repo root: `cd /path/to/HC-SMoE`. On HPC, run merge jobs on a compute node (e.g. after `salloc` + `srun --jobid=JOBID --pty bash`). Set `LD_PRELOAD` if you hit OpenSSL errors:  
`export LD_PRELOAD=$CONDA_PREFIX/lib/libcrypto.so.3:$CONDA_PREFIX/lib/libssl.so.3`

**Multi-GPU (e.g. 8 cards):** Use an Accelerate config with `num_processes: 8` and set `ACCEL_CONFIG` to that YAML, or create `static/finetune_config_8gpu.yaml` with `num_processes: 8` and run e.g. `ACCEL_CONFIG=static/finetune_config_8gpu.yaml NUM_GROUPS=45 bash experiment/qwen/run_zipit.sh`. Slurm example: `salloc -p commons --gres=gpu:h200:8 --time=24:00:00 --cpus-per-task=64 --mem=256G`.

### Merge (Qwen)

```bash
# ZipIt (paper main method) — 45 groups, output under /scratch/.../qwen/zipit/
NUM_GROUPS=45 bash experiment/qwen/run_zipit.sh

# Debug (random grouping + uniform average) — fast pipeline test
NUM_GROUPS=45 bash experiment/qwen/run_debug.sh

# Freq (usage-weighted average, no ZipIt)
NUM_GROUPS=45 bash experiment/qwen/run_freq.sh

# Fix-dom-same (dominant-guided two-stage merge)
NUM_GROUPS=45 bash experiment/qwen/run_fixdom.sh
```

### Merge (Mixtral)

```bash
# ZipIt — default group sizes 2 3 4 6; single size:
NUM_GROUPS=4 bash experiment/mixtral/run_zipit.sh

# Freq (usage-weighted average)
NUM_GROUPS=4 bash experiment/mixtral/run_freq.sh

# Fix-dom-same
NUM_GROUPS=4 bash experiment/mixtral/run_fixdom.sh

# Prune (structural expert removal); PRUNE_MODE=normal | zero-output
NUM_GROUPS=4 PRUNE_MODE=normal bash experiment/mixtral/run_prune.sh
```

### Run all variants (Qwen / Mixtral)

```bash
# Qwen: freq + zipit + fixdom for GROUP_SIZES 16 24 32 45
bash experiment/run_all_qwen.sh
GROUP_SIZES="45" bash experiment/run_all_qwen.sh /scratch/mz81/hc_smoe/qwen

# Mixtral: freq + zipit + fixdom + prune for GROUP_SIZES 2 3 4 6
bash experiment/run_all_mixtral.sh
GROUP_SIZES="4" bash experiment/run_all_mixtral.sh /scratch/mz81/hc_smoe/mixtral
```

### Eval (saved model vs baseline)

```bash
# Eval a saved merged model (e.g. ZipIt or debug) — 0-shot Winogrande
bash scripts/run_eval_winogrande.sh /scratch/mz81/hc_smoe/qwen/zipit/groups_45_ing_act

# Eval baseline (unpruned Qwen MoE from HuggingFace)
python scripts/eval_baseline_winogrande.py --output /scratch/mz81/hc_smoe/qwen/baseline_winogrande.txt

# Eval any saved model (Qwen or Mixtral) — same script
bash scripts/run_eval_winogrande.sh /scratch/mz81/hc_smoe/mixtral/zipit/groups_4_ing_act
```

### Override output dir / env

```bash
# Custom output root (positional arg or OUTPUT_BASE)
bash experiment/qwen/run_zipit.sh /scratch/mz81/hc_smoe/qwen/zipit
OUTPUT_BASE=/other/root bash experiment/qwen/run_debug.sh

# Eval with custom result file
OUTPUT=/scratch/mz81/hc_smoe/qwen/zipit_wino.txt bash scripts/run_eval_winogrande.sh /scratch/mz81/hc_smoe/qwen/zipit/groups_45_ing_act
```

### Upload to Hugging Face

```bash
# Upload a merged model dir to HF (public repo by default)
python scripts/upload_to_hf.py /scratch/mz81/hc_smoe/qwen/zipit/groups_45_ing_act <your_username>/qwen15-moe-zipit-45g

# Private repo
python scripts/upload_to_hf.py /scratch/mz81/hc_smoe/qwen/zipit/groups_45_ing_act <your_username>/qwen15-moe-zipit-45g --private
```

Requires `huggingface_hub` and login: `huggingface-cli login`.

---

## 2. Detail — Merge scripts (experiment/qwen/)

### 2.1 run_zipit.sh

| Env / arg | Default | Description |
|-----------|---------|-------------|
| `OUTPUT_BASE` or `$1` | `/scratch/mz81/hc_smoe/qwen/zipit` | Root dir for saved models |
| `NUM_GROUPS` | `16 24 32 45` (all) | If set, only this many groups (e.g. `45`) |
| `MODEL_NAME` | `Qwen/Qwen1.5-MoE-A2.7B-Chat` | HF model id |
| `N_SENTENCES` | `32` | Calibration sentences (C4) per stat |
| `TRAIN_BS` | `2` | Batch size for calibration forward |
| `START_LAYER` | `0` | First layer to merge (0 = all) |
| `PARTITION` | `1` | Increase to 2/4 if OOM |
| `DATA_LIMIT` | `1000000` | Max tokens for ZipIt activation collection |
| `INGREDIENT` | `act` | ZipIt feature: `act`, `weight`, `act+weight` |
| `HF_HOME` | `/scratch/mz81/huggingface` | HuggingFace cache |
| `ACCEL_CONFIG` | `static/finetune_config.yaml` | Accelerate config |
| `MAIN_PORT` | `29521` | Port for distributed |

**Output:** `$OUTPUT_BASE/groups_<G>_ing_<INGREDIENT>/` (e.g. `groups_45_ing_act/`). Log: `run.log` in that dir.

**Python args passed to merging-qwen.py:**  
`--task=no --merge=zipit --dominant=no --similarity_base=expert-output --cluster=hierarchical --linkage=average --mode=normal --ingredient=$INGREDIENT` plus the above.

---

### 2.2 run_debug.sh

| Env / arg | Default | Description |
|-----------|---------|-------------|
| `OUTPUT_BASE` or `$1` | `/scratch/mz81/hc_smoe/qwen/debug` | Root dir for saved models |
| `NUM_GROUPS` | `45` | Number of random groups |
| `MODEL_NAME` | `Qwen/Qwen1.5-MoE-A2.7B-Chat` | HF model id |
| `PARTITION` | `1` | Partition for merge step |
| `HF_HOME` | `/scratch/mz81/huggingface` | HuggingFace cache |
| `ACCEL_CONFIG` | `static/finetune_config.yaml` | Accelerate config |
| `MAIN_PORT` | `29522` | Port (different from zipit to avoid clash) |

**Output:** `$OUTPUT_BASE/groups_<NUM_GROUPS>_uniform/` (e.g. `groups_45_uniform/`). No calibration data; random grouping + uniform average per group.

**Python args:** `--merge=uniform --dominant=random` (other merge-related args are effectively unused).

---

### 2.3 run_freq.sh

| Env / arg | Default | Description |
|-----------|---------|-------------|
| `OUTPUT_BASE` or `$1` | `/scratch/mz81/hc_smoe/qwen/freq` | Root dir for saved models |
| `NUM_GROUPS` | `16 24 32 45` (all) | If set, single group count |
| `MODEL_NAME` | `Qwen/Qwen1.5-MoE-A2.7B-Chat` | HF model id |
| `N_SENTENCES` | `32` | Calibration sentences for usage stats |
| `TRAIN_BS` | `2` | Batch size for calibration |
| `START_LAYER` | `0` | First layer to merge |
| `HF_HOME` | `/scratch/mz81/huggingface` | HuggingFace cache |
| `ACCEL_CONFIG`, `MAIN_PORT` | as above | Same as other scripts |

**Output:** `$OUTPUT_BASE/groups_<G>/` (no `_ing_*` suffix).

**Python args:** `--merge=freq --dominant=no`. Clustering for grouping; merge is usage-frequency-weighted average (no ZipIt).

---

### 2.4 run_fixdom.sh

| Env / arg | Default | Description |
|-----------|---------|-------------|
| `OUTPUT_BASE` or `$1` | `/scratch/mz81/hc_smoe/qwen/fixdom` | Root dir for saved models |
| `NUM_GROUPS` | `16 24 32 45` (all) | If set, single group count |
| `MODEL_NAME` | `Qwen/Qwen1.5-MoE-A2.7B-Chat` | HF model id |
| `N_SENTENCES` | `32` | Calibration sentences |
| `TRAIN_BS` | `2` | Batch size |
| `START_LAYER` | `0` | First layer to merge |
| `PARTITION` | `1` | Increase if OOM |
| `DATA_LIMIT` | `1000000` | Max tokens for activation collection |
| `INGREDIENT` | `act` | Feature for merge: `act`, `weight`, `act+weight` |
| `HF_HOME`, `ACCEL_CONFIG`, `MAIN_PORT` | as above | Same as other scripts |

**Output:** `$OUTPUT_BASE/groups_<G>/`.

**Python args:** `--merge=fix-dom-same`. Dominant expert per group; two-stage merge with ZipIt-style matching.

---

## 2.5 Detail — Merge scripts (experiment/mixtral/)

Mixtral has **8 experts** per layer. Default group sweep: `2 3 4 6`. Entrypoint: `hcsmoe/merging-mixtral.py`.

### run_zipit.sh

| Env / arg | Default | Description |
|-----------|---------|-------------|
| `OUTPUT_BASE` or `$1` | `/scratch/mz81/hc_smoe/mixtral/zipit` | Root dir for saved models |
| `NUM_GROUPS` | `2 3 4 6` (all) | If set, single group count |
| `MODEL_NAME` | `mistralai/Mixtral-8x7B-v0.1` | HF model id |
| `N_SENTENCES` | `32` | Calibration sentences |
| `TRAIN_BS` | `2` | Calibration batch size |
| `START_LAYER` | `0` | First layer to merge |
| `PARTITION` | `1` | Increase if OOM |
| `DATA_LIMIT` | `50000` | Max tokens for ZipIt (smaller than Qwen) |
| `INGREDIENT` | `act` | ZipIt feature: `act`, `weight`, `act+weight` |
| `HF_HOME` | `/scratch/mz81/huggingface` | HF cache |
| `ACCEL_CONFIG` | `static/finetune_config.yaml` | Accelerate config |
| `MAIN_PORT` | `29511` | Port (unique per script) |

**Output:** `$OUTPUT_BASE/groups_<G>_ing_<INGREDIENT>/` (e.g. `groups_4_ing_act/`).

---

### run_freq.sh

| Env / arg | Default | Description |
|-----------|---------|-------------|
| `OUTPUT_BASE` or `$1` | `/scratch/mz81/hc_smoe/mixtral/freq` | Root dir |
| `NUM_GROUPS` | `2 3 4 6` (all) | If set, single group count |
| `MODEL_NAME` | `mistralai/Mixtral-8x7B-v0.1` | HF model id |
| `N_SENTENCES`, `TRAIN_BS`, `START_LAYER` | `32`, `2`, `0` | Same as Qwen |
| `MAIN_PORT` | `29510` | Port |

**Output:** `$OUTPUT_BASE/groups_<G>/`.

---

### run_fixdom.sh

| Env / arg | Default | Description |
|-----------|---------|-------------|
| `OUTPUT_BASE` or `$1` | `/scratch/mz81/hc_smoe/mixtral/fixdom` | Root dir |
| `NUM_GROUPS` | `2 3 4 6` (all) | If set, single group count |
| `DATA_LIMIT` | `50000` | Max tokens for activation collection |
| `INGREDIENT` | `act` | Feature for merge |
| `MAIN_PORT` | `29513` | Port |

**Output:** `$OUTPUT_BASE/groups_<G>/`.

---

### run_prune.sh

| Env / arg | Default | Description |
|-----------|---------|-------------|
| `OUTPUT_BASE` or `$1` | `/scratch/mz81/hc_smoe/mixtral/prune` | Root dir |
| `NUM_GROUPS` | `2 3 4 6` (all) | If set, single group count |
| `PRUNE_MODE` | `normal` | `normal` = structural removal (gate + experts rebuilt); `zero-output` = soft prune (w2 zeroed, arch unchanged) |
| `MODEL_NAME` | `mistralai/Mixtral-8x7B-v0.1` | HF model id |
| `MAIN_PORT` | `29512` | Port |

**Output:** `$OUTPUT_BASE/groups_<G>_mode_<PRUNE_MODE>/` (e.g. `groups_4_mode_normal/`).

**Python args:** `--merge=prune --dominant=frequency --similarity_base=router-logits --mode=$PRUNE_MODE`.

---

## 2.6 run_all_qwen.sh

Runs **freq → zipit → fixdom** in sequence for each value in `GROUP_SIZES`. Does **not** run debug.

| Env / arg | Default | Description |
|-----------|---------|-------------|
| `$1` or `MODELS_ROOT` | `/scratch/mz81/hc_smoe/qwen` | Root for all variants |
| `GROUP_SIZES` | `16 24 32 45` | Space-separated group counts |
| `MODEL_NAME` | `Qwen/Qwen1.5-MoE-A2.7B-Chat` | HF model id |
| `N_SENTENCES`, `TRAIN_BS` | `32`, `2` | Passed to each sub-script |
| `ACCEL_CONFIG` | `static/finetune_config.yaml` | Accelerate config |

**Output:** `$MODELS_ROOT/freq/groups_<G>/`, `$MODELS_ROOT/zipit/groups_<G>_ing_act/`, `$MODELS_ROOT/fixdom/groups_<G>/`.

---

## 2.7 run_all_mixtral.sh

Runs **freq → zipit → fixdom → prune** in sequence for each value in `GROUP_SIZES`.

| Env / arg | Default | Description |
|-----------|---------|-------------|
| `$1` or `MODELS_ROOT` | `/scratch/mz81/hc_smoe/mixtral` | Root for all variants |
| `GROUP_SIZES` | `2 3 4 6` | Space-separated group counts |
| `MODEL_NAME` | `mistralai/Mixtral-8x7B-v0.1` | HF model id |
| `N_SENTENCES`, `TRAIN_BS` | `32`, `2` | Passed to each sub-script |
| `ACCEL_CONFIG` | `static/finetune_config.yaml` | Accelerate config |

**Output:** `$MODELS_ROOT/freq/groups_<G>/`, `$MODELS_ROOT/zipit/groups_<G>_ing_act/`, `$MODELS_ROOT/fixdom/groups_<G>/`, `$MODELS_ROOT/prune/groups_<G>_mode_normal/`.

---

## 3. Detail — Eval scripts

### 3.1 run_eval_winogrande.sh

Runs 0-shot Winogrande on a **saved model dir** (merged or otherwise).

| Arg / env | Required / default | Description |
|-----------|--------------------|-------------|
| `$1` (model_dir) | **required** | Path to saved HF-style model dir (e.g. zipit or debug output) |
| `TASK` | `winogrande` | lm_eval task name |
| `NUM_FEWSHOT` | `0` | Few-shot examples (0 = zero-shot) |
| `BATCH_SIZE` | `4` | Eval batch size |
| `OUTPUT` | (none) | If set, result file path; else `<model_dir>/eval_winogrande.txt` |
| `HF_HOME` | `/scratch/mz81/huggingface` | HF cache |
| `HF_DATASETS_CACHE` | `.../huggingface/datasets` | Datasets cache |

**Example:**

```bash
bash scripts/run_eval_winogrande.sh /scratch/mz81/hc_smoe/qwen/zipit/groups_45_ing_act
NUM_FEWSHOT=5 OUTPUT=/scratch/mz81/hc_smoe/qwen/zipit_wino5.txt bash scripts/run_eval_winogrande.sh /scratch/mz81/hc_smoe/qwen/zipit/groups_45_ing_act
```

---

### 3.2 eval_winogrande.py

Called by `run_eval_winogrande.sh`; can be used directly.

| Arg | Default | Description |
|-----|---------|-------------|
| `model_path` | (positional) | Saved model directory (HF format) |
| `--task` | `winogrande` | lm_eval task |
| `--num_fewshot` | `0` | Few-shot (0 = zero-shot) |
| `--batch_size` | `4` | Eval batch size |
| `--output` | `<model_path>/eval_winogrande.txt` | File to append results |

**Example:**

```bash
python scripts/eval_winogrande.py /scratch/mz81/hc_smoe/qwen/zipit/groups_45_ing_act --num_fewshot 0 --output /scratch/mz81/hc_smoe/qwen/zipit_wino.txt
```

---

### 3.3 eval_baseline_winogrande.py

Evaluates the **unpruned** Qwen MoE from HuggingFace (no merge).

| Arg | Default | Description |
|-----|---------|-------------|
| `--model_name` | `Qwen/Qwen1.5-MoE-A2.7B-Chat` | HF model id |
| `--task` | `winogrande` | lm_eval task |
| `--num_fewshot` | `0` | Few-shot |
| `--batch_size` | `4` | Eval batch size |
| `--output` | `/scratch/mz81/hc_smoe/qwen/baseline_winogrande.txt` | Result file |

**Example:**

```bash
python scripts/eval_baseline_winogrande.py --output /scratch/mz81/hc_smoe/qwen/baseline_winogrande.txt
```

---

### 3.4 upload_to_hf.py

Upload a local model directory (merge/prune output) to Hugging Face Hub. Creates a **public** model repo by default; use `--private` for a private repo.

| Arg | Required / default | Description |
|-----|--------------------|-------------|
| `local_dir` | **required** (positional) | Path to the model dir (must contain `config.json`, HF-style checkpoint) |
| `repo_id` | **required** (positional) | Hugging Face repo id, e.g. `username/model-name` |
| `--private` | off (public) | If set, create/use repo as private |
| `--repo-type` | `model` | `model`, `dataset`, or `space` |

**Requirements:** `pip install huggingface_hub`, and logged in (`huggingface-cli login` or `HF_TOKEN`).

**Examples:**

```bash
# Public repo (default)
python scripts/upload_to_hf.py /scratch/mz81/hc_smoe/qwen/zipit/groups_45_ing_act mz81/qwen15-moe-zipit-45g

# Private repo
python scripts/upload_to_hf.py /scratch/mz81/hc_smoe/qwen/zipit/groups_45_ing_act mz81/qwen15-moe-zipit-45g --private
```

After upload, the script prints the model page URL (e.g. `https://huggingface.co/mz81/qwen15-moe-zipit-45g`). Others can load with `AutoModelForCausalLM.from_pretrained("username/repo-id", trust_remote_code=True)`.

---

## 4. merging-qwen.py — Main Python args (reference)

Used by all Qwen merge scripts; can be called via `accelerate launch ... hcsmoe/merging-qwen.py` with:

| Arg | Typical / default | Description |
|-----|-------------------|-------------|
| `--task` | `no` | In-script eval: `no` to skip; or e.g. `winogrande` |
| `--model_name` | `Qwen/Qwen1.5-MoE-A2.7B-Chat` | HF model id |
| `--output_path` | (set by script) | Where to save merged model |
| `--result_path` | (set by script) | Where to write eval results if `task != no` |
| `--num_average_groups` | e.g. `45` | Target number of expert groups per layer |
| `--merge` | `zipit` / `uniform` / `freq` / `fix-dom-same` | Merge method |
| `--dominant` | `no` / `random` | Expert selection: `no` (clustering), `random` (debug) |
| `--similarity_base` | `expert-output` | Base for clustering (e.g. expert outputs) |
| `--cluster` | `hierarchical` | Clustering algorithm |
| `--linkage` | `average` | Linkage for hierarchical |
| `--mode` | `normal` | Merge mode (e.g. normal, input-weight) |
| `--ingredient` | `act` | For zipit/fixdom: `act`, `weight`, `act+weight` |
| `--n_sentences` | `32` | Calibration sentences |
| `--train_batch_size` | `2` | Calibration batch size |
| `--start_layer` | `0` | First layer index to merge |
| `--partition` | `1` | Partition for low-VRAM |
| `--data_limit` | `1000000` (zipit/fixdom) | Max tokens for activation collection |

---

## 5. Output layout (defaults)

```
/scratch/mz81/hc_smoe/
├── qwen/
│   ├── zipit/
│   │   └── groups_45_ing_act/         # ZipIt 45 groups, ingredient=act
│   ├── debug/
│   │   └── groups_45_uniform/         # Debug: random + uniform
│   ├── freq/
│   │   └── groups_45/                  # Freq merge 45 groups
│   ├── fixdom/
│   │   └── groups_45/                  # Fix-dom-same 45 groups
│   ├── baseline_winogrande.txt         # From eval_baseline_winogrande.py
│   └── (optional custom OUTPUT paths for eval)
│
└── mixtral/
    ├── zipit/
    │   └── groups_4_ing_act/           # ZipIt 4 groups (default sweep: 2 3 4 6)
    ├── freq/
    │   └── groups_4/
    ├── fixdom/
    │   └── groups_4/
    └── prune/
        └── groups_4_mode_normal/       # or mode_zero-output

/scratch/mz81/huggingface/              # HF cache (models, datasets)
```

Each model dir contains HuggingFace checkpoint files and `run.log`. Eval appends to `eval_winogrande.txt` inside the model dir unless `OUTPUT` (or `--output`) is set.

---

## 6. Accelerate / multi-GPU

- Default `static/finetune_config.yaml` uses `num_processes: 1` (single process, `device_map="auto"` over all visible GPUs).
- For **8-GPU** (e.g. 8× H200): use `static/finetune_config_8gpu.yaml` (already has `num_processes: 8`), then:
  ```bash
  ACCEL_CONFIG=static/finetune_config_8gpu.yaml NUM_GROUPS=45 bash experiment/qwen/run_zipit.sh
  ```
- Slurm example (8 cards, 24h):
  ```bash
  salloc -p commons --gres=gpu:h200:8 --time=24:00:00 --cpus-per-task=64 --mem=256G
  srun --jobid=<JOBID> --pty bash
  # then in that shell: conda activate hc_smoe; export LD_PRELOAD=...; run merge/eval
  ```
