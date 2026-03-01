export NCCL_P2P_DISABLE=0
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export TOKENIZERS_PARALLELISM="false"
export HF_HOME="${HF_HOME:-/scratch/mz81/huggingface}"
mkdir -p "$HF_HOME"

OUTPUT_BASE="${1:-results}"
mkdir -p "$OUTPUT_BASE"

accelerate launch --config_file static/finetune_config.yaml \
  --main_process_port 29512 hcsmoe/merging-mixtral.py \
  --task="winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu,openbookqa,rte" \
  --model_name="mistralai/Mixtral-8x7B-v0.1" \
  --dominant="no" \
  --similarity_base="expert-output" \
  --cluster="hirarchical" \
  --linkage="average" \
  --merge="freq" \
  --num_average_groups=4 \
  --n_sentences=32 \
  --train_batch_size=2 \
  --eval_batch_size=16 \
  --start_layer=0 \
  --result_path="${OUTPUT_BASE}/result_mixtral_test.txt" \
  --output_path="${OUTPUT_BASE}" |& tee "${OUTPUT_BASE}/log_mixtral_test"