export NCCL_P2P_DISABLE=0
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export TOKENIZERS_PARALLELISM="false"
export HF_HOME="your-huggingface-home-path"

accelerate launch --config_file static/finetune_config.yaml \
  --main_process_port 29512 hcsmoe/merging-mixtral.py \
  --task="rte" \
  --model_name="s3nh/TinyLLama-4x1.1B-MoE" \
  --dominant="no" \
  --similarity_base="expert-output" \
  --cluster="hirarchical" \
  --linkage="average" \
  --merge="freq" \
  --num_average_groups=2 \
  --n_sentences=32 \
  --train_batch_size=2 \
  --eval_batch_size=16 \
  --start_layer=0 \
  --result_path="results/result_tinyllama_test.txt" \
  --output_path="results/" |& tee results/log_tinyllama_test