#!/usr/bin/env python3
"""Load a saved (merged) model and run LM eval on Winogrande for quick inference check.

Usage:
  python scripts/eval_winogrande.py /path/to/model
  python scripts/eval_winogrande.py morriszjm/Mixtral-8x7B-v0.1-sdpa   # HuggingFace model ID
  python scripts/eval_winogrande.py <model_path_or_hf_id> [--task winogrande] [--num_fewshot 5]

Note: Use default task winogrande; hellaswag can hit datasets/lm_eval cache compat issues.
"""
import argparse
import os
import sys

# Use scratch for datasets cache to avoid NFS + old-cache compat issues (e.g. hellaswag)
if not os.environ.get("HF_DATASETS_CACHE") and os.path.isdir("/scratch/mz81"):
    os.environ["HF_DATASETS_CACHE"] = "/scratch/mz81/huggingface/datasets"

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# repo root
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from hcsmoe.evaluation import evaluate_fewshot


def main():
    parser = argparse.ArgumentParser(description="Eval saved model on Winogrande (or other lm_eval task)")
    parser.add_argument("model_path", type=str, help="Path to saved model or HuggingFace model ID (e.g. user/repo)")
    parser.add_argument("--task", type=str, default="winogrande", help="lm_eval task (e.g. winogrande, hellaswag)")
    parser.add_argument("--num_fewshot", type=int, default=0, help="0 for zero-shot (default), 5 for 5-shot")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--output", type=str, default=None, help="Optional file to append results")
    args = parser.parse_args()

    model_path_or_id = args.model_path
    is_local = os.path.isdir(model_path_or_id)
    if is_local:
        model_path_or_id = os.path.abspath(model_path_or_id)
        # So that trust_remote_code can load local configuration_mixtral.py / modeling from saved dir
        if model_path_or_id not in sys.path:
            sys.path.insert(0, model_path_or_id)

    print(f"Loading model and tokenizer from {model_path_or_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path_or_id, trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    # Use SDPA so we don't need flash_attn installed (saved config may have flash_attention_2)
    config = AutoConfig.from_pretrained(model_path_or_id, trust_remote_code=True)
    if getattr(config, "_attn_implementation", None) == "flash_attention_2":
        config._attn_implementation = "sdpa"
    model = AutoModelForCausalLM.from_pretrained(
        model_path_or_id,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    output_path = args.output or (os.path.join(model_path_or_id, "eval_winogrande.txt") if is_local else "eval_winogrande.txt")
    print(f"Running {args.task} (num_fewshot={args.num_fewshot}) ...")
    evaluate_fewshot(
        model,
        tokenizer=tokenizer,
        task=args.task,
        num_fewshot=args.num_fewshot,
        eval_batch_size=args.batch_size,
        log=True,
        output_path=output_path,
    )
    print(f"Results appended to {output_path}")


if __name__ == "__main__":
    main()
