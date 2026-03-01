#!/usr/bin/env python3
"""Load a saved (merged) model and run LM eval on Winogrande for quick inference check.

Usage:
  python scripts/eval_winogrande.py /scratch/mz81/hc_smoe/qwen/debug/groups_45_uniform
  python scripts/eval_winogrande.py <model_path> [--task winogrande] [--num_fewshot 5]

Note: Use default task winogrande; hellaswag can hit datasets/lm_eval cache compat issues.
"""
import argparse
import os
import sys

# Use scratch for datasets cache to avoid NFS + old-cache compat issues (e.g. hellaswag)
if not os.environ.get("HF_DATASETS_CACHE") and os.path.isdir("/scratch/mz81"):
    os.environ["HF_DATASETS_CACHE"] = "/scratch/mz81/huggingface/datasets"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# repo root
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from hcsmoe.evaluation import evaluate_fewshot


def main():
    parser = argparse.ArgumentParser(description="Eval saved model on Winogrande (or other lm_eval task)")
    parser.add_argument("model_path", type=str, help="Path to saved model (HF format)")
    parser.add_argument("--task", type=str, default="winogrande", help="lm_eval task (e.g. winogrande, hellaswag)")
    parser.add_argument("--num_fewshot", type=int, default=0, help="0 for zero-shot (default), 5 for 5-shot")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--output", type=str, default=None, help="Optional file to append results")
    args = parser.parse_args()

    model_path = os.path.abspath(args.model_path)
    if not os.path.isdir(model_path):
        raise SystemExit(f"Model path not found: {model_path}")

    print(f"Loading model and tokenizer from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    output_path = args.output or os.path.join(model_path, "eval_winogrande.txt")
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
