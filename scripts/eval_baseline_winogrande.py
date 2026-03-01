#!/usr/bin/env python3
"""Eval the unpruned / baseline Qwen MoE on Winogrande (0-shot by default).

Use this to get baseline numbers to compare with merged models.

Usage:
  python scripts/eval_baseline_winogrande.py
  python scripts/eval_baseline_winogrande.py --model_name Qwen/Qwen1.5-MoE-A2.7B-Chat --output /scratch/mz81/hc_smoe/qwen/baseline_winogrande.txt
"""
import argparse
import os
import sys

# Use scratch for datasets cache
if not os.environ.get("HF_DATASETS_CACHE") and os.path.isdir("/scratch/mz81"):
    os.environ["HF_DATASETS_CACHE"] = "/scratch/mz81/huggingface/datasets"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from hcsmoe.evaluation import evaluate_fewshot


def main():
    parser = argparse.ArgumentParser(description="Eval unpruned (baseline) Qwen MoE on Winogrande")
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen1.5-MoE-A2.7B-Chat",
        help="HuggingFace model id (unpruned baseline)",
    )
    parser.add_argument("--task", type=str, default="winogrande")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--output",
        type=str,
        default="/scratch/mz81/hc_smoe/qwen/baseline_winogrande.txt",
        help="Where to append results",
    )
    args = parser.parse_args()

    print(f"Loading baseline (unpruned) model: {args.model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    print(f"Running {args.task} (num_fewshot={args.num_fewshot}) ...")
    evaluate_fewshot(
        model,
        tokenizer=tokenizer,
        task=args.task,
        num_fewshot=args.num_fewshot,
        eval_batch_size=args.batch_size,
        log=True,
        output_path=args.output,
    )
    print(f"Baseline results appended to {args.output}")


if __name__ == "__main__":
    main()
