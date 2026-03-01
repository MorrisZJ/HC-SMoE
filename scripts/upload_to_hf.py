#!/usr/bin/env python3
"""
Upload a local model directory (e.g. merged/pruned HC-SMoE output) to Hugging Face Hub.
Creates a public model repo by default.

Usage:
  python scripts/upload_to_hf.py <local_dir> <repo_id> [--private]
  python scripts/upload_to_hf.py /scratch/mz81/hc_smoe/qwen/zipit/groups_45_ing_act username/qwen15-moe-zipit-45g

Requires: pip install huggingface_hub; must be logged in (huggingface-cli login).
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Upload a local model dir to Hugging Face Hub (public by default)."
    )
    parser.add_argument(
        "local_dir",
        type=str,
        help="Path to the model directory (HF-style: config.json, pytorch_model*.bin, tokenizer files, etc.)",
    )
    parser.add_argument(
        "repo_id",
        type=str,
        help="Hugging Face repo id, e.g. username/model-name",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create/use repo as private (default: public)",
    )
    parser.add_argument(
        "--repo-type",
        type=str,
        default="model",
        choices=("model", "dataset", "space"),
        help="Repo type (default: model)",
    )
    args = parser.parse_args()

    local_dir = os.path.abspath(args.local_dir)
    if not os.path.isdir(local_dir):
        print(f"Error: not a directory: {local_dir}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(os.path.join(local_dir, "config.json")):
        print(f"Error: no config.json in {local_dir} (not a HF model dir?)", file=sys.stderr)
        sys.exit(1)

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("Error: install with  pip install huggingface_hub", file=sys.stderr)
        sys.exit(1)

    api = HfApi()
    private = args.private
    repo_type = args.repo_type

    print(f"Creating repo {args.repo_id} (private={private})...")
    api.create_repo(
        repo_id=args.repo_id,
        repo_type=repo_type,
        private=private,
        exist_ok=True,
    )
    print(f"Uploading {local_dir} -> {args.repo_id} ...")
    api.upload_folder(
        folder_path=local_dir,
        repo_id=args.repo_id,
        repo_type=repo_type,
    )
    print(f"Done. https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
