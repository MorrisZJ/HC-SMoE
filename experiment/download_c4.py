#!/usr/bin/env python3
"""
Download C4 (en) calibration data and save to hcsmoe/data/c4-train.00000-of-01024.json
so that merging scripts use local data instead of pulling from HuggingFace at runtime.

Usage:
  python experiment/download_c4.py [--num_examples 2048] [--repo_root /path/to/HC-SMoE]

Default: 2048 examples (enough for n_sentences=128). Use a larger number for full calibration.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Download C4 calibration data for HC-SMoE")
    parser.add_argument(
        "--num_examples",
        type=int,
        default=2048,
        help="Number of C4 examples to download (default: 2048, enough for n_sentences=128)",
    )
    parser.add_argument(
        "--repo_root",
        type=Path,
        default=None,
        help="Path to HC-SMoE repo root (default: parent of experiment/)",
    )
    args = parser.parse_args()

    repo_root = args.repo_root or Path(__file__).resolve().parent.parent
    data_dir = repo_root / "hcsmoe" / "data"
    out_file = data_dir / "c4-train.00000-of-01024.json"
    data_dir.mkdir(parents=True, exist_ok=True)

    if out_file.exists():
        print(f"File already exists: {out_file}", file=sys.stderr)
        print("Delete it or use a different path to re-download.", file=sys.stderr)
        sys.exit(0)

    print(f"Downloading {args.num_examples} C4 (en) examples via streaming...")
    from datasets import load_dataset

    # Streaming fetches only the data needed for the first N examples (avoids full shard download)
    stream = load_dataset("allenai/c4", "en", split="train", streaming=True, trust_remote_code=True)
    rows = []
    for i, ex in enumerate(stream):
        if i >= args.num_examples:
            break
        rows.append(ex)
        if (i + 1) % 512 == 0:
            print(f"  {i + 1}/{args.num_examples}")

    from datasets import Dataset
    ds = Dataset.from_list(rows)
    ds.to_json(out_file)
    print(f"Saved {len(ds)} examples to {out_file}")

if __name__ == "__main__":
    main()
