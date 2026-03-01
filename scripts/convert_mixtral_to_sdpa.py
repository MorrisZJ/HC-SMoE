#!/usr/bin/env python3
"""
Download Mixtral-8x7B-v0.1 and convert to a SDPA-only (no flash-attn) variant that can be
loaded with trust_remote_code=True without installing flash_attn.

Requires: HuggingFace login if the model is gated (run `huggingface-cli login`).
For --push_to_hub you must be logged in.

Usage:
  # Download to /mnt/scratch/model_zoo and create SDPA version (default)
  python scripts/convert_mixtral_to_sdpa.py

  # Custom paths
  python scripts/convert_mixtral_to_sdpa.py --output_dir /mnt/scratch/model_zoo/Mixtral-8x7B-v0.1-sdpa

  # Convert existing local clone (no re-download)
  python scripts/convert_mixtral_to_sdpa.py --from_local /path/to/Mixtral-8x7B-v0.1 --output_dir /mnt/scratch/model_zoo/Mixtral-8x7B-v0.1-sdpa

  # Also push to your HuggingFace repo (morriszjm/Mixtral-8x7B-v0.1-sdpa)
  python scripts/convert_mixtral_to_sdpa.py --push_to_hub --hf_username morriszjm --hf_repo Mixtral-8x7B-v0.1-sdpa
"""
from __future__ import annotations

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Register hcsmoe_mixtral and get config class
import hcsmoe.models.mixtral  # noqa: F401
from hcsmoe.models.mixtral import MixtralConfig as HCSMoEMixtralConfig

HCSMOE_MIXTRAL_SRC_DIR = os.path.dirname(os.path.abspath(hcsmoe.models.mixtral.__file__))


def main(
    model_id: str = "mistralai/Mixtral-8x7B-v0.1",
    output_dir: str = "/mnt/scratch/model_zoo/Mixtral-8x7B-v0.1-sdpa",
    from_local: str | None = None,
    push_to_hub: bool = False,
    hf_username: str = "morriszjm",
    hf_repo: str | None = None,
) -> None:
    from huggingface_hub import snapshot_download, HfApi
    from transformers import AutoConfig

    if from_local is not None:
        # Convert in place or copy from existing download (no re-download)
        import shutil
        if not os.path.isdir(from_local):
            raise SystemExit(f"Not a directory: {from_local}")
        src_abs = os.path.abspath(from_local)
        out_abs = os.path.abspath(output_dir)
        if src_abs != out_abs:
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            shutil.copytree(from_local, output_dir)
        work_dir = output_dir
    else:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Downloading {model_id} to {output_dir} ...")
        snapshot_download(
            model_id,
            local_dir=output_dir,
            local_dir_use_symlinks=False,
        )
        work_dir = output_dir

    print("Building SDPA config (hcsmoe_mixtral, no flash_attn required) ...")
    orig_config = AutoConfig.from_pretrained(work_dir, trust_remote_code=True)
    d = orig_config.to_dict()
    # Remove keys that might conflict; we set model_type and add auto_map
    for k in ("transformers_version", "model_type", "auto_map"):
        d.pop(k, None)
    # Force SDPA so loading does not require flash_attn
    d["_attn_implementation"] = "sdpa"

    new_config = HCSMoEMixtralConfig(**d)
    new_config.auto_map = {
        "AutoConfig": "configuration_mixtral.MixtralConfig",
        "AutoModelForCausalLM": "modeling_mixtral.MixtralForCausalLM",
    }
    new_config.save_pretrained(work_dir)

    print("Copying HC-SMoE modeling and config (no top-level flash_attn import) ...")
    for fname in ("modeling_mixtral.py", "configuration_mixtral.py"):
        src = os.path.join(HCSMOE_MIXTRAL_SRC_DIR, fname)
        dst = os.path.join(work_dir, fname)
        with open(src, "r", encoding="utf-8") as f:
            content = f.read()
        with open(dst, "w", encoding="utf-8") as f:
            f.write(content)

    print(f"Done. Load with: AutoModelForCausalLM.from_pretrained({work_dir!r}, trust_remote_code=True)")

    if push_to_hub:
        repo_id = f"{hf_username}/{hf_repo or os.path.basename(work_dir.rstrip('/'))}"
        print(f"Pushing to HuggingFace: {repo_id} ...")
        try:
            api = HfApi()
            api.create_repo(repo_id, private=False, exist_ok=True)
            api.upload_folder(
                folder_path=work_dir,
                repo_id=repo_id,
                repo_type="model",
            )
            print(f"Pushed to https://huggingface.co/{repo_id}")
        except Exception as e:
            err_str = str(e)
            if "401" in err_str or "Unauthorized" in err_str or "Invalid username or password" in err_str:
                print("Push failed: not logged in or token missing write permission.")
                print("  Run: huggingface-cli login")
                print("  Create a token with 'write' at: https://huggingface.co/settings/tokens")
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Mixtral to SDPA-only (no flash_attn) and optionally push to HF")
    parser.add_argument("--model_id", default="mistralai/Mixtral-8x7B-v0.1", help="HuggingFace model id")
    parser.add_argument("--output_dir", default="/mnt/scratch/model_zoo/Mixtral-8x7B-v0.1-sdpa", help="Local output directory")
    parser.add_argument("--from_local", default=None, help="Skip download; convert existing dir (path to model dir)")
    parser.add_argument("--push_to_hub", action="store_true", help="Push result to HuggingFace")
    parser.add_argument("--hf_username", default="morriszjm", help="HuggingFace username/org")
    parser.add_argument("--hf_repo", default=None, help="Repo name (default: basename of output_dir)")
    args = parser.parse_args()
    main(
        model_id=args.model_id,
        output_dir=args.output_dir,
        from_local=args.from_local,
        push_to_hub=args.push_to_hub,
        hf_username=args.hf_username,
        hf_repo=args.hf_repo,
    )
