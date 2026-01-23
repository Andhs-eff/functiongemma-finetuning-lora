#!/usr/bin/env python3
"""Upload a locally saved Transformers model to the Hugging Face Hub.

This script is intended for models saved via `save_pretrained()` (e.g. the
merged model saved in `train_func.py` under `.../full_model`).

Prerequisites:
  1) Install deps:
       pip install -U "huggingface_hub>=0.20" "transformers>=4.38" "safetensors" "torch"

  2) Login (interactive):
       python upload_model_to_hf.py

Usage:
  # Windows (cmd.exe)
  set MODEL_PATH=path\\to\\my\\model
  python upload_model_to_hf.py

  # PowerShell
  $env:MODEL_PATH="path\\to\\my\\model"
  python upload_model_to_hf.py

  # Linux/macOS (bash/zsh)
  export MODEL_PATH="path/to/my/model"
  python3 upload_model_to_hf.py

Notes:
  - MODEL_PATH is read from the environment.
  - REPO_NAME must be in the form "namespace/repo".
  - This script uses `from huggingface_hub import login; login()` for auth.
"""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo, login
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM


# Read from environment so you can run:
#   set MODEL_PATH=... && python upload_model_to_hf.py
MODEL_PATH = os.environ.get("MODEL_PATH", "")

# Target repo on Hugging Face Hub
REPO_NAME = "Andhs/fgemma-pin-language"

# Optional: set to True if you want the repo to be private
PRIVATE = True

# If you prefer non-interactive auth, you can set HF_TOKEN in your environment.
# Otherwise, `login()` below will prompt you.
HF_TOKEN = os.environ.get("HF_TOKEN")


def main() -> None:
    if not MODEL_PATH:
        raise ValueError(
            "MODEL_PATH is not set. Provide it via environment variable, e.g.\n"
            "  set MODEL_PATH=path\\to\\my\\model\n"
            "  python upload_model_to_hf.py"
        )

    model_dir = Path(MODEL_PATH)
    if not model_dir.exists():
        raise FileNotFoundError(
            f"MODEL_PATH does not exist: {model_dir}. "
            "Set MODEL_PATH to the directory created by save_pretrained()."
        )

    # Basic sanity checks
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Could not find {config_path}. "
            "MODEL_PATH must point to a Transformers model directory."
        )

    print("=" * 80)
    print("🤗 Uploading model to Hugging Face Hub")
    print("=" * 80)
    print(f"Local model dir: {model_dir}")
    print(f"Target repo:     {REPO_NAME}")
    print(f"Private repo:    {PRIVATE}")
    print()

    # Auth (interactive by default)
    # If you are already logged in, this is a no-op.
    # If you want to avoid prompts, set HF_TOKEN and pass it to login(token=...).
    if HF_TOKEN:
        login(token=HF_TOKEN)
    else:
        login()

    # Ensure repo exists (idempotent)
    create_repo(
        repo_id=REPO_NAME,
        private=PRIVATE,
        exist_ok=True,
    )

    # Load to validate the directory is complete and to ensure tokenizer is present.
    # (This also helps catch missing files before upload.)
    print("Loading config/tokenizer/model for validation...")
    _ = AutoConfig.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)

    # Push to hub
    print("Pushing model...")
    model.push_to_hub(REPO_NAME)

    print("Pushing tokenizer...")
    tokenizer.push_to_hub(REPO_NAME)

    # Optional: show repo URL
    api = HfApi()
    info = api.repo_info(REPO_NAME)
    print()
    print("✅ Upload complete")
    print(f"Repo URL: https://huggingface.co/{REPO_NAME}")
    print(f"Repo id: {info.id}")
    print(f"Private: {info.private}")
    print(f"SHA: {info.sha}")

if __name__ == "__main__":
    main()
