#!/usr/bin/env python3
"""Upload an ONNX-exported model folder to the Hugging Face Hub.

This is for cases where you exported your model to ONNX (e.g. with Optimum) and
want to publish the ONNX artifacts (and optionally tokenizer/config) to a Hub
repo.

This script uploads the *files* in a directory (unlike `push_to_hub()` which
expects a standard Transformers model layout).

Prerequisites:
  pip install -U "huggingface_hub>=0.20"

Auth:
  This script uses `from huggingface_hub import login; login()`.

Usage:
  python upload_onnx_to_hf.py

Notes:
  - ONNX_DIR should point to the directory containing your .onnx files.
  - If you also have tokenizer/config files in the same directory, they will be
    uploaded too.
"""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import HfApi, create_repo, login


# TODO: set this to your ONNX export directory
ONNX_DIR = "path/to/my/model"

# Target repo for ONNX artifacts
REPO_NAME = "Andhs/fgemma-pin-language-ONNX"

# Optional: set to True if you want the repo to be private
PRIVATE = False

# Optional: commit message
COMMIT_MESSAGE = "Upload ONNX model"


def main() -> None:
    onnx_dir = Path(ONNX_DIR)
    if not onnx_dir.exists():
        raise FileNotFoundError(f"ONNX_DIR does not exist: {onnx_dir}")
    if not onnx_dir.is_dir():
        raise NotADirectoryError(f"ONNX_DIR is not a directory: {onnx_dir}")

    # Quick sanity check: at least one .onnx file
    onnx_files = list(onnx_dir.glob("*.onnx"))
    if not onnx_files:
        raise FileNotFoundError(
            f"No .onnx files found in {onnx_dir}. "
            "Point ONNX_DIR to the folder containing the exported ONNX files."
        )

    print("=" * 80)
    print("🤗 Uploading ONNX artifacts to Hugging Face Hub")
    print("=" * 80)
    print(f"Local ONNX dir: {onnx_dir}")
    print(f"Target repo:    {REPO_NAME}")
    print(f"Private repo:   {PRIVATE}")
    print(f"ONNX files:     {len(onnx_files)}")
    print()

    # Auth (interactive)
    login()

    # Ensure repo exists
    create_repo(
        repo_id=REPO_NAME,
        private=PRIVATE,
        exist_ok=True,
    )

    # Upload entire folder contents
    api = HfApi()
    api.upload_folder(
        repo_id=REPO_NAME,
        folder_path=str(onnx_dir),
        path_in_repo=".",
        commit_message=COMMIT_MESSAGE,
    )

    info = api.repo_info(REPO_NAME)
    print()
    print("✅ Upload complete")
    print(f"Repo URL: {info.url}")


if __name__ == "__main__":
    main()
