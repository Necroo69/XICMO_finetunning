"""Push a trained adapter / merged model to the Hugging Face Hub.

Usage:
    python shared/upload_to_hf.py \
        --path experiments/seo/adapter \
        --repo xicmo/seo-8b \
        [--private] [--commit "seo v1"]

Requires HF_TOKEN in the environment (or a prior `huggingface-cli login`).
"""

from __future__ import annotations

import argparse
import os

from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Upload weights/adapters to the HF Hub.")
    p.add_argument("--path", required=True, help="Local folder to upload.")
    p.add_argument("--repo", required=True, help="Target repo id, e.g. xicmo/seo-8b.")
    p.add_argument("--private", action="store_true", help="Create the repo as private.")
    p.add_argument("--commit", default="Upload from Xicmo pipeline", help="Commit message.")
    return p.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN not set. Add it to .env or your environment.")

    if not os.path.isdir(args.path):
        raise SystemExit(f"Not a directory: {args.path}")

    create_repo(args.repo, token=token, private=args.private, exist_ok=True)

    api = HfApi()
    api.upload_folder(
        folder_path=args.path,
        repo_id=args.repo,
        commit_message=args.commit,
        token=token,
    )
    print(f"Uploaded {args.path} -> https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
