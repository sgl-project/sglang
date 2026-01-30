#!/usr/bin/env python3
"""
Pre-download critical models before CI tests run.

This script runs during CI initialization to download models that are commonly
used in multi-GPU tests. By downloading them during the install step (single
process), we avoid race conditions that occur when multiple TP workers try
to download the same model simultaneously during tests.

NOTE: This script ONLY runs in CI environments (SGLANG_IS_IN_CI=true).
"""

import os
import sys
import time
from pathlib import Path


def is_in_ci() -> bool:
    """Check if we're running in CI environment."""
    return os.environ.get("SGLANG_IS_IN_CI", "").lower() == "true"


# Exit immediately if not in CI - this script should never run locally
if not is_in_ci():
    print(
        "Not in CI environment (SGLANG_IS_IN_CI != true), skipping model pre-download"
    )
    sys.exit(0)


# Models that need to be pre-downloaded for multi-GPU tests
# These are models where concurrent download race conditions have been observed
CI_PREDOWNLOAD_MODELS = [
    # MoE models used in 4-GPU tests with TP parallelism
    "Qwen/Qwen3-30B-A3B",  # Used in test_enable_thinking.py (4-GPU)
]

# Maximum time to spend on pre-download (seconds)
MAX_PREDOWNLOAD_TIME = 600  # 10 minutes


def get_hf_cache_path() -> Path:
    """Get HuggingFace cache directory path."""
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    return Path(hf_home) / "hub"


def model_id_to_cache_dir(model_id: str) -> str:
    """Convert model ID to HF cache directory name."""
    return "models--" + model_id.replace("/", "--")


def is_model_cached(model_id: str) -> bool:
    """Check if a model is already cached (has snapshots directory with content)."""
    cache_dir = get_hf_cache_path()
    model_cache_dir = cache_dir / model_id_to_cache_dir(model_id)
    snapshots_dir = model_cache_dir / "snapshots"

    if not snapshots_dir.exists():
        return False

    # Check if there's at least one snapshot
    snapshots = list(snapshots_dir.iterdir())
    return len(snapshots) > 0


def predownload_model(model_id: str) -> bool:
    """
    Pre-download a model using huggingface_hub.

    Returns True if successful, False otherwise.
    """
    try:
        from huggingface_hub import snapshot_download

        print(f"  Downloading {model_id}...")
        start = time.time()

        # Download the model snapshot
        # This will be cached in HF_HOME/hub/models--org--model/snapshots/
        snapshot_download(
            repo_id=model_id,
            # Don't download all files, just the essential ones for model loading
            # This avoids downloading README, etc.
            allow_patterns=["*.json", "*.safetensors", "*.model", "*.tiktoken", "*.txt"],
            # Ignore large files that might not be needed
            ignore_patterns=["*.bin", "*.msgpack", "*.h5", "*.ot", "consolidated*"],
        )

        elapsed = time.time() - start
        print(f"  Downloaded {model_id} in {elapsed:.1f}s")
        return True

    except Exception as e:
        print(f"  Warning: Failed to download {model_id}: {e}")
        return False


def main():
    print("=" * 70)
    print("CI Model Pre-Download")
    print("=" * 70)
    print()

    start_time = time.time()
    cache_dir = get_hf_cache_path()

    print(f"HF cache: {cache_dir}")
    print(f"Models to check: {len(CI_PREDOWNLOAD_MODELS)}")
    print()

    downloaded = 0
    skipped = 0
    failed = 0

    for model_id in CI_PREDOWNLOAD_MODELS:
        # Check time limit
        elapsed = time.time() - start_time
        if elapsed > MAX_PREDOWNLOAD_TIME:
            print(f"\nTime limit reached ({elapsed:.1f}s), stopping pre-download")
            break

        print(f"[{model_id}]")

        # Check if already cached
        if is_model_cached(model_id):
            print("  Already cached, skipping")
            skipped += 1
            continue

        # Download
        if predownload_model(model_id):
            downloaded += 1
        else:
            failed += 1

        print()

    elapsed = time.time() - start_time

    print("=" * 70)
    print(f"Pre-download complete ({elapsed:.1f}s)")
    print(f"  Downloaded: {downloaded}")
    print(f"  Skipped (cached): {skipped}")
    print(f"  Failed: {failed}")
    print("=" * 70)


if __name__ == "__main__":
    main()
