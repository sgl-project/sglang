#!/usr/bin/env python3
"""
Validate model weight integrity using SHA256 checksums.

This script runs during CI initialization to detect corrupted model weights
by comparing local file checksums against HuggingFace Hub metadata.
If corruption is detected, the corrupted files are removed to trigger
a fresh re-download.

This reuses the existing model_file_verifier.py logic for checksum validation.

NOTE: This script ONLY runs in CI environments (SGLANG_IS_IN_CI=true).
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Optional


def is_in_ci() -> bool:
    """Check if we're running in CI environment."""
    return os.environ.get("SGLANG_IS_IN_CI", "").lower() == "true"


# Exit immediately if not in CI - this script should never run locally
if not is_in_ci():
    print(
        "Not in CI environment (SGLANG_IS_IN_CI != true), skipping checksum validation"
    )
    sys.exit(0)


# Add python directory to path to import sglang modules
REPO_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

from sglang.srt.utils.model_file_verifier import (
    IntegrityError,
    _compute_manifest_from_folder,
    _load_file_infos_from_hf,
)

# Models commonly used in CI tests that should be validated with checksums
# These are models where 0.0 accuracy failures have been observed
CI_CRITICAL_MODELS = [
    # GLM models - observed 0.0 accuracy failures
    "THUDM/glm-4-9b-chat",
    "THUDM/glm-4-9b-chat-1m",
    "zai-org/GLM-4.5-Air-FP8",  # Used in test_glm4_moe_models.py
    # Llama models - commonly tested
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    # Qwen models
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen3-30B-A3B",  # Used in test_enable_thinking.py
    # Mistral/Mixtral
    "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",  # Used in benchmark tests
]

# Maximum time to spend on validation (seconds)
MAX_VALIDATION_TIME = 300

# Maximum number of files to validate per model (to avoid timeout on large models)
MAX_FILES_PER_MODEL = 20


def get_hf_cache_path() -> Path:
    """Get HuggingFace cache directory path."""
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    return Path(hf_home) / "hub"


def model_id_to_cache_dir(model_id: str) -> str:
    """Convert model ID to HF cache directory name."""
    return "models--" + model_id.replace("/", "--")


def find_snapshot_dir(model_id: str) -> Optional[Path]:
    """Find the snapshot directory for a model in HF cache."""
    cache_dir = get_hf_cache_path()
    model_cache_dir = cache_dir / model_id_to_cache_dir(model_id)

    if not model_cache_dir.exists():
        return None

    snapshots_dir = model_cache_dir / "snapshots"
    if not snapshots_dir.exists():
        return None

    # Find the most recent snapshot
    snapshots = list(snapshots_dir.iterdir())
    if not snapshots:
        return None

    # Sort by modification time, newest first
    snapshots.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return snapshots[0]


def cleanup_corrupted_files(corrupted_files: List[str]) -> int:
    """
    Remove corrupted files and their blobs to force re-download.

    Returns number of files cleaned up.
    """
    cleaned = 0
    for file_path in corrupted_files:
        try:
            path = Path(file_path)

            # If it's a symlink, also remove the blob it points to
            if path.is_symlink():
                blob_path = None
                try:
                    blob_path = path.resolve()
                except FileNotFoundError:
                    # This is a broken symlink, so we can't resolve the blob
                    print(f"  Note: symlink {path.name} is broken.")

                # Remove the symlink itself
                path.unlink()
                print(f"  Removed symlink: {path.name}")

                # If we resolved a blob path and it exists, remove it
                if blob_path and blob_path.exists():
                    blob_path.unlink()
                    print(f"  Removed blob: {blob_path.name}")
            elif path.exists():
                path.unlink()
                print(f"  Removed file: {path.name}")

            cleaned += 1
        except Exception as e:
            print(f"  Warning: Failed to remove {file_path}: {e}")

    return cleaned


def validate_model(model_id: str, snapshot_dir: Path) -> List[str]:
    """
    Validate a model's checksums using model_file_verifier logic.

    Returns list of corrupted file paths.
    """
    corrupted_files = []

    # Fetch expected checksums from HuggingFace Hub
    try:
        expected_files = _load_file_infos_from_hf(repo_id=model_id)
    except IntegrityError as e:
        print(f"  Warning: {e}")
        return []
    except Exception as e:
        print(f"  Warning: Could not fetch checksums from Hub: {e}")
        return []

    if not expected_files:
        print("  No files found to validate")
        return []

    # Filter to only weight files and limit count
    weight_extensions = (".safetensors", ".bin", ".pt")
    weight_files = {
        k: v for k, v in expected_files.items() if k.endswith(weight_extensions)
    }

    if not weight_files:
        print("  No weight files found to validate")
        return []

    # Limit number of files to validate
    filenames = list(weight_files.keys())[:MAX_FILES_PER_MODEL]
    if len(weight_files) > MAX_FILES_PER_MODEL:
        print(f"  Limiting validation to {MAX_FILES_PER_MODEL} files")

    print(f"  Validating {len(filenames)} weight files...")

    # Compute actual checksums
    try:
        actual_manifest = _compute_manifest_from_folder(
            model_path=snapshot_dir,
            filenames=filenames,
            max_workers=4,
        )
    except Exception as e:
        print(f"  Warning: Failed to compute checksums: {e}")
        return []

    # Compare and find corrupted files
    for filename in filenames:
        expected_info = weight_files[filename]
        actual_info = actual_manifest.files.get(filename)

        if actual_info is None:
            # File doesn't exist locally - not corruption, just incomplete download
            continue

        if actual_info.sha256 != expected_info.sha256:
            print(f"  CORRUPTED: {filename}")
            print(f"    Expected: {expected_info.sha256[:16]}...")
            print(f"    Actual:   {actual_info.sha256[:16]}...")
            corrupted_files.append(str(snapshot_dir / filename))

    return corrupted_files


def main():
    print("=" * 70)
    print("CI Weight Checksum Validation")
    print("=" * 70)
    print()

    start_time = time.time()

    cache_dir = get_hf_cache_path()
    if not cache_dir.exists():
        print(f"HF cache directory not found: {cache_dir}")
        return

    print(f"HF cache: {cache_dir}")
    print(f"Critical models to validate: {len(CI_CRITICAL_MODELS)}")
    print()

    total_validated = 0
    total_corrupted = 0
    total_cleaned = 0

    for model_id in CI_CRITICAL_MODELS:
        # Check time limit
        elapsed = time.time() - start_time
        if elapsed > MAX_VALIDATION_TIME:
            print(f"\nTime limit reached ({elapsed:.1f}s), stopping validation")
            break

        print(f"[{model_id}]")

        # Find local snapshot
        snapshot_dir = find_snapshot_dir(model_id)
        if snapshot_dir is None:
            print("  Not cached locally, skipping")
            print()
            continue

        print(f"  Snapshot: {snapshot_dir.name[:12]}...")

        # Validate using model_file_verifier logic
        corrupted_files = validate_model(model_id, snapshot_dir)
        total_validated += 1

        if not corrupted_files:
            print("  Result: PASS - all checksums match")
        else:
            print(f"  Result: FAIL - {len(corrupted_files)} corrupted file(s)")
            total_corrupted += len(corrupted_files)

            # Clean up corrupted files
            print("  Cleaning up corrupted files...")
            cleaned = cleanup_corrupted_files(corrupted_files)
            total_cleaned += cleaned
            print(f"  Cleaned {cleaned} file(s) - will be re-downloaded on next run")

        print()

    elapsed = time.time() - start_time

    print("=" * 70)
    print(f"Checksum validation complete ({elapsed:.1f}s)")
    print(f"  Models validated: {total_validated}")
    print(f"  Corrupted files found: {total_corrupted}")
    print(f"  Files cleaned up: {total_cleaned}")
    print("=" * 70)

    if total_corrupted > 0:
        print("\nNOTE: Corrupted files were removed. They will be re-downloaded")
        print("when the model is loaded during tests.")


if __name__ == "__main__":
    main()
