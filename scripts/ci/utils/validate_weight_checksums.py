#!/usr/bin/env python3
"""
Validate model weight integrity using SHA256 checksums.

This script runs during CI initialization to detect corrupted model weights
by comparing local file checksums against HuggingFace Hub metadata.
If corruption is detected, the corrupted files are removed to trigger
a fresh re-download.

This catches bit-flip corruption that the basic safetensors header validation
might miss, preventing silent 0.0 accuracy test failures.

NOTE: This script ONLY runs in CI environments (SGLANG_IS_IN_CI=true).
"""

import hashlib
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


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


# Models commonly used in CI tests that should be validated with checksums
# These are models where 0.0 accuracy failures have been observed
CI_CRITICAL_MODELS = [
    # GLM models - observed 0.0 accuracy failures
    "THUDM/glm-4-9b-chat",
    "THUDM/glm-4-9b-chat-1m",
    # Llama models - commonly tested
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    # Qwen models
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    # Mistral
    "mistralai/Mistral-7B-Instruct-v0.3",
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
    # meta-llama/Llama-3.1-8B -> models--meta-llama--Llama-3.1-8B
    return "models--" + model_id.replace("/", "--")


def compute_sha256(file_path: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(64 * 1024):
            sha256.update(chunk)
    return sha256.hexdigest()


def get_expected_checksums_from_hub(model_id: str) -> Optional[Dict[str, str]]:
    """
    Fetch expected SHA256 checksums from HuggingFace Hub.

    Returns dict mapping filename -> sha256, or None if fetch fails.
    """
    try:
        from huggingface_hub import HfFileSystem

        fs = HfFileSystem()
        files = fs.ls(model_id, detail=True)

        checksums = {}
        for file_info in files:
            if file_info.get("type") != "file":
                continue

            filename = Path(file_info.get("name", "")).name

            # Only validate weight files
            if not filename.endswith((".safetensors", ".bin", ".pt")):
                continue

            # Get SHA256 from LFS info (large files) or compute from content
            lfs_info = file_info.get("lfs")
            if lfs_info and "sha256" in lfs_info:
                checksums[filename] = lfs_info["sha256"]
            elif "sha256" in file_info:
                checksums[filename] = file_info["sha256"]

        return checksums

    except Exception as e:
        print(f"  Warning: Could not fetch checksums from Hub: {e}")
        return None


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


def validate_model_checksums(
    model_id: str, expected_checksums: Dict[str, str], snapshot_dir: Path
) -> Tuple[bool, List[str]]:
    """
    Validate local files against expected checksums.

    Returns:
        Tuple of (all_valid, list_of_corrupted_files)
    """
    corrupted_files = []
    validated_count = 0

    for filename, expected_sha256 in expected_checksums.items():
        if validated_count >= MAX_FILES_PER_MODEL:
            print(f"  (reached max files limit, skipping remaining)")
            break

        file_path = snapshot_dir / filename
        if not file_path.exists():
            # File doesn't exist locally - not corruption, just incomplete download
            continue

        # Resolve symlink to actual blob
        if file_path.is_symlink():
            real_path = file_path.resolve()
            if not real_path.exists():
                print(f"  {filename}: BROKEN SYMLINK")
                corrupted_files.append(str(file_path))
                continue
        else:
            real_path = file_path

        print(f"  Validating {filename}...", end=" ", flush=True)
        actual_sha256 = compute_sha256(real_path)
        validated_count += 1

        if actual_sha256 != expected_sha256:
            print(f"CORRUPTED!")
            print(f"    Expected: {expected_sha256[:16]}...")
            print(f"    Actual:   {actual_sha256[:16]}...")
            corrupted_files.append(str(file_path))
        else:
            print("OK")

    return len(corrupted_files) == 0, corrupted_files


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

        # Fetch expected checksums from Hub
        expected_checksums = get_expected_checksums_from_hub(model_id)
        if expected_checksums is None:
            print("  Could not fetch checksums from Hub, skipping")
            print()
            continue

        if not expected_checksums:
            print("  No weight files found to validate, skipping")
            print()
            continue

        print(f"  Found {len(expected_checksums)} weight files to validate")

        # Validate checksums
        is_valid, corrupted_files = validate_model_checksums(
            model_id, expected_checksums, snapshot_dir
        )
        total_validated += 1

        if is_valid:
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
