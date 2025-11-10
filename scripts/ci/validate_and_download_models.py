#!/usr/bin/env python3
"""
Validate model integrity for CI runners and download if needed.

This script checks HuggingFace cache for model completeness and downloads
missing models. It exits with code 1 if download was required (indicating
cache corruption), which causes the CI job to fail and surface cache issues.
"""

import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from huggingface_hub import constants, snapshot_download

    HF_HUB_AVAILABLE = True
except ImportError:
    print(
        "Warning: huggingface_hub not available. Install with: pip install huggingface_hub"
    )
    HF_HUB_AVAILABLE = False

try:
    from safetensors import safe_open

    SAFETENSORS_AVAILABLE = True
except ImportError:
    print("Warning: safetensors not available. Install with: pip install safetensors")
    SAFETENSORS_AVAILABLE = False


# Mapping of runner labels to their required models
# Add new runner labels and models here as needed
RUNNER_LABEL_MODEL_MAP: Dict[str, List[str]] = {
    "8-gpu-h200": ["deepseek-ai/DeepSeek-V3-0324", "moonshotai/Kimi-K2-Thinking"],
}


def get_hf_cache_dir() -> str:
    """Get the HuggingFace cache directory."""
    if HF_HUB_AVAILABLE:
        return constants.HF_HUB_CACHE

    # Fallback to environment variable or default
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    return os.path.join(hf_home, "hub")


def get_model_cache_path(model_id: str, cache_dir: str) -> Optional[Path]:
    """
    Find the model's cache directory in HuggingFace hub cache.

    Args:
        model_id: Model identifier (e.g., "deepseek-ai/DeepSeek-V3-0324")
        cache_dir: HuggingFace cache directory

    Returns:
        Path to model's snapshot directory, or None if not found
    """
    # Convert model_id to cache directory name format
    # "deepseek-ai/DeepSeek-V3-0324" -> "models--deepseek-ai--DeepSeek-V3-0324"
    cache_model_name = "models--" + model_id.replace("/", "--")
    model_path = Path(cache_dir) / cache_model_name

    if not model_path.exists():
        return None

    # Find the most recent snapshot directory
    snapshots_dir = model_path / "snapshots"
    if not snapshots_dir.exists():
        return None

    # Get all snapshot directories (sorted by modification time, most recent first)
    snapshot_dirs = sorted(
        [d for d in snapshots_dir.iterdir() if d.is_dir()],
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )

    if not snapshot_dirs:
        return None

    return snapshot_dirs[0]


def check_incomplete_files(model_path: Path, cache_dir: str) -> List[str]:
    """
    Check for incomplete download marker files specific to this model.

    Args:
        model_path: Path to model's snapshot directory
        cache_dir: HuggingFace cache directory

    Returns:
        List of incomplete files found for this specific model
    """
    incomplete_in_snapshot = []

    # Check if any files in the snapshot are symlinks to .incomplete blobs
    # This ensures we only flag incomplete files for THIS specific model,
    # not other models that might be downloading concurrently
    for file_path in model_path.glob("*"):
        if file_path.is_symlink():
            try:
                target = file_path.resolve()
                # Check if the symlink target has .incomplete suffix
                if str(target).endswith(".incomplete"):
                    incomplete_in_snapshot.append(str(target))
            except (OSError, RuntimeError):
                # Broken symlink - also indicates incomplete download
                incomplete_in_snapshot.append(str(file_path))

    return incomplete_in_snapshot


def validate_safetensors_file(file_path: Path) -> Tuple[bool, Optional[str]]:
    """
    Validate that a safetensors file is readable and not corrupted.

    Args:
        file_path: Path to the safetensors file

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not SAFETENSORS_AVAILABLE:
        # Skip validation if safetensors library is not available
        return True, None

    try:
        # Attempt to open and read the header
        # This will fail if the file is corrupted or incomplete
        with safe_open(file_path, framework="pt", device="cpu") as f:
            # Just accessing the keys validates the header is readable
            _ = f.keys()
        return True, None
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        # Return detailed error for debugging
        return False, f"{error_type}: {error_msg}"


def validate_model_shards(model_path: Path) -> Tuple[bool, Optional[str]]:
    """
    Validate that all model shards are present and complete.

    Args:
        model_path: Path to model's snapshot directory

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Pattern for sharded files: model-00001-of-00009.safetensors or pytorch_model-00001-of-00009.bin
    shard_pattern = re.compile(
        r"(?:model|pytorch_model)-(\d+)-of-(\d+)\.(safetensors|bin)"
    )

    # Find all shard files (both .safetensors and .bin)
    shard_files = (
        list(model_path.glob("model-*-of-*.safetensors"))
        + list(model_path.glob("model-*-of-*.bin"))
        + list(model_path.glob("pytorch_model-*-of-*.bin"))
    )

    if not shard_files:
        # No sharded files - check for single model file
        single_files = list(model_path.glob("model.safetensors")) or list(
            model_path.glob("pytorch_model.bin")
        )
        if single_files:
            # Validate the single safetensors file if it exists
            if single_files[0].suffix == ".safetensors":
                is_valid, error_msg = validate_safetensors_file(single_files[0])
                if not is_valid:
                    return False, f"Corrupted file {single_files[0].name}: {error_msg}"
            return True, None
        return False, "No model files found (safetensors or bin)"

    # Extract total shard count from any shard filename
    total_shards = None
    for shard_file in shard_files:
        match = shard_pattern.search(shard_file.name)
        if match:
            total_shards = int(match.group(2))
            break

    if total_shards is None:
        return False, "Could not determine total shard count from filenames"

    # Check that all shards exist
    expected_shards = set(range(1, total_shards + 1))
    found_shards = set()

    for shard_file in shard_files:
        match = shard_pattern.search(shard_file.name)
        if match:
            shard_num = int(match.group(1))
            found_shards.add(shard_num)

    missing_shards = expected_shards - found_shards

    if missing_shards:
        missing_list = sorted(missing_shards)
        return False, f"Missing shards: {missing_list} (expected {total_shards} total)"

    # Check for index file
    index_file = model_path / "model.safetensors.index.json"
    if not index_file.exists():
        return False, "Missing model.safetensors.index.json"

    # Validate each safetensors shard file for corruption
    print(f"  Validating {len(shard_files)} shard file(s) for corruption...")
    for shard_file in shard_files:
        if shard_file.suffix == ".safetensors":
            is_valid, error_msg = validate_safetensors_file(shard_file)
            if not is_valid:
                return False, f"Corrupted shard {shard_file.name}: {error_msg}"

    return True, None


def validate_model(model_id: str, cache_dir: str) -> Tuple[bool, Optional[str]]:
    """
    Validate a model's cache integrity.

    Args:
        model_id: Model identifier
        cache_dir: HuggingFace cache directory

    Returns:
        Tuple of (is_valid, error_message)
    """
    print(f"Validating model: {model_id}")

    # Find model in cache
    model_path = get_model_cache_path(model_id, cache_dir)
    if model_path is None:
        return False, "Model not found in cache"

    print(f"  Found in cache: {model_path}")

    # Check for incomplete files
    incomplete_files = check_incomplete_files(model_path, cache_dir)
    if incomplete_files:
        return False, f"Found incomplete download files: {len(incomplete_files)} files"

    # Validate shards
    is_valid, error_msg = validate_model_shards(model_path)
    if not is_valid:
        return False, error_msg

    print(f"  ✓ Model validated successfully")
    return True, None


def download_model(model_id: str) -> bool:
    """
    Download a model from HuggingFace.

    Args:
        model_id: Model identifier

    Returns:
        True if download succeeded, False otherwise
    """
    if not HF_HUB_AVAILABLE:
        print(f"ERROR: Cannot download model - huggingface_hub not available")
        return False

    print(f"Downloading model: {model_id}")
    print(f"  This may take a while for large models...")

    try:
        snapshot_download(
            repo_id=model_id,
            allow_patterns=["*.safetensors", "*.bin", "*.json", "*.txt", "*.model"],
            ignore_patterns=["*.msgpack", "*.h5", "*.ot"],  # codespell:ignore ot
        )
        print(f"  ✓ Download completed: {model_id}")
        return True
    except Exception as e:
        print(f"  ✗ Download failed: {e}")
        return False


def get_runner_labels() -> List[str]:
    """
    Get the runner labels from environment variables.

    GitHub Actions doesn't expose runner labels directly as environment variables.
    Workflows should set the RUNNER_LABELS environment variable with a comma-separated
    list of labels (e.g., "self-hosted,8-gpu-h200,linux").

    Returns:
        List of runner labels, empty list if not set
    """
    labels_str = os.environ.get("RUNNER_LABELS", "")
    if not labels_str:
        return []

    # Split by comma and strip whitespace
    return [label.strip() for label in labels_str.split(",") if label.strip()]


def should_validate_runner(runner_labels: List[str]) -> bool:
    """
    Check if the runner should have model validation based on its labels.

    Args:
        runner_labels: List of runner labels

    Returns:
        True if any label matches a configured label in RUNNER_LABEL_MODEL_MAP
    """
    if not runner_labels:
        return False

    # Check if any label is in the configured map
    return any(label in RUNNER_LABEL_MODEL_MAP for label in runner_labels)


def get_required_models(runner_labels: List[str]) -> List[str]:
    """
    Get list of models required based on runner labels.

    Args:
        runner_labels: List of runner labels (e.g., ["self-hosted", "8-gpu-h200", "linux"])

    Returns:
        List of model identifiers to validate (deduplicated)
    """
    all_models = []

    for label in runner_labels:
        if label in RUNNER_LABEL_MODEL_MAP:
            models = RUNNER_LABEL_MODEL_MAP[label]
            print(
                f"  ✓ Matched label configuration: '{label}' -> {len(models)} model(s)"
            )
            all_models.extend(models)

    if not all_models:
        print(f"  ⚠ No configuration found for any label in: {runner_labels}")

    # Remove duplicates while preserving order
    seen = set()
    unique_models = []
    for model in all_models:
        if model not in seen:
            seen.add(model)
            unique_models.append(model)

    return unique_models


def main() -> int:
    """
    Main validation logic.

    Returns:
        0 if all models are valid or runner doesn't need validation
        1 if models needed to be downloaded or validation failed
    """
    print("=" * 70)
    print("Model Validation for CI Runners")
    print("=" * 70)

    runner_labels = get_runner_labels()
    print(f"Runner labels: {', '.join(runner_labels) if runner_labels else 'NOT SET'}")

    # Check if this runner needs validation
    if not should_validate_runner(runner_labels):
        print(
            "Skipping validation: No runner labels match configured model requirements"
        )
        return 0

    print(f"Proceeding with model validation for this runner")

    # Get required models for these runner labels
    required_models = get_required_models(runner_labels)

    if not required_models:
        print(f"Warning: No models configured for labels: {runner_labels}")
        return 0

    print(f"Models to validate: {required_models}")
    print("-" * 70)

    # Get cache directory
    cache_dir = get_hf_cache_dir()
    print(f"HuggingFace cache: {cache_dir}")
    print("-" * 70)

    # Track validation results
    models_needing_download = []
    validation_errors = []

    # Validate each required model
    for model_id in required_models:
        is_valid, error_msg = validate_model(model_id, cache_dir)

        if not is_valid:
            print(f"  ✗ Validation failed: {error_msg}")
            models_needing_download.append(model_id)
            validation_errors.append(f"{model_id}: {error_msg}")

    print("-" * 70)

    # If all models are valid, exit successfully
    if not models_needing_download:
        print("✓ All models validated successfully!")
        return 0

    # Models need to be downloaded
    print(f"⚠ Cache validation failed for {len(models_needing_download)} model(s)")
    for error in validation_errors:
        print(f"  - {error}")

    print("-" * 70)
    print("Attempting to download missing/corrupted models...")
    print("-" * 70)

    download_failed = False
    for model_id in models_needing_download:
        if not download_model(model_id):
            download_failed = True

    print("-" * 70)

    if download_failed:
        print("✗ FAILED: Some models could not be downloaded")
        return 1

    # All downloads succeeded, but we still exit with error to flag cache issues
    print("✗ FAILED: Models were downloaded due to cache corruption/missing files")
    print("This indicates the cache was invalid and needed to be repaired.")
    print("Failing the job to surface this issue for investigation.")
    return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
