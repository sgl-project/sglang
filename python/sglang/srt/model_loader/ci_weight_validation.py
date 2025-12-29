"""
CI-specific weight validation and cache cleanup utilities.

This module contains validation and cleanup logic that is ONLY used in CI environments.
These functions handle:
- Validating safetensors files for corruption
- Checking for missing shards in sharded models
- Cleaning up corrupted files (selective or full cache deletion)
- Automatic retry logic for corrupted downloads

For regular users, weight_utils.py provides simple download functionality without
the overhead of validation and automatic cleanup. The CI-specific behavior is
gated by is_in_ci() checks in weight_utils.py.
"""

import glob as glob_module
import json
import logging
import os
import re
import shutil
from typing import List, Optional, Tuple

import safetensors

from sglang.srt.utils import log_info_on_rank0

logger = logging.getLogger(__name__)


def _validate_safetensors_file(file_path: str) -> bool:
    """
    Validate that a safetensors file is readable and not corrupted.

    Args:
        file_path: Path to the safetensors file

    Returns:
        True if the file is valid, False if corrupted
    """
    try:
        # Attempt to open and read the header
        # This will fail if the file is corrupted or incomplete
        with safetensors.safe_open(file_path, framework="pt", device="cpu") as f:
            # Just accessing the keys validates the header is readable
            _ = list(f.keys())
        return True
    except Exception as e:
        logger.warning(
            "Corrupted safetensors file detected: %s - %s: %s",
            file_path,
            type(e).__name__,
            str(e),
        )
        return False


def _check_index_files_exist(snapshot_dir: str) -> Tuple[bool, Optional[str]]:
    """
    Check if all files listed in safetensors index files actually exist on disk.

    This catches cases where the snapshot directory exists but files are missing
    (e.g., due to incomplete downloads or corrupted cache).

    Args:
        snapshot_dir: Path to the model snapshot directory

    Returns:
        Tuple of (all_exist, error_message)
    """
    # Find all safetensors index files
    index_files = [
        f for f in os.listdir(snapshot_dir) if f.endswith(".safetensors.index.json")
    ]

    if not index_files:
        # No index files means it's not a sharded model, skip this check
        return True, None

    for index_file in index_files:
        index_path = os.path.join(snapshot_dir, index_file)

        # Check if index file is a broken symlink (exists in listing but blob missing)
        if os.path.islink(index_path) and not os.path.exists(index_path):
            # Broken symlink - clean it up so download can proceed
            try:
                blob_path = os.path.realpath(index_path)
                os.remove(index_path)
                logger.warning(
                    "Removed broken index symlink: %s (blob missing)", index_file
                )
                # Also try to remove dangling blob reference if it somehow exists
                if os.path.exists(blob_path):
                    os.remove(blob_path)
            except Exception as e:
                logger.error("Failed to remove broken symlink %s: %s", index_file, e)
            return (
                False,
                f"Broken index file symlink: {index_file} (cleaned up, will re-download)",
            )

        try:
            with open(index_path) as f:
                index_data = json.load(f)

            weight_map = index_data.get("weight_map", {})
            if not weight_map:
                continue

            # Check that all files in weight_map exist
            required_files = set(weight_map.values())
            missing_files = []

            for file_name in required_files:
                file_path = os.path.join(snapshot_dir, file_name)
                # Check both existence and that it's not a broken symlink
                if not os.path.exists(file_path):
                    missing_files.append(file_name)

            if missing_files:
                return (
                    False,
                    f"Missing {len(missing_files)} file(s) from index {index_file}: {missing_files[:3]}{'...' if len(missing_files) > 3 else ''}",
                )

        except FileNotFoundError as e:
            # Index file was listed but can't be read - could be race condition or broken state
            logger.warning("Failed to read index file %s: %s", index_file, e)
            return (
                False,
                f"Index file {index_file} unreadable (will re-download)",
            )
        except Exception as e:
            logger.warning("Failed to read index file %s: %s", index_file, e)
            continue

    return True, None


def _validate_sharded_model(
    snapshot_dir: str, weight_files: List[str]
) -> Tuple[bool, Optional[str], List[str]]:
    """
    Validate that all model shards are present and not corrupted.

    Args:
        snapshot_dir: Path to the model snapshot directory
        weight_files: List of weight file paths

    Returns:
        Tuple of (is_valid, error_message, corrupted_files)
        - corrupted_files: List of file paths that are corrupted (for selective cleanup)
    """
    # First, check if all files from the index actually exist
    # This catches missing files that wouldn't be found by glob
    index_check_valid, index_error = _check_index_files_exist(snapshot_dir)
    if not index_check_valid:
        return False, index_error, []

    # Pattern for sharded files: model-00001-of-00009.safetensors
    shard_pattern = re.compile(r"(.*?)-(\d+)-of-(\d+)\.(safetensors|bin)")

    # Group files by shard pattern (prefix-*-of-N)
    shard_groups = {}
    for f in weight_files:
        base_name = os.path.basename(f)
        match = shard_pattern.match(base_name)
        if match:
            prefix = match.group(1)
            total_shards_str = match.group(3)
            suffix = match.group(4)

            group_key = f"{prefix}-of-{total_shards_str}.{suffix}"
            if group_key not in shard_groups:
                shard_groups[group_key] = {
                    "prefix": prefix,
                    "total": int(total_shards_str),
                    "suffix": suffix,
                    "found_shards": [],
                    "files": [],
                }

            shard_id = int(match.group(2))
            shard_groups[group_key]["found_shards"].append(shard_id)
            shard_groups[group_key]["files"].append(f)

    # Track corrupted files for selective cleanup
    corrupted_files = []

    # Validate each shard group
    for group_key, group_info in shard_groups.items():
        total_shards = group_info["total"]
        found_shards = set(group_info["found_shards"])
        expected_shards = set(range(1, total_shards + 1))

        # Check for missing shards
        missing_shards = expected_shards - found_shards
        if missing_shards:
            return (
                False,
                f"Missing shards in {group_key}: {sorted(missing_shards)}",
                [],
            )

        # Validate safetensors files for corruption
        if group_info["suffix"] == "safetensors":
            for f in group_info["files"]:
                if not _validate_safetensors_file(f):
                    corrupted_files.append(f)

        # Check for required index file for safetensors shards
        if group_info["suffix"] == "safetensors":
            index_file = os.path.join(
                snapshot_dir, f"{group_info['prefix']}.safetensors.index.json"
            )
            if not os.path.exists(index_file):
                return (
                    False,
                    f"Missing index file: {os.path.basename(index_file)}",
                    [],
                )

    if corrupted_files:
        return (
            False,
            f"Corrupted shard files: {[os.path.basename(f) for f in corrupted_files]}",
            corrupted_files,
        )

    return True, None, []


def _cleanup_corrupted_files_selective(
    model_name_or_path: str, corrupted_files: List[str]
) -> int:
    """
    Selectively remove corrupted files and their blobs to force re-download.

    This is more efficient than removing the entire model cache as it only
    re-downloads corrupted files rather than the entire model.

    Args:
        model_name_or_path: Model identifier
        corrupted_files: List of corrupted file paths (symlinks in snapshot)

    Returns:
        Number of files successfully cleaned up
    """
    cleaned_count = 0

    for file_path in corrupted_files:
        try:
            # Resolve symlink to get blob path before deleting symlink
            if os.path.islink(file_path):
                blob_path = os.path.realpath(file_path)

                # Delete the symlink
                os.remove(file_path)
                logger.info(
                    "Removed corrupted symlink: %s", os.path.basename(file_path)
                )

                # Delete the blob (the actual corrupted data)
                if os.path.exists(blob_path):
                    os.remove(blob_path)
                    logger.info(
                        "Removed corrupted blob: %s", os.path.basename(blob_path)
                    )

                cleaned_count += 1
            elif os.path.exists(file_path):
                # Not a symlink, just delete the file
                os.remove(file_path)
                logger.info("Removed corrupted file: %s", os.path.basename(file_path))
                cleaned_count += 1

        except Exception as e:
            logger.error(
                "Failed to remove corrupted file %s: %s",
                os.path.basename(file_path),
                e,
            )

    if cleaned_count > 0:
        logger.warning(
            "Removed %d corrupted file(s) for %s. "
            "These will be re-downloaded on next load.",
            cleaned_count,
            model_name_or_path,
        )

    return cleaned_count


def _cleanup_corrupted_model_cache(
    model_name_or_path: str, snapshot_dir: str, reason: str
) -> None:
    """
    Remove entire corrupted model cache directory to force a clean re-download.

    This is used when we cannot selectively clean (e.g., missing shards, incomplete
    downloads with unknown affected files).

    Args:
        model_name_or_path: Model identifier
        snapshot_dir: Path to the snapshot directory
        reason: Reason for cleanup
    """
    # Navigate up to the model root directory: snapshots/hash -> snapshots -> model_root
    repo_folder = os.path.abspath(os.path.join(snapshot_dir, "..", ".."))

    try:
        logger.warning(
            "Removing entire cache for %s at %s. Reason: %s",
            model_name_or_path,
            repo_folder,
            reason,
        )
        shutil.rmtree(repo_folder)
        logger.info("Successfully removed corrupted cache directory")
    except Exception as e:
        logger.error(
            "Failed to remove corrupted cache directory %s: %s. "
            "Manual cleanup may be required.",
            repo_folder,
            e,
        )


def ci_validate_and_cleanup_local_snapshot(
    model_name_or_path: str,
    found_local_snapshot_dir: str,
    local_weight_files: List[str],
) -> bool:
    """
    CI-specific validation and cleanup for local model snapshots.

    This function validates the local snapshot and performs automatic cleanup
    if corruption or missing files are detected. This behavior is only appropriate
    for CI environments where we want automatic recovery.

    Args:
        model_name_or_path: Model identifier for logging
        found_local_snapshot_dir: Path to the local snapshot directory
        local_weight_files: List of weight file paths found in the snapshot

    Returns:
        True if the snapshot is valid and can be used, False if it was invalid
        and cleanup was performed (caller should re-download)
    """
    # Check for incomplete files and clean up if found
    repo_folder = os.path.abspath(os.path.join(found_local_snapshot_dir, "..", ".."))
    blobs_dir = os.path.join(repo_folder, "blobs")

    # Check for incomplete download markers
    incomplete_files = []
    if os.path.isdir(blobs_dir):
        incomplete_files = glob_module.glob(os.path.join(blobs_dir, "*.incomplete"))

    if incomplete_files:
        log_info_on_rank0(
            logger,
            f"Found {len(incomplete_files)} .incomplete files in {blobs_dir} for "
            f"{model_name_or_path}. Will clean up and re-download.",
        )
        _cleanup_corrupted_model_cache(
            model_name_or_path,
            found_local_snapshot_dir,
            f"Incomplete download detected ({len(incomplete_files)} incomplete files)",
        )
        return False

    # Validate sharded models and check for corruption
    if local_weight_files:
        is_valid, error_msg, corrupted_files = _validate_sharded_model(
            found_local_snapshot_dir, local_weight_files
        )
        if not is_valid:
            if corrupted_files:
                # Selective cleanup: only remove corrupted files
                log_info_on_rank0(
                    logger,
                    f"Found {len(corrupted_files)} corrupted file(s) for "
                    f"{model_name_or_path}: {error_msg}. "
                    "Will selectively clean and re-download only these files.",
                )
                _cleanup_corrupted_files_selective(model_name_or_path, corrupted_files)
                return False
            else:
                # Missing shards (not corruption) - let snapshot_download handle it.
                # IMPORTANT: Do NOT delete the entire cache here, as other processes
                # (TP/EP ranks) may already be loading weights from these files.
                log_info_on_rank0(
                    logger,
                    f"Validation failed for {model_name_or_path}: {error_msg}. "
                    "Will attempt to download missing files.",
                )
                return False

        # Also validate single (non-sharded) safetensors files
        for f in local_weight_files:
            base_name = os.path.basename(f)
            # Check if this is a single model file (not sharded)
            # Include adapter_model.safetensors for LoRA adapters
            if base_name in [
                "model.safetensors",
                "pytorch_model.safetensors",
                "adapter_model.safetensors",
            ]:
                if not _validate_safetensors_file(f):
                    log_info_on_rank0(
                        logger,
                        f"Corrupted model file {base_name} for {model_name_or_path}. "
                        "Will selectively clean and re-download this file.",
                    )
                    # Selective cleanup for single file
                    _cleanup_corrupted_files_selective(model_name_or_path, [f])
                    return False

    return True


def _validate_weights_after_download(
    hf_folder: str,
    allow_patterns: List[str],
    model_name_or_path: str,
) -> bool:
    """
    Validate downloaded weight files to catch corruption early.

    This function validates safetensors files after download to catch
    corruption issues (truncated downloads, network errors, etc.) before
    model loading fails with cryptic errors. If corruption is found,
    the corrupted files are automatically cleaned up.

    Args:
        hf_folder: Path to the downloaded model folder
        allow_patterns: Patterns used to match weight files
        model_name_or_path: Model identifier for error messages

    Returns:
        True if all files are valid, False if corrupted files were found and cleaned up
    """
    # Find all weight files that were downloaded
    weight_files: List[str] = []
    for pattern in allow_patterns:
        weight_files.extend(glob_module.glob(os.path.join(hf_folder, pattern)))

    if not weight_files:
        return True  # No weight files to validate

    # Validate safetensors files
    corrupted_files = []
    for f in weight_files:
        if f.endswith(".safetensors") and os.path.exists(f):
            if not _validate_safetensors_file(f):
                corrupted_files.append(os.path.basename(f))

    if corrupted_files:
        # Clean up corrupted files so next attempt re-downloads them
        _cleanup_corrupted_files_selective(
            model_name_or_path,
            [os.path.join(hf_folder, f) for f in corrupted_files],
        )
        log_info_on_rank0(
            logger,
            f"Downloaded model files are corrupted for {model_name_or_path}: "
            f"{corrupted_files}. The corrupted files have been removed. "
            "Will retry download.",
        )
        return False

    return True


def ci_download_with_validation_and_retry(
    model_name_or_path: str,
    allow_patterns: List[str],
    ignore_patterns,
    cache_dir: Optional[str],
    revision: Optional[str],
    max_retries: int = 3,
) -> str:
    """
    CI-specific download with validation and automatic retry on corruption.

    This function handles the download of model weights in CI environments,
    with automatic validation and retry logic for handling corrupted downloads.

    Args:
        model_name_or_path: The model name or path
        allow_patterns: The allowed patterns for weight files
        ignore_patterns: The patterns to filter out weight files
        cache_dir: The cache directory to store model weights
        revision: The revision of the model
        max_retries: Maximum number of download retries if corruption is detected

    Returns:
        str: The path to the downloaded model weights

    Raises:
        RuntimeError: If download fails after max_retries attempts
    """
    # Lazy imports to avoid circular dependencies
    import huggingface_hub.constants
    from huggingface_hub import snapshot_download
    from tqdm.auto import tqdm

    class DisabledTqdm(tqdm):
        def __init__(self, *args, **kwargs):
            kwargs["disable"] = True
            super().__init__(*args, **kwargs)

    # Retry loop for handling corrupted downloads
    for attempt in range(max_retries):
        hf_folder = snapshot_download(
            model_name_or_path,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            cache_dir=cache_dir,
            tqdm_class=DisabledTqdm,
            revision=revision,
            local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
        )

        # Validate downloaded files to catch corruption early
        is_valid = _validate_weights_after_download(
            hf_folder, allow_patterns, model_name_or_path
        )

        if is_valid:
            return hf_folder

        # Validation failed, corrupted files were cleaned up
        if attempt < max_retries - 1:
            log_info_on_rank0(
                logger,
                f"Retrying download for {model_name_or_path} "
                f"(attempt {attempt + 2}/{max_retries})...",
            )
        else:
            raise RuntimeError(
                f"Downloaded model files are still corrupted for "
                f"{model_name_or_path} after {max_retries} attempts. "
                "This may indicate a persistent issue with the model files "
                "on Hugging Face Hub or network problems."
            )

    # This should never be reached, but just in case
    return hf_folder


def ci_validate_and_clean_hf_cache(model_path: str) -> None:
    """
    Validate and clean corrupted safetensors files in HF cache before loading.

    This function is needed because HFRunner (used in tests) calls transformers'
    from_pretrained() directly, which bypasses SGLang's weight validation.
    Corrupted cached files can cause cryptic errors like "EOF while parsing"
    from safetensors.

    Only runs in CI to avoid overhead for regular users.

    Args:
        model_path: Model identifier (e.g., "meta-llama/Llama-2-7b")
    """
    from sglang.utils import is_in_ci

    if not is_in_ci():
        return

    # Skip for local paths
    if os.path.isdir(model_path):
        return

    try:
        import huggingface_hub.constants

        # Find the HF cache directory for this model
        cache_dir = huggingface_hub.constants.HF_HUB_CACHE
        repo_folder = os.path.join(
            cache_dir,
            huggingface_hub.constants.REPO_ID_SEPARATOR.join(
                ["models", *model_path.split("/")]
            ),
        )

        if not os.path.isdir(repo_folder):
            return

        # Find snapshot directories
        snapshots_dir = os.path.join(repo_folder, "snapshots")
        if not os.path.isdir(snapshots_dir):
            return

        # Check each snapshot for corrupted files
        corrupted_files = []
        for snapshot_hash in os.listdir(snapshots_dir):
            snapshot_dir = os.path.join(snapshots_dir, snapshot_hash)
            if not os.path.isdir(snapshot_dir):
                continue

            # Find all safetensors files
            safetensors_files = glob_module.glob(
                os.path.join(snapshot_dir, "*.safetensors")
            )

            for sf_file in safetensors_files:
                # Skip broken symlinks (os.path.exists returns False for them)
                if not os.path.exists(sf_file):
                    continue

                if not _validate_safetensors_file(sf_file):
                    corrupted_files.append(sf_file)

        if corrupted_files:
            logger.warning(
                "HFRunner: Found %d corrupted safetensors file(s) for %s. "
                "Removing to force re-download.",
                len(corrupted_files),
                model_path,
            )
            _cleanup_corrupted_files_selective(model_path, corrupted_files)

    except Exception as e:
        # Don't fail if validation itself fails - let HF handle it
        logger.debug("HF cache validation failed (non-fatal): %s", e)
