"""
CI-specific weight validation and cache cleanup utilities.

This module contains validation and cleanup logic that is ONLY used in CI environments.
These functions handle:
- Validating safetensors files for corruption
- Checking for missing shards in sharded models
- Cleaning up corrupted files (selective or full cache deletion)
- Automatic retry logic for corrupted downloads
- Validating config/tokenizer files completeness to enable offline mode

For regular users, weight_utils.py provides simple download functionality without
the overhead of validation and automatic cleanup. The CI-specific behavior is
gated by is_in_ci() checks in weight_utils.py.
"""

import glob as glob_module
import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
import time
from typing import List, Optional, Tuple

import safetensors

from sglang.srt.utils import log_info_on_rank0

logger = logging.getLogger(__name__)


def _get_per_run_marker_dir() -> str:
    """
    Get the directory for per-run validation markers.

    These markers are specific to the current CI run and are not shared across
    runners. They are stored in a temporary directory that is cleaned up after
    the run completes.

    Returns:
        Path to per-run marker directory
    """
    # Prefer RUNNER_TEMP (GitHub Actions) or TMPDIR, fallback to /tmp
    base_dir = os.environ.get("RUNNER_TEMP", os.environ.get("TMPDIR", "/tmp"))
    marker_dir = os.path.join(base_dir, "sglang_ci_offline_markers")
    os.makedirs(marker_dir, exist_ok=True)
    return marker_dir


def _get_per_run_marker_path(snapshot_dir: str) -> Optional[str]:
    """
    Get the path to per-run validation marker file for a snapshot.

    Per-run markers are specific to the current CI run and are not shared
    across runners. This prevents cross-runner cache state pollution.

    Args:
        snapshot_dir: Path to snapshot directory

    Returns:
        Path to per-run marker file or None if snapshot_dir is invalid
    """
    if not snapshot_dir or not os.path.isdir(snapshot_dir):
        return None

    normalized_dir = os.path.realpath(snapshot_dir).rstrip("/")
    dir_hash = hashlib.sha256(normalized_dir.encode("utf-8")).hexdigest()[:12]

    marker_dir = _get_per_run_marker_dir()
    return os.path.join(marker_dir, f"{dir_hash}.json")


def _read_per_run_marker(snapshot_dir: str) -> Optional[dict]:
    """
    Read per-run validation marker for a snapshot.

    Args:
        snapshot_dir: Path to snapshot directory

    Returns:
        Marker dict if exists and valid, None otherwise
    """
    marker_path = _get_per_run_marker_path(snapshot_dir)
    if not marker_path or not os.path.exists(marker_path):
        return None

    try:
        with open(marker_path, "r", encoding="utf-8") as f:
            marker = json.load(f)

        # Validate marker structure
        if not isinstance(marker, dict):
            return None

        required_keys = ["timestamp", "model_id", "snapshot_hash", "validation_passed"]
        if not all(k in marker for k in required_keys):
            return None

        if marker.get("validation_passed") is not True:
            return None

        return marker

    except Exception as e:
        logger.debug("Failed to read per-run marker from %s: %s", marker_path, e)
        return None


def _write_per_run_marker(
    snapshot_dir: str, model_id: str, required_files: Optional[list] = None
) -> None:
    """
    Write per-run validation marker for a snapshot.

    Args:
        snapshot_dir: Path to snapshot directory
        model_id: Model identifier
        required_files: List of required files that were validated
    """
    marker_path = _get_per_run_marker_path(snapshot_dir)
    if not marker_path:
        logger.debug("Cannot write per-run marker: invalid snapshot_dir")
        return

    from datetime import datetime

    snapshot_hash = os.path.basename(snapshot_dir)

    marker = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model_id": model_id,
        "snapshot_hash": snapshot_hash,
        "validation_passed": True,
        "required_files": required_files or [],
    }

    try:
        marker_dir = os.path.dirname(marker_path)
        os.makedirs(marker_dir, exist_ok=True)

        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=marker_dir,
            delete=False,
            suffix=".tmp",
        ) as f:
            temp_path = f.name
            json.dump(marker, f, indent=2)

        os.replace(temp_path, marker_path)
        logger.debug("Wrote per-run marker to %s", marker_path)
    except Exception as e:
        logger.warning("Failed to write per-run marker to %s: %s", marker_path, e)
        try:
            if "temp_path" in locals() and os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:
            pass


def validate_cache_lightweight(
    snapshot_dir: str, requires_hf_quant_config: bool = False
) -> bool:
    """
    Lightweight runtime validation for cache completeness.

    This is used during test runs to ensure the current runner's cache
    is complete before enabling offline mode. Much faster than full validation
    as it only checks file existence, not corruption.

    Args:
        snapshot_dir: Path to the model snapshot directory
        requires_hf_quant_config: If True, hf_quant_config.json must exist
                                  (required for modelopt quantization)

    Returns:
        True if cache is complete, False otherwise
    """
    # Check required config files
    required_files = [
        "config.json",
        "tokenizer_config.json",
    ]

    for fname in required_files:
        if not os.path.exists(os.path.join(snapshot_dir, fname)):
            return False

    # Check tokenizer files (at least one must exist)
    tokenizer_files = [
        "tokenizer.json",
        "tokenizer.model",
        "tiktoken.model",
    ]

    has_tokenizer = any(
        os.path.exists(os.path.join(snapshot_dir, fname)) for fname in tokenizer_files
    )
    if not has_tokenizer:
        return False

    # Check for trust_remote_code dynamic module files if needed
    # When auto_map exists in config.json, the model requires custom Python files
    # These files must be present for offline mode to work
    config_path = os.path.join(snapshot_dir, "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)

            auto_map = config.get("auto_map", {})
            if auto_map and isinstance(auto_map, dict):
                # Extract Python module files from auto_map
                # auto_map format: {"AutoConfig": "configuration_xxx.ConfigClass", ...}
                # We need to check if the .py files exist
                custom_files = set()
                for key, value in auto_map.items():
                    if isinstance(value, str) and "." in value:
                        # Extract module name (e.g., "configuration_xxx" from "configuration_xxx.ConfigClass")
                        module_name = value.split(".")[0]
                        custom_files.add(f"{module_name}.py")

                # Check if all custom files exist in snapshot directory
                for custom_file in custom_files:
                    custom_file_path = os.path.join(snapshot_dir, custom_file)
                    if not os.path.exists(custom_file_path):
                        logger.debug(
                            "Custom module file not in snapshot: %s for %s",
                            custom_file,
                            snapshot_dir,
                        )
                        return False
                    elif not os.path.isfile(custom_file_path):
                        logger.debug(
                            "Custom module path exists but not a file: %s",
                            custom_file_path,
                        )
                        return False
        except (json.JSONDecodeError, OSError, KeyError) as e:
            # If we can't read config.json, it will be caught by earlier validation
            logger.debug("Failed to check auto_map in config.json: %s", e)

    # Check for weight files with index self-consistency
    index_path = os.path.join(snapshot_dir, "model.safetensors.index.json")
    has_index = os.path.exists(index_path)

    if has_index:
        # If index exists, validate that all shards listed in it exist
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                index_data = json.load(f)
            weight_map = index_data.get("weight_map", {})
            if weight_map:
                # Check that all shard files referenced in index exist
                required_shards = set(weight_map.values())
                for shard_name in required_shards:
                    shard_path = os.path.join(snapshot_dir, shard_name)
                    if not os.path.exists(shard_path):
                        logger.debug(
                            "Index validation failed: missing shard %s in %s",
                            shard_name,
                            snapshot_dir,
                        )
                        return False
        except (json.JSONDecodeError, OSError, KeyError) as e:
            logger.debug("Failed to validate index file %s: %s", index_path, e)
            return False
    else:
        # No index file - check for weight files and validate shard completeness
        safetensors_files = glob_module.glob(
            os.path.join(snapshot_dir, "*.safetensors")
        )
        if not safetensors_files:
            return False

        # Check shard completeness for sharded models (e.g., model-00001-of-00047.safetensors)
        # Pattern: prefix-NNNNN-of-NNNNN.safetensors
        shard_pattern = re.compile(r"(.*?)-(\d+)-of-(\d+)\.safetensors$")
        shard_groups = {}

        for f in safetensors_files:
            base_name = os.path.basename(f)
            match = shard_pattern.match(base_name)
            if match:
                prefix = match.group(1)
                shard_id = int(match.group(2))
                total_shards = int(match.group(3))
                group_key = f"{prefix}-of-{total_shards}"

                if group_key not in shard_groups:
                    shard_groups[group_key] = {
                        "total": total_shards,
                        "found_shards": set(),
                    }
                shard_groups[group_key]["found_shards"].add(shard_id)

        # Validate each shard group has all expected shards
        for group_key, group_info in shard_groups.items():
            total_shards = group_info["total"]
            found_shards = group_info["found_shards"]
            expected_shards = set(range(1, total_shards + 1))
            missing_shards = expected_shards - found_shards

            if missing_shards:
                logger.debug(
                    "Shard validation failed: missing shards %s in %s for %s",
                    sorted(missing_shards),
                    group_key,
                    snapshot_dir,
                )
                return False

    # Check hf_quant_config.json if required (for modelopt quantization)
    if requires_hf_quant_config:
        hf_quant_path = os.path.join(snapshot_dir, "hf_quant_config.json")
        if not os.path.exists(hf_quant_path):
            return False

    return True


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


def _validate_pytorch_bin_file(file_path: str) -> bool:
    """
    Validate that a PyTorch .bin file is readable and not corrupted.

    This catches corruption issues like truncated downloads or invalid archives
    that would cause errors like:
    "RuntimeError: PytorchStreamReader failed reading file data/X: invalid header
    or archive is corrupted"

    Args:
        file_path: Path to the .bin file

    Returns:
        True if the file is valid, False if corrupted
    """
    try:
        import torch

        # Use weights_only=True for security and to avoid executing arbitrary code
        # mmap=False to fully read the file and catch all corruption
        torch.load(file_path, map_location="cpu", weights_only=True, mmap=False)
        return True
    except Exception as e:
        logger.warning(
            "Corrupted PyTorch bin file detected: %s - %s: %s",
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
        # Shards may be 0-indexed (e.g. inclusionAI/Ring-2.5-1T) or 1-indexed
        # (e.g. deepseek-ai/DeepSeek-V3); both are valid HF conventions.
        min_idx = min(found_shards) if found_shards else 1
        expected_shards = set(range(min_idx, min_idx + total_shards))

        # Check for missing shards
        missing_shards = expected_shards - found_shards
        if missing_shards:
            return (
                False,
                f"Missing shards in {group_key}: {sorted(missing_shards)}",
                [],
            )

        # Validate weight files for corruption
        if group_info["suffix"] == "safetensors":
            for f in group_info["files"]:
                if not _validate_safetensors_file(f):
                    corrupted_files.append(f)
        elif group_info["suffix"] == "bin":
            for f in group_info["files"]:
                if not _validate_pytorch_bin_file(f):
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

        # Also validate single (non-sharded) weight files
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
            # Also validate single PyTorch .bin files
            elif base_name in [
                "pytorch_model.bin",
                "model.bin",
                "adapter_model.bin",
            ]:
                if not _validate_pytorch_bin_file(f):
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

    # Validate weight files (safetensors and .bin)
    corrupted_files = []
    for f in weight_files:
        if f.endswith(".safetensors") and os.path.exists(f):
            if not _validate_safetensors_file(f):
                corrupted_files.append(os.path.basename(f))
        elif f.endswith(".bin") and os.path.exists(f):
            if not _validate_pytorch_bin_file(f):
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


def _get_lock_file_path(
    model_name_or_path: str, cache_dir: Optional[str] = None
) -> str:
    """
    Generate a unique lock file path for download coordination.

    In CI environments where multiple containers share an NFS-mounted HF cache,
    the lock file is placed on the shared cache directory so ALL containers
    coordinate on the same lock. This prevents cross-container .incomplete
    file race conditions.

    Falls back to /dev/shm (container-local) for non-CI or when the cache
    dir is not accessible.

    Args:
        model_name_or_path: Model identifier
        cache_dir: HF cache directory (None to use default)

    Returns:
        Path to the lock file
    """
    key_hash = hashlib.sha256(model_name_or_path.encode()).hexdigest()[:16]

    # In CI, place lock on the shared HF cache directory so that ALL containers
    # sharing the same NFS-mounted cache coordinate downloads.
    # /dev/shm is container-local and doesn't prevent cross-container races.
    try:
        import huggingface_hub.constants

        effective_cache_dir = cache_dir or huggingface_hub.constants.HF_HUB_CACHE
        if os.path.isdir(effective_cache_dir):
            lock_dir = os.path.join(effective_cache_dir, ".sglang_locks")
            os.makedirs(lock_dir, exist_ok=True)
            return os.path.join(lock_dir, f"download_{key_hash}.lock")
    except Exception:
        pass

    # Fallback to container-local lock
    if os.path.isdir("/dev/shm"):
        return f"/dev/shm/sglang_download_lock_{key_hash}"
    return f"/tmp/sglang_download_lock_{key_hash}"


def _cleanup_incomplete_blobs(model_name_or_path: str, cache_dir: Optional[str]) -> int:
    """
    Remove stale .incomplete files from the model's blobs directory.

    This is lighter than _cleanup_corrupted_model_cache (which deletes the
    entire cache). We only remove .incomplete files so snapshot_download
    starts fresh on retry, preserving any successfully downloaded blobs.

    Args:
        model_name_or_path: Model identifier (e.g., "meta-llama/Llama-2-7b-hf")
        cache_dir: HF cache directory (None to use default)

    Returns:
        Number of .incomplete files removed
    """
    try:
        import huggingface_hub.constants

        effective_cache_dir = cache_dir or huggingface_hub.constants.HF_HUB_CACHE
        repo_folder_name = huggingface_hub.constants.REPO_ID_SEPARATOR.join(
            ["models", *model_name_or_path.split("/")]
        )
        blobs_dir = os.path.join(effective_cache_dir, repo_folder_name, "blobs")

        if not os.path.isdir(blobs_dir):
            return 0

        incomplete_files = glob_module.glob(os.path.join(blobs_dir, "*.incomplete"))
        removed = 0
        for f in incomplete_files:
            try:
                os.remove(f)
                removed += 1
                logger.debug("Removed incomplete blob: %s", os.path.basename(f))
            except OSError as e:
                logger.debug(
                    "Failed to remove incomplete blob %s: %s", os.path.basename(f), e
                )

        if removed > 0:
            logger.warning(
                "Cleaned up %d .incomplete blob(s) for %s in %s",
                removed,
                model_name_or_path,
                blobs_dir,
            )
        return removed

    except Exception as e:
        logger.debug("Failed to clean up incomplete blobs: %s", e)
        return 0


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

    Uses filelock.FileLock on the shared HF cache directory to coordinate
    downloads across all processes AND all containers sharing the same
    NFS-mounted cache. Only one process downloads at a time; others wait
    for the lock then use the cached result.

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
    import filelock
    import huggingface_hub.constants
    from huggingface_hub import snapshot_download
    from tqdm.auto import tqdm

    class DisabledTqdm(tqdm):
        def __init__(self, *args, **kwargs):
            kwargs["disable"] = True
            super().__init__(*args, **kwargs)

    # Use filelock on the shared HF cache directory to coordinate downloads
    # across all processes AND all containers sharing the same NFS mount.
    # This prevents cross-container .incomplete file race conditions.
    lock_file_path = _get_lock_file_path(model_name_or_path, cache_dir)

    logger.info(
        "[CI Download] Process %d using lock file: %s",
        os.getpid(),
        lock_file_path,
    )

    # filelock.FileLock handles creation, acquisition, and release cleanly.
    # timeout=-1 means wait indefinitely (another container may be downloading
    # a large model for 30+ minutes).
    lock = filelock.FileLock(lock_file_path, timeout=-1, mode=0o666)

    logger.info(
        "[CI Download] Process %d waiting to acquire lock for %s",
        os.getpid(),
        model_name_or_path,
    )

    with lock:
        logger.info(
            "[CI Download] Process %d ACQUIRED lock for %s",
            os.getpid(),
            model_name_or_path,
        )

        # Re-check if another container already downloaded the model while
        # we were waiting for the lock. This avoids redundant downloads.
        try:
            from sglang.srt.model_loader.weight_utils import (
                _find_local_hf_snapshot_dir_unlocked,
            )

            cached_path = _find_local_hf_snapshot_dir_unlocked(
                model_name_or_path, cache_dir, allow_patterns, revision
            )
            if cached_path is not None:
                logger.info(
                    "[CI Download] Process %d found cached model after "
                    "acquiring lock (downloaded by another container): %s",
                    os.getpid(),
                    cached_path,
                )
                return cached_path
        except Exception as e:
            logger.debug(
                "[CI Download] Re-check for cached model failed (non-fatal): %s", e
            )

        # Clean up stale .incomplete files from previous failed downloads
        # before starting. Only do this once before the first attempt.
        cleaned = _cleanup_incomplete_blobs(model_name_or_path, cache_dir)
        if cleaned > 0:
            logger.info(
                "[CI Download] Pre-download cleanup: removed %d stale "
                ".incomplete file(s) for %s",
                cleaned,
                model_name_or_path,
            )

        hf_folder = None
        for attempt in range(max_retries):
            try:
                hf_folder = snapshot_download(
                    model_name_or_path,
                    allow_patterns=allow_patterns,
                    ignore_patterns=ignore_patterns,
                    cache_dir=cache_dir,
                    tqdm_class=DisabledTqdm,
                    revision=revision,
                    local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
                    # Force single-threaded downloads to prevent race conditions
                    # on NFS. HF hub defaults to max_workers=8, which can cause
                    # .incomplete file conflicts when multiple threads operate
                    # on the same files
                    max_workers=1,
                )
            except (FileNotFoundError, OSError) as e:
                # Race condition: .incomplete file was moved/deleted by another
                # process. With NFS-level locking this should be rare, but can
                # still happen if lock acquisition fails on some NFS setups.
                logger.warning(
                    "[CI Download] Process %d hit download error "
                    "(attempt %d/%d) for %s: %s: %s",
                    os.getpid(),
                    attempt + 1,
                    max_retries,
                    model_name_or_path,
                    type(e).__name__,
                    e,
                )
                if attempt < max_retries - 1:
                    # Backoff: 10s, 20s, 40s. Clean only the stale
                    # .incomplete files (not active ones from other processes).
                    backoff = 10 * (2**attempt)
                    logger.info(
                        "[CI Download] Cleaning up .incomplete files and "
                        "retrying in %ds...",
                        backoff,
                    )
                    _cleanup_incomplete_blobs(model_name_or_path, cache_dir)
                    time.sleep(backoff)
                    continue
                raise RuntimeError(
                    f"Download failed for {model_name_or_path} after "
                    f"{max_retries} attempts due to download errors. "
                    f"Last error: {type(e).__name__}: {e}"
                ) from e

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

        # Should never reach here, but return hf_folder just in case
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

            # Also find and validate PyTorch .bin files
            bin_files = glob_module.glob(os.path.join(snapshot_dir, "*.bin"))

            for bin_file in bin_files:
                # Skip broken symlinks (os.path.exists returns False for them)
                if not os.path.exists(bin_file):
                    continue

                if not _validate_pytorch_bin_file(bin_file):
                    corrupted_files.append(bin_file)

        if corrupted_files:
            logger.warning(
                "HFRunner: Found %d corrupted weight file(s) for %s. "
                "Removing to force re-download.",
                len(corrupted_files),
                model_path,
            )
            _cleanup_corrupted_files_selective(model_path, corrupted_files)

    except Exception as e:
        # Don't fail if validation itself fails - let HF handle it
        logger.debug("HF cache validation failed (non-fatal): %s", e)
