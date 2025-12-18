import json
import logging
import os
import re
import shutil
from typing import List, Optional, Tuple

import safetensors

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
