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
from typing import List, Optional, Tuple

import safetensors

from sglang.srt.utils import log_info_on_rank0

logger = logging.getLogger(__name__)

# Validation marker version - increment when validation logic changes
# v2: Added trust_remote_code module validation (modeling_*.py must exist in snapshot)
# v3: Added remote file existence checks for hf_quant_config.json
# v5: Invalidate all previous markers to force fresh validation
VALIDATION_MARKER_VERSION = "5"


def _remote_file_exists(
    repo_id: str, filename: str, revision: Optional[str], allow_remote_check: bool
) -> Optional[bool]:
    """
    Check if a file exists on Hugging Face Hub for a specific revision.

    Args:
        repo_id: Repository ID (e.g., "meta-llama/Llama-2-7b-hf")
        filename: File name to check (e.g., "hf_quant_config.json")
        revision: Git revision (commit hash, branch, or tag). None means default branch.
        allow_remote_check: Whether remote checks are allowed (e.g., CI validation phase)

    Returns:
        True if file exists on hub, False if it doesn't exist, None if we cannot determine
        (network error or remote check not allowed - be conservative and assume incomplete)
    """
    if not allow_remote_check:
        logger.debug(
            "Remote check disabled for %s/%s, returning None (unknown)",
            repo_id,
            filename,
        )
        return None

    try:
        from huggingface_hub import HfApi

        api = HfApi()
        exists = api.file_exists(repo_id=repo_id, filename=filename, revision=revision)
        logger.debug(
            "Remote file check: %s/%s (revision=%s) exists=%s",
            repo_id,
            filename,
            revision or "default",
            exists,
        )
        return exists
    except Exception as e:
        # Network errors, auth issues, repo not found, etc.
        # Return None (unknown) - caller will treat as optional
        logger.debug(
            "Failed to check remote file existence for %s/%s (revision=%s): %s. "
            "Will treat as optional.",
            repo_id,
            filename,
            revision or "default",
            e,
        )
        return None


def _get_validation_marker_path(snapshot_dir: str) -> Optional[str]:
    """
    Get the path to validation marker file for a snapshot.

    Marker is stored in /tmp to avoid permission issues with HF cache directory.
    Marker key is sha256(snapshot_dir) to avoid any collisions regardless of
    model_name_or_path format.

    Args:
        snapshot_dir: Path to snapshot directory

    Returns:
        Path to marker file or None if snapshot_dir is invalid
    """
    if not snapshot_dir or not os.path.isdir(snapshot_dir):
        return None

    # Normalize path to avoid marker misses due to trailing slashes or symlinks
    # realpath resolves symlinks, rstrip removes trailing slashes
    normalized_dir = os.path.realpath(snapshot_dir).rstrip("/")

    # Use sha256 of normalized snapshot_dir path as unique key
    # This avoids any collision issues with repo naming or snapshot hash reuse
    dir_hash = hashlib.sha256(normalized_dir.encode("utf-8")).hexdigest()[:12]

    # Store in /tmp with directory hash
    return f"/tmp/sglang_hf_validation_{dir_hash}.json"


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


def _remove_per_run_marker(snapshot_dir: str) -> None:
    """
    Remove per-run validation marker for a snapshot.

    Args:
        snapshot_dir: Path to snapshot directory
    """
    marker_path = _get_per_run_marker_path(snapshot_dir)
    if marker_path and os.path.exists(marker_path):
        try:
            os.remove(marker_path)
            logger.debug("Removed per-run marker: %s", marker_path)
        except Exception as e:
            logger.warning("Failed to remove per-run marker %s: %s", marker_path, e)


def _read_validation_marker(snapshot_dir: str) -> Optional[dict]:
    """
    Read validation marker for a snapshot.

    Args:
        snapshot_dir: Path to snapshot directory

    Returns:
        Marker dict with keys: version, validated_at, validation_passed
        None if marker doesn't exist or is invalid or validation_passed is not True
    """
    marker_path = _get_validation_marker_path(snapshot_dir)
    if not marker_path:
        return None

    if not os.path.exists(marker_path):
        return None

    try:
        with open(marker_path, "r", encoding="utf-8") as f:
            marker = json.load(f)

        # Validate marker structure
        if not isinstance(marker, dict):
            return None

        required_keys = ["version", "validated_at", "validation_passed"]
        if not all(key in marker for key in required_keys):
            return None

        # Check version match
        if marker["version"] != VALIDATION_MARKER_VERSION:
            logger.debug(
                "Validation marker version mismatch: %s != %s, will re-validate",
                marker["version"],
                VALIDATION_MARKER_VERSION,
            )
            return None

        # Explicitly check validation_passed is True (defensive check)
        # Even though we only write markers on success, this guards against
        # manual edits or future code changes
        if marker.get("validation_passed") is not True:
            logger.debug(
                "Validation marker has validation_passed=%s, treating as invalid",
                marker.get("validation_passed"),
            )
            return None

        return marker
    except (json.JSONDecodeError, OSError) as e:
        logger.debug("Failed to read validation marker at %s: %s", marker_path, e)
        return None


def _write_validation_marker(snapshot_dir: str, passed: bool) -> None:
    """
    Write validation marker for a snapshot (atomic write).

    IMPORTANT: We only cache successful validations. Failed validations are NOT
    cached to allow retry after files are downloaded.

    Args:
        snapshot_dir: Path to snapshot directory
        passed: Whether validation passed
    """
    if not passed:
        # Don't cache failures - allow retry on next launch
        return

    marker_path = _get_validation_marker_path(snapshot_dir)
    if not marker_path:
        logger.debug("Cannot write marker: invalid snapshot_dir")
        return

    from datetime import datetime

    marker = {
        "version": VALIDATION_MARKER_VERSION,
        "validated_at": datetime.utcnow().isoformat() + "Z",
        "validation_passed": passed,
    }

    try:
        # Atomic write: write to temp file then os.replace
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

        # Atomic replace (overwrites existing file if any)
        os.replace(temp_path, marker_path)
        logger.debug("Wrote validation marker to %s (passed=%s)", marker_path, passed)
    except Exception as e:
        logger.warning("Failed to write validation marker to %s: %s", marker_path, e)
        # Clean up temp file if it exists
        try:
            if "temp_path" in locals() and os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:
            pass


def _validate_json_file(file_path: str, file_name: str) -> bool:
    """
    Validate that a JSON file exists, is non-empty, and can be parsed.

    Args:
        file_path: Path to the JSON file
        file_name: Name of the file (for logging)

    Returns:
        True if the file is valid, False otherwise
    """
    if not os.path.exists(file_path):
        logger.debug("CI cache validation: %s not found at %s", file_name, file_path)
        return False

    if not os.path.isfile(file_path):
        logger.warning(
            "CI cache validation: %s is not a file: %s", file_name, file_path
        )
        return False

    # Check if file is non-empty
    try:
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            logger.warning("CI cache validation: %s is empty: %s", file_name, file_path)
            return False
    except OSError as e:
        logger.warning("CI cache validation: Cannot get size of %s: %s", file_name, e)
        return False

    # Try to parse JSON
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            json.load(f)
        return True
    except json.JSONDecodeError as e:
        logger.warning(
            "CI cache validation: %s is not valid JSON: %s - %s",
            file_name,
            file_path,
            e,
        )
        return False
    except Exception as e:
        logger.warning(
            "CI cache validation: Failed to read %s: %s - %s",
            file_name,
            file_path,
            e,
        )
        return False


def _validate_config_and_tokenizer_files(
    snapshot_dir: str,
    model_id: Optional[str] = None,
    revision: Optional[str] = None,
    allow_remote_check: bool = False,
) -> Tuple[bool, List[str]]:
    """
    Validate that critical config and tokenizer files exist and are valid.

    This checks for:
    - config.json (required)
    - tokenizer_config.json (required)
    - generation_config.json (optional but validated if present)
    - hf_quant_config.json (conditionally required based on Hub) - for FP4/FP8/ModelOpt
    - quantize_config.json / quant_config.json (optional but validated if present) - for AWQ/GPTQ
    - params.json (optional but validated if present) - for Mistral native format
    - preprocessor_config.json (optional but validated if present) - for vision models
    - trust_remote_code dynamic modules (required if auto_map present in config.json)
    - At least one tokenizer file: tokenizer.json, tokenizer.model, or tiktoken.model

    Args:
        snapshot_dir: Path to the model snapshot directory
        model_id: Model repository ID (e.g., "meta-llama/Llama-2-7b-hf"), used for remote checks
        revision: Git revision (commit hash), used for remote checks
        allow_remote_check: Whether to check Hub for file existence to determine requirements

    Returns:
        Tuple of (is_valid, missing_files)
        - is_valid: True if all required files are present and valid
        - missing_files: List of missing or invalid file names
    """
    missing_files = []

    # Check required config files
    required_files = [
        "config.json",
        "tokenizer_config.json",
    ]

    for file_name in required_files:
        file_path = os.path.join(snapshot_dir, file_name)
        if not _validate_json_file(file_path, file_name):
            missing_files.append(file_name)

    # Check optional generation_config.json (validate if exists)
    generation_config_path = os.path.join(snapshot_dir, "generation_config.json")
    if os.path.exists(generation_config_path):
        if not _validate_json_file(generation_config_path, "generation_config.json"):
            missing_files.append("generation_config.json (exists but invalid)")

    # Check hf_quant_config.json with remote existence check
    # This file is needed for quantized models (FP4/FP8/ModelOpt)
    # Example: nvidia/Llama-3.1-8B-Instruct-FP8, nvidia/DeepSeek-V3-0324-FP4
    hf_quant_config_path = os.path.join(snapshot_dir, "hf_quant_config.json")
    local_hf_quant_exists = os.path.exists(hf_quant_config_path)

    # Check if file exists on Hub for this revision
    # Only do remote check if model_id looks like a HF repo_id (org/model format)
    # Skip if it's a local path (absolute path or doesn't contain '/')
    remote_hf_quant_exists = None
    is_hf_repo = (
        model_id is not None
        and "/" in model_id
        and not os.path.isabs(model_id)
        and not model_id.startswith("/")
    )
    if is_hf_repo and allow_remote_check:
        remote_hf_quant_exists = _remote_file_exists(
            repo_id=model_id,
            filename="hf_quant_config.json",
            revision=revision,
            allow_remote_check=allow_remote_check,
        )

    # Apply conditional requirement logic
    if remote_hf_quant_exists is True:
        # Hub has this file for this revision - it's REQUIRED
        if not local_hf_quant_exists:
            missing_files.append(
                f"hf_quant_config.json (required: exists on Hub for revision {revision or 'default'} but missing locally)"
            )
            log_info_on_rank0(
                logger,
                f"Hub has hf_quant_config.json for {model_id} revision {revision or 'default'} "
                f"but local snapshot missing it. Cache incomplete, will not write marker.",
            )
        elif not _validate_json_file(hf_quant_config_path, "hf_quant_config.json"):
            missing_files.append("hf_quant_config.json (exists but invalid)")
    elif remote_hf_quant_exists is False:
        # Hub doesn't have this file - it's OPTIONAL
        # Only validate if it happens to exist locally
        if local_hf_quant_exists:
            if not _validate_json_file(hf_quant_config_path, "hf_quant_config.json"):
                missing_files.append("hf_quant_config.json (exists but invalid)")
    else:
        # remote_hf_quant_exists is None - unknown (network error or remote check disabled)
        # Treat as OPTIONAL - only enforce when we can positively confirm Hub has it
        if local_hf_quant_exists:
            # Local file exists - validate it
            if not _validate_json_file(hf_quant_config_path, "hf_quant_config.json"):
                missing_files.append("hf_quant_config.json (exists but invalid)")
        # If local file missing and remote unknown, just log it - don't block marker
        logger.debug(
            "Cannot verify hf_quant_config.json on Hub for %s (revision=%s), "
            "treating as optional since remote status unknown",
            model_id or "unknown",
            revision or "default",
        )

    # Check optional quantize_config.json / quant_config.json (validate if exists)
    # These files are needed for AWQ/GPTQ/AutoRound quantized models
    # Example: TheBloke/Llama-2-7B-AWQ, casperhansen/vicuna-7b-v1.5-awq
    for quant_config_name in ["quantize_config.json", "quant_config.json"]:
        quant_config_path = os.path.join(snapshot_dir, quant_config_name)
        if os.path.exists(quant_config_path):
            if not _validate_json_file(quant_config_path, quant_config_name):
                missing_files.append(f"{quant_config_name} (exists but invalid)")
            break  # Only need to check one of these

    # Check optional params.json (validate if exists)
    # This file is needed for Mistral native format models
    # Example: mistralai/Mistral-7B-v0.1
    params_json_path = os.path.join(snapshot_dir, "params.json")
    if os.path.exists(params_json_path):
        if not _validate_json_file(params_json_path, "params.json"):
            missing_files.append("params.json (exists but invalid)")

    # Check optional preprocessor_config.json (validate if exists)
    # This file is needed for vision/multimodal models
    # Example: llava-hf/llava-1.5-7b-hf, Qwen/Qwen2-VL-7B-Instruct
    preprocessor_config_path = os.path.join(snapshot_dir, "preprocessor_config.json")
    if os.path.exists(preprocessor_config_path):
        if not _validate_json_file(
            preprocessor_config_path, "preprocessor_config.json"
        ):
            missing_files.append("preprocessor_config.json (exists but invalid)")

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
                # NOTE: Some models (like nvidia/DeepSeek-V3-0324-FP4) have auto_map
                # but don't include modeling_*.py in their repo, relying on transformers
                # to fetch it from the base model. We MUST mark these as missing to
                # prevent offline mode, which would fail to load the dynamic modules.
                for custom_file in custom_files:
                    custom_file_path = os.path.join(snapshot_dir, custom_file)
                    if not os.path.exists(custom_file_path):
                        missing_files.append(
                            f"{custom_file} (required for trust_remote_code)"
                        )
                        logger.debug(
                            f"Custom module file not in snapshot: {custom_file} for {snapshot_dir}"
                        )
                    elif not os.path.isfile(custom_file_path):
                        missing_files.append(f"{custom_file} (exists but not a file)")
        except (json.JSONDecodeError, OSError, KeyError) as e:
            # If we can't read config.json, it will be caught by earlier validation
            logger.debug("Failed to check auto_map in config.json: %s", e)

    # Check for at least one tokenizer file
    tokenizer_files = [
        "tokenizer.json",
        "tokenizer.model",
        "tiktoken.model",
    ]

    tokenizer_found = False
    for tokenizer_file in tokenizer_files:
        tokenizer_path = os.path.join(snapshot_dir, tokenizer_file)
        if os.path.exists(tokenizer_path) and os.path.isfile(tokenizer_path):
            # For tokenizer.json, validate it's proper JSON
            if tokenizer_file == "tokenizer.json":
                if _validate_json_file(tokenizer_path, tokenizer_file):
                    tokenizer_found = True
                    break
            else:
                # For .model files, just check they're non-empty
                try:
                    if os.path.getsize(tokenizer_path) > 0:
                        tokenizer_found = True
                        break
                except OSError:
                    pass

    if not tokenizer_found:
        missing_files.append("tokenizer file")

    is_valid = len(missing_files) == 0
    return is_valid, missing_files


def ci_validate_cache_and_enable_offline_if_complete(
    snapshot_dir: str,
    weight_files: List[str],
    model_name_or_path: str,
) -> bool:
    """
    Validate local cache completeness (config/tokenizer/weights) and determine
    if offline mode can be safely enabled.

    This function uses a snapshot-level marker to cache validation results,
    so the heavy validation is done at most once per snapshot per runner.

    This function checks:
    1. Validation marker (if exists and version matches, skip re-validation)
    2. Config and tokenizer files (config.json, tokenizer_config.json, etc.)
    3. Weight files (safetensors shards, index files, corruption check)

    If all are present and valid, it returns True to signal that offline
    mode can be safely enabled.

    IMPORTANT: This should be called BEFORE any HF operations, and if it
    returns True, the caller should set HF_HUB_OFFLINE=1 for the server
    subprocess env ONLY (not global environment).

    Args:
        snapshot_dir: Path to the model snapshot directory
        weight_files: List of weight file paths to validate (must be non-empty)
        model_name_or_path: Model identifier for logging

    Returns:
        True if cache is complete and offline mode can be enabled, False otherwise
    """
    # Guard: weight_files is required
    if not weight_files:
        log_info_on_rank0(
            logger,
            f"CI_OFFLINE: No weight files provided, skip offline, keep online allowed - {model_name_or_path}",
        )
        return False

    # Fast-path: Check if validation marker exists and is valid
    # We only cache successful validations, so if marker exists, it means cache is complete
    marker = _read_validation_marker(snapshot_dir)
    if marker is not None:
        marker_path = _get_validation_marker_path(snapshot_dir)
        marker_name = os.path.basename(marker_path) if marker_path else "unknown"
        log_info_on_rank0(
            logger,
            f"CI_OFFLINE: Marker hit (marker={marker_name}), skip re-validation, offline mode will be enabled - {model_name_or_path}",
        )
        return True

    # No marker - perform full validation
    # (Failures are not cached, so we'll retry validation each time until success)

    # Extract revision (snapshot hash) from snapshot_dir path
    # snapshot_dir format: /path/to/cache/models--org--model/snapshots/<commit_hash>
    revision = os.path.basename(snapshot_dir)

    # Only allow remote checks if we're not in offline mode
    # This avoids unnecessary API calls and warnings in offline CI environments
    import huggingface_hub.constants

    allow_remote_check = not huggingface_hub.constants.HF_HUB_OFFLINE

    log_info_on_rank0(
        logger,
        f"CI_OFFLINE: No marker found, performing full validation "
        f"(snapshot={revision}, allow_remote_check={allow_remote_check}) - {model_name_or_path}",
    )

    # Validate config and tokenizer files with remote existence checks
    config_valid, missing_config_files = _validate_config_and_tokenizer_files(
        snapshot_dir=snapshot_dir,
        model_id=model_name_or_path,
        revision=revision,
        allow_remote_check=allow_remote_check,
    )

    if not config_valid:
        log_info_on_rank0(
            logger,
            f"CI_OFFLINE: Missing config/tokenizer files {missing_config_files}, skip offline, keep online allowed - {model_name_or_path}",
        )
        # Don't write marker for failures - allow retry after download
        return False

    # Validate weight files using existing validation from PR #15216
    # This checks for missing shards, corrupted safetensors, etc.
    weights_valid, error_msg, _ = _validate_sharded_model(snapshot_dir, weight_files)
    if not weights_valid:
        log_info_on_rank0(
            logger,
            f"CI_OFFLINE: Weight validation failed ({error_msg}), skip offline, keep online allowed - {model_name_or_path}",
        )
        # Don't write marker for failures - allow retry after download
        return False

    log_info_on_rank0(
        logger,
        f"CI_OFFLINE: Cache validation PASSED, offline mode will be enabled - {model_name_or_path}",
    )

    # Write marker with passed=True for future reuse
    # (Failures are not cached, so this only happens on success)
    _write_validation_marker(snapshot_dir, passed=True)
    return True


def _infer_component_type(component_name: str, component_info: list) -> str:
    """
    Infer component type from component name and info.

    Args:
        component_name: Name of the component (e.g., "scheduler", "tokenizer")
        component_info: Component info from model_index.json (e.g., ["diffusers", "SchedulerClass"])

    Returns:
        Component type string for validation rules
    """
    # Normalize component name for type detection
    name_lower = component_name.lower()

    # Infer type based on name
    if "scheduler" in name_lower:
        return "scheduler"
    elif "tokenizer" in name_lower:
        return "tokenizer"
    elif "image_processor" in name_lower:
        return "image_processor"
    elif "feature_extractor" in name_lower:
        return "feature_extractor"
    elif "processor" in name_lower:
        return "processor"
    else:
        # Default to model component (needs config.json + weights)
        return "model"


def _check_component_config(
    component_dir: str, component_type: str
) -> Tuple[bool, List[str]]:
    """
    Check if component has required config files based on type.

    Args:
        component_dir: Path to component directory
        component_type: Type of component (scheduler, tokenizer, processor, model, etc.)

    Returns:
        Tuple of (has_valid_config, list_of_candidates_tried)
    """
    if component_type == "scheduler":
        # Scheduler: scheduler_config.json or config.json
        candidates = ["scheduler_config.json", "config.json"]
        for candidate in candidates:
            candidate_path = os.path.join(component_dir, candidate)
            if _validate_json_file(candidate_path, candidate):
                return True, candidates
        return False, candidates

    elif component_type == "tokenizer":
        # Tokenizer must have actual tokenizer files (not just tokenizer_config.json)
        # Valid combinations:
        # - tokenizer.json
        # - tokenizer.model
        # - vocab.json + merges.txt
        candidates = [
            "tokenizer.json",
            "tokenizer.model",
            "vocab.json+merges.txt",
        ]

        # Check tokenizer.json (validate as JSON)
        tokenizer_json_path = os.path.join(component_dir, "tokenizer.json")
        if _validate_json_file(tokenizer_json_path, "tokenizer.json"):
            return True, candidates

        # Check tokenizer.model (non-empty file)
        tokenizer_model_path = os.path.join(component_dir, "tokenizer.model")
        if os.path.exists(tokenizer_model_path) and os.path.isfile(
            tokenizer_model_path
        ):
            try:
                if os.path.getsize(tokenizer_model_path) > 0:
                    return True, candidates
            except OSError:
                pass

        # Check vocab.json + merges.txt pair
        vocab_path = os.path.join(component_dir, "vocab.json")
        merges_path = os.path.join(component_dir, "merges.txt")
        if _validate_json_file(vocab_path, "vocab.json") and os.path.exists(
            merges_path
        ):
            return True, candidates

        return False, candidates

    elif component_type in ["processor", "feature_extractor", "image_processor"]:
        # Processor/feature_extractor/image_processor: preprocessor_config.json or config.json
        candidates = ["preprocessor_config.json", "config.json"]
        for candidate in candidates:
            candidate_path = os.path.join(component_dir, candidate)
            if _validate_json_file(candidate_path, candidate):
                return True, candidates
        return False, candidates

    else:
        # Default model components: config.json
        candidates = ["config.json"]
        config_path = os.path.join(component_dir, "config.json")
        if _validate_json_file(config_path, "config.json"):
            return True, candidates
        return False, candidates


def _check_component_weights(component_dir: str) -> bool:
    """
    Check if component directory has weight files.

    Args:
        component_dir: Path to component directory

    Returns:
        True if weight files found, False otherwise
    """
    weight_patterns = ["*.safetensors", "*.bin", "*.pt", "*.pth"]

    for pattern in weight_patterns:
        weight_files = glob_module.glob(os.path.join(component_dir, pattern))
        if weight_files:
            return True

    return False


def _format_component_list(components: List[str], max_show: int = 5) -> str:
    """
    Format component list with truncation.

    Args:
        components: List of component names
        max_show: Maximum number to show before truncating

    Returns:
        Formatted string like "comp1, comp2, comp3" or "comp1, comp2, +3 more"
    """
    if len(components) <= max_show:
        return ", ".join(components)
    else:
        shown = components[:max_show]
        remaining = len(components) - max_show
        return f"{', '.join(shown)}, +{remaining} more"


def _validate_diffusion_model(
    snapshot_dir: str,
) -> Tuple[bool, Optional[str]]:
    """
    Validate diffusion model (diffusers pipeline) cache completeness.

    This validation is based on model_index.json as the single source of truth.
    Error reporting uses coarse-grained error codes unless verbose mode is enabled.

    Error codes:
    - DIFFUSERS_INVALID_INDEX: model_index.json missing or corrupted
    - DIFFUSERS_INVALID_COMPONENTS: model_index.json has no valid components
    - DIFFUSERS_MISSING_COMPONENT: component directory or config missing
    - DIFFUSERS_MISSING_WEIGHTS: component weights missing

    Args:
        snapshot_dir: Path to the model snapshot directory

    Returns:
        Tuple of (is_valid, error_message)
        - (True, None) if validation passed
        - (False, error_code_with_components) if validation failed
    """
    # Check verbose mode from environment
    verbose = os.environ.get("SGLANG_CI_VALIDATE_VERBOSE") == "1"

    # 1. Check for model_index.json (required for diffusers models)
    model_index_path = os.path.join(snapshot_dir, "model_index.json")
    if not os.path.exists(model_index_path):
        return False, "DIFFUSERS_INVALID_INDEX: model_index.json not found"

    # Parse model_index.json
    try:
        with open(model_index_path, "r", encoding="utf-8") as f:
            model_index = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        if verbose:
            return False, f"DIFFUSERS_INVALID_INDEX: model_index.json parse error - {e}"
        return False, "DIFFUSERS_INVALID_INDEX: model_index.json corrupted"

    # 2. Extract components (non-underscore keys with list values)
    components = {
        k: v
        for k, v in model_index.items()
        if not k.startswith("_") and isinstance(v, list)
    }

    if not components:
        return False, "DIFFUSERS_INVALID_COMPONENTS: no valid components defined"

    # Categorize errors by type
    missing_dirs = []
    missing_configs = []
    missing_configs_verbose = []
    missing_weights = []

    # 3. Validate each component
    for component_name, component_info in components.items():
        component_dir = os.path.join(snapshot_dir, component_name)

        # Component directory must exist
        if not os.path.isdir(component_dir):
            missing_dirs.append(component_name)
            continue

        # Infer component type for validation rules
        component_type = _infer_component_type(component_name, component_info)

        # Check for required config files based on component type
        has_valid_config, config_candidates = _check_component_config(
            component_dir, component_type
        )

        if not has_valid_config:
            missing_configs.append(component_name)
            if verbose:
                candidates_str = ", ".join(config_candidates)
                missing_configs_verbose.append(
                    f"{component_name} (tried: {candidates_str})"
                )
            continue

        # 4. Check for weights if component needs them
        # These components don't require weight files (config-only)
        needs_weights = component_type not in [
            "scheduler",
            "tokenizer",
            "processor",
            "feature_extractor",
            "image_processor",
        ]

        if needs_weights:
            has_weights = _check_component_weights(component_dir)
            if not has_weights:
                missing_weights.append(component_name)

    # 5. Build error message based on categorized errors
    if missing_dirs or missing_configs or missing_weights:
        errors = []

        if missing_dirs:
            dir_str = _format_component_list(missing_dirs)
            if verbose:
                errors.append(f"DIFFUSERS_MISSING_COMPONENT (dirs): {dir_str}")
            else:
                errors.append(f"DIFFUSERS_MISSING_COMPONENT(dir): {dir_str}")

        if missing_configs:
            if verbose:
                config_str = "; ".join(missing_configs_verbose)
                errors.append(f"DIFFUSERS_MISSING_COMPONENT (configs): {config_str}")
            else:
                config_str = _format_component_list(missing_configs)
                errors.append(f"DIFFUSERS_MISSING_COMPONENT(cfg): {config_str}")

        if missing_weights:
            weight_str = _format_component_list(missing_weights)
            errors.append(f"DIFFUSERS_MISSING_WEIGHTS: {weight_str}")

        return False, " | ".join(errors)

    return True, None


def validate_cache_with_detailed_reason(
    snapshot_dir: str, weight_files: List[str], model_name_or_path: str
) -> Tuple[bool, Optional[str]]:
    """
    Validate cache and return detailed reason for failure.

    This function performs validation without relying on shared validation markers.
    Used by prevalidate_cached_models.py to provide detailed feedback.

    Args:
        snapshot_dir: Path to the model snapshot directory
        weight_files: List of weight file paths to validate
        model_name_or_path: Model identifier for logging

    Returns:
        Tuple of (success, reason):
        - (True, None) if validation passed
        - (False, reason_str) if validation failed with specific reason
    """
    # Guard: weight_files is required
    if not weight_files:
        return False, "No weight files provided"

    # Perform full validation and capture failure reasons
    revision = os.path.basename(snapshot_dir)

    # Read from environment variable instead of huggingface_hub.constants
    allow_remote_check = os.environ.get("HF_HUB_OFFLINE") != "1"

    # Validate config and tokenizer files
    config_valid, missing_config_files = _validate_config_and_tokenizer_files(
        snapshot_dir=snapshot_dir,
        model_id=model_name_or_path,
        revision=revision,
        allow_remote_check=allow_remote_check,
    )

    if not config_valid:
        missing_files_str = ", ".join(missing_config_files)
        return False, f"Missing config/tokenizer files: {missing_files_str}"

    # Validate weight files
    weights_valid, error_msg, _ = _validate_sharded_model(snapshot_dir, weight_files)
    if not weights_valid:
        return False, f"Weight validation failed: {error_msg}"

    # All validations passed
    return True, None


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
        expected_shards = set(range(1, total_shards + 1))

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


def _get_lock_file_path(model_name_or_path: str) -> str:
    """
    Generate a unique lock file path for download coordination.

    Uses file-based locking (fcntl.flock) to ensure only one process downloads
    while others wait. This works regardless of how processes are spawned
    (mp.Process, torchrun, etc.).

    Args:
        model_name_or_path: Model identifier

    Returns:
        Path to the lock file
    """
    # Create a unique hash based on model name only (not cache_dir)
    # This ensures all processes coordinate on the same lock regardless of
    # cache_dir configuration differences between processes
    key_hash = hashlib.sha256(model_name_or_path.encode()).hexdigest()[:16]

    # Use /dev/shm (shared memory filesystem) for lock files because:
    # 1. It's always local to the machine (not NFS)
    # 2. It properly supports file locking
    # 3. It's shared across all processes on the same machine
    # Fall back to /tmp if /dev/shm doesn't exist
    if os.path.isdir("/dev/shm"):
        return f"/dev/shm/sglang_download_lock_{key_hash}"
    return f"/tmp/sglang_download_lock_{key_hash}"


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

    Uses file-based locking (fcntl.flock) to prevent HuggingFace hub race
    conditions where multiple processes try to download simultaneously,
    causing .incomplete file conflicts. Only one process downloads at a time;
    others wait for the lock then use the cached result.

    This approach works regardless of how processes are spawned (mp.Process,
    torchrun, etc.) since it doesn't rely on environment variables.

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
    import fcntl

    import huggingface_hub.constants
    from huggingface_hub import snapshot_download
    from tqdm.auto import tqdm

    class DisabledTqdm(tqdm):
        def __init__(self, *args, **kwargs):
            kwargs["disable"] = True
            super().__init__(*args, **kwargs)

    # Use file-based locking to serialize downloads across all processes
    # This prevents HF hub race conditions with .incomplete files
    lock_file_path = _get_lock_file_path(model_name_or_path)

    # Log lock file path for debugging
    logger.info(
        "[CI Download] Process %d using lock file: %s",
        os.getpid(),
        lock_file_path,
    )

    # Create lock file if it doesn't exist
    lock_file = open(lock_file_path, "w")

    try:
        # Acquire exclusive lock - blocks until lock is available
        # This ensures only one process downloads at a time
        logger.info(
            "[CI Download] Process %d waiting to acquire lock for %s",
            os.getpid(),
            model_name_or_path,
        )
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        logger.info(
            "[CI Download] Process %d ACQUIRED lock for %s",
            os.getpid(),
            model_name_or_path,
        )

        # Now we have exclusive access - perform download with retry logic
        hf_folder = None
        for attempt in range(max_retries):
            hf_folder = snapshot_download(
                model_name_or_path,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
                cache_dir=cache_dir,
                tqdm_class=DisabledTqdm,
                revision=revision,
                local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
                # Force single-threaded downloads to prevent race conditions on NFS
                # HF hub defaults to max_workers=8, which can cause .incomplete file
                # conflicts when multiple threads operate on the same files
                max_workers=1,
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

        # Should never reach here, but return hf_folder just in case
        return hf_folder

    finally:
        # Always release the lock
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        lock_file.close()
        logger.info(
            "[CI Download] Process %d RELEASED lock for %s",
            os.getpid(),
            model_name_or_path,
        )


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
