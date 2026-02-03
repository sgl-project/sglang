#!/usr/bin/env python3
"""
Pre-validate all cached HuggingFace models to provide detailed feedback.

This script runs once during CI initialization (in prepare_runner.sh) to:
1. Scan snapshots in ~/.cache/huggingface/hub/ (with time/quantity limits)
2. Validate completeness (config/tokenizer/weights)
3. Output detailed failure reasons for debugging

NOTE: This script no longer writes shared validation markers. Each test run
independently validates its cache using per-run markers to avoid cross-runner
cache state pollution.
"""

import glob
import json
import os
import sys
import time
from pathlib import Path

# Add python directory to path to import sglang modules
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

from sglang.srt.model_loader.ci_weight_validation import (  # noqa: E402
    _validate_diffusion_model,
    validate_cache_with_detailed_reason,
)

# Limits to avoid spending too much time on validation
MAX_VALIDATION_TIME_SECONDS = 300  # Max 5 minutes total


def find_all_hf_snapshots():
    """
    Find all HuggingFace snapshots in cache.

    Returns:
        List of (model_name, snapshot_dir) tuples, sorted by mtime (newest first)
    """
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    hub_dir = os.path.join(hf_home, "hub")

    if not os.path.isdir(hub_dir):
        print(f"HF hub directory not found: {hub_dir}")
        return []

    snapshots = []

    # Pattern: models--org--model/snapshots/hash
    for model_dir in glob.glob(os.path.join(hub_dir, "models--*")):
        # Extract model name from directory (models--org--model -> org/model)
        dir_name = os.path.basename(model_dir)
        if not dir_name.startswith("models--"):
            continue

        # models--meta-llama--Llama-2-7b-hf -> meta-llama/Llama-2-7b-hf
        # Handle multi-part names: models--a--b--c -> a/b-c (join parts 1+ with /)
        parts = dir_name.split("--")
        if len(parts) < 3 or parts[0] != "models":
            # Invalid format, skip
            continue
        # Standard format: models--org--repo -> org/repo
        # Extended format: models--org--repo--extra -> org/repo-extra (join with -)
        model_name = parts[1] + "/" + "-".join(parts[2:])

        snapshots_dir = os.path.join(model_dir, "snapshots")
        if not os.path.isdir(snapshots_dir):
            continue

        # Find all snapshot hashes
        for snapshot_hash_dir in os.listdir(snapshots_dir):
            snapshot_path = os.path.join(snapshots_dir, snapshot_hash_dir)
            if os.path.isdir(snapshot_path):
                try:
                    mtime = os.path.getmtime(snapshot_path)
                    snapshots.append((model_name, snapshot_path, mtime))
                except OSError:
                    continue

    # Sort by mtime (newest first) - prioritize recently used models
    snapshots.sort(key=lambda x: x[2], reverse=True)

    # Return without mtime
    return [(name, path) for name, path, _ in snapshots]


def is_transformers_text_model(snapshot_dir):
    """
    Check if a snapshot is a transformers text model.

    Only excludes (returns False) for models with STRONG evidence of being
    diffusers/generation pipelines. Uses conservative heuristics to avoid
    false negatives on multimodal LLMs with tokenizers.

    Args:
        snapshot_dir: Path to snapshot directory

    Returns:
        True if this looks like a transformers text model, False otherwise (N/A)
    """
    # Check for diffusers pipeline markers (strong evidence)
    diffusers_markers = [
        "model_index.json",  # Diffusers pipeline config
        "scheduler",  # Scheduler directory (diffusers)
    ]
    if any(
        os.path.exists(os.path.join(snapshot_dir, marker))
        for marker in diffusers_markers
    ):
        return False

    config_path = os.path.join(snapshot_dir, "config.json")
    if not os.path.exists(config_path):
        # No config.json - likely not a transformers model
        return False

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        # Check for explicit diffusers/generation model types (conservative keywords)
        model_type = config.get("_class_name") or config.get("model_type")
        if model_type:
            model_type_lower = str(model_type).lower()
            # Only exclude clear diffusion/generation models
            if any(
                keyword in model_type_lower
                for keyword in [
                    "diffusion",
                    "unet",
                    "vae",
                    "controlnet",
                    "stable-diffusion",
                    "latent-diffusion",
                ]
            ):
                return False

        # Check architectures for explicit generation/diffusion classes
        architectures = config.get("architectures", [])
        if architectures:
            arch_str = " ".join(architectures).lower()
            # Conservative: only exclude obvious diffusion/generation architectures
            # Use word boundaries to avoid false positives (e.g., "dit" in "conditional")
            for keyword in [
                "diffusion",
                "unet2d",
                "unet3d",
                "vaedecoder",  # More specific than "vae"
                "vaeencoder",
                "controlnet",
                "autoencoder",
                "ditmodel",  # Diffusion Transformer - use more specific pattern
                "pixart",  # PixArt diffusion model
            ]:
                if keyword in arch_str:
                    return False

        # Check for standalone vision encoder/image processor (no text component)
        # Only if model name explicitly indicates non-text usage
        model_name = config.get("_name_or_path", "").lower()

        if any(
            keyword in model_name
            for keyword in [
                "image-edit-",  # Pure image editing (e.g., Qwen-Image-Edit)
                "-image-editing",
                "dit-",  # DiT generation models
                "pixart-",  # PixArt generation models
            ]
        ):
            # Additional check: does it have tokenizer? If yes, might be multimodal LLM
            has_tokenizer = any(
                os.path.exists(os.path.join(snapshot_dir, fname))
                for fname in ["tokenizer.json", "tokenizer.model", "tiktoken.model"]
            )
            if not has_tokenizer:
                # Image-edit model without tokenizer -> likely pure vision pipeline
                return False

        # Default: assume it's a transformers text/multimodal model
        # Even if it lacks tokenizer, let validation report the actual error
        # (better false positive than false negative for text models)
        return True

    except (json.JSONDecodeError, OSError, KeyError):
        # Can't parse config - assume it's transformers and let validation report failure
        return True


def scan_weight_files(snapshot_dir):
    """
    Scan for weight files in a snapshot.

    Returns:
        List of weight file paths, or empty list if scan fails
    """
    weight_files = []

    # First, look for index files
    index_patterns = ["*.safetensors.index.json", "pytorch_model.bin.index.json"]
    index_files = []
    for pattern in index_patterns:
        index_files.extend(glob.glob(os.path.join(snapshot_dir, pattern)))

    # If we have safetensors index, collect shards from it
    for index_file in index_files:
        if index_file.endswith(".safetensors.index.json"):
            try:
                with open(index_file, "r", encoding="utf-8") as f:
                    index_data = json.load(f)
                weight_map = index_data.get("weight_map", {})
                for weight_file in set(weight_map.values()):
                    weight_path = os.path.join(snapshot_dir, weight_file)
                    if os.path.exists(weight_path):
                        weight_files.append(weight_path)
            except Exception as e:
                print(
                    f"  Warning: Failed to parse index {os.path.basename(index_file)}: {e}"
                )

    # If no index found or no shards from index, do recursive glob
    if not weight_files:
        matched = glob.glob(
            os.path.join(snapshot_dir, "**/*.safetensors"), recursive=True
        )
        MAX_WEIGHT_FILES = 1000
        if len(matched) > MAX_WEIGHT_FILES:
            print(
                f"  Warning: Too many safetensors files ({len(matched)} > {MAX_WEIGHT_FILES})"
            )
            return []

        for f in matched:
            if os.path.exists(f):  # Filter out broken symlinks
                weight_files.append(f)

    return weight_files


def validate_snapshot(model_name, snapshot_dir, weight_files, validated_cache):
    """
    Validate a snapshot and return detailed status.

    Uses in-process cache to avoid duplicate validation within the same run.

    Args:
        model_name: Model identifier
        snapshot_dir: Path to snapshot directory
        weight_files: List of weight files to validate
        validated_cache: Dict to track already-validated snapshots in this run

    Returns:
        Tuple of (result, reason):
        - (True, None) if validation passed
        - (False, reason_str) if validation failed
        - (None, None) if skipped (already validated in this run)
    """
    # Fast path: check in-process cache first
    if snapshot_dir in validated_cache:
        return None, None  # Already validated in this run, skip

    try:
        # Perform validation with detailed reason
        is_complete, reason = validate_cache_with_detailed_reason(
            snapshot_dir=snapshot_dir,
            weight_files=weight_files,
            model_name_or_path=model_name,
        )

        # Cache result to avoid re-validation in this run
        validated_cache[snapshot_dir] = (is_complete, reason)

        return is_complete, reason

    except Exception as e:
        error_msg = f"Validation raised exception: {e}"
        return False, error_msg


def main():
    start_time = time.time()

    print("=" * 70)
    print("CI_OFFLINE: Pre-validating cached HuggingFace models")
    print("=" * 70)
    print(f"Max time: {MAX_VALIDATION_TIME_SECONDS}s")
    print()

    print("Scanning HuggingFace cache for models...")
    snapshots = find_all_hf_snapshots()

    if not snapshots:
        print("No cached models found, skipping validation")
        print("=" * 70)
        return

    print(f"Found {len(snapshots)} snapshot(s) in cache")
    print()

    validated_count = 0
    failed_count = 0
    skipped_count = 0
    processed_count = 0

    # In-process cache to avoid re-validating same snapshot in this run
    validated_cache = {}

    for model_name, snapshot_dir in snapshots:
        # Check time limit
        elapsed = time.time() - start_time
        if elapsed > MAX_VALIDATION_TIME_SECONDS:
            print()
            print(
                f"Time limit reached ({elapsed:.1f}s > {MAX_VALIDATION_TIME_SECONDS}s)"
            )
            print(
                f"Stopping validation, {len(snapshots) - processed_count} snapshots remaining"
            )
            break

        snapshot_hash = os.path.basename(snapshot_dir)
        print(
            f"[{processed_count + 1}/{len(snapshots)}] {model_name} ({snapshot_hash[:8]}...)"
        )
        processed_count += 1

        # Determine model type by checking for model_index.json (diffusers pipeline marker)
        model_index_path = os.path.join(snapshot_dir, "model_index.json")
        is_diffusion_model = os.path.exists(model_index_path)

        if is_diffusion_model:
            # This is a diffusers pipeline - use diffusion validation
            try:
                is_valid, reason = _validate_diffusion_model(snapshot_dir)

                if is_valid:
                    print("  PASS (diffusion) - Cache complete & valid")
                    validated_count += 1
                else:
                    print(f"  FAIL (diffusion) - {reason}")
                    failed_count += 1

            except Exception as e:
                print(f"  FAIL (diffusion) - Validation raised exception: {e}")
                failed_count += 1

            continue

        # Transformers model - use standard validation
        # First check if this looks like a transformers text model
        if not is_transformers_text_model(snapshot_dir):
            # Not a recognized model type, skip
            print(
                "  SKIP (unknown type) - Not a diffusers pipeline or transformers model"
            )
            skipped_count += 1
            continue

        # Scan weight files
        weight_files = scan_weight_files(snapshot_dir)

        if not weight_files:
            print("  SKIP (no weights) - empty or incomplete download")
            skipped_count += 1
            continue

        # Validate
        try:
            result, reason = validate_snapshot(
                model_name, snapshot_dir, weight_files, validated_cache
            )

            if result is True:
                print("  PASS - Cache complete & valid")
                validated_count += 1
            elif result is False:
                # Print detailed failure reason
                if reason:
                    print(f"  FAIL (incomplete) - {reason}")
                else:
                    print("  FAIL (incomplete) - cache validation failed")
                failed_count += 1
            else:  # None (skipped)
                print("  SKIP (already validated in this run)")
                skipped_count += 1

        except Exception as e:
            print(f"  FAIL (error) - Validation raised exception: {e}")
            failed_count += 1

    elapsed_total = time.time() - start_time

    print()
    print("=" * 70)
    print(f"Validation summary (completed in {elapsed_total:.1f}s):")
    print(f"  PASS (complete & valid):      {validated_count}")
    print(f"  FAIL (incomplete/corrupted):  {failed_count}")
    print(f"  SKIP (no weights/duplicate):  {skipped_count}")
    print(f"  Total processed:              {processed_count}/{len(snapshots)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
