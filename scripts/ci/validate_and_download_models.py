#!/usr/bin/env python3
"""
Validate model integrity for CI runners and download if needed.

This script checks HuggingFace cache for model completeness and downloads
missing models. It exits with code 0 if models are present or successfully
downloaded (emitting a warning annotation if repairs were needed), and exits
with code 1 only if download attempts fail.
"""

import os
import re
import shutil
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
    "1-gpu-runner": [
        "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
        "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
        "deepseek-ai/DeepSeek-OCR",
        "google/gemma-3-4b-it",
        "intfloat/e5-mistral-7b-instruct",
        "lmms-lab/llava-onevision-qwen2-0.5b-ov",
        "lmsys/sglang-ci-dsv3-test",
        "lmsys/sglang-EAGLE-llama2-chat-7B",
        "lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B",
        "LxzGordon/URM-LLaMa-3.1-8B",
        "marco/mcdse-2b-v1",
        "meta-llama/Llama-2-7b-chat-hf",
        "meta-llama/Llama-3.2-1B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct",
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "moonshotai/Kimi-VL-A3B-Instruct",
        "nvidia/NVIDIA-Nemotron-Nano-9B-v2",
        "nvidia/NVIDIA-Nemotron-Nano-9B-v2-FP8",
        "openai/gpt-oss-20b",
        "lmsys/gpt-oss-20b-bf16",
        "OpenGVLab/InternVL2_5-2B",
        "Qwen/Qwen1.5-MoE-A2.7B",
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen3-8B",
        "Qwen/Qwen3-Coder-30B-A3B-Instruct",
        "Qwen/Qwen3-Embedding-8B",
        "Qwen/QwQ-32B-AWQ",
        "Qwen/Qwen3-30B-A3B",
        "Qwen/Qwen-Image",
        "Qwen/Qwen-Image-Edit",
        "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2",
        "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    ],
    "2-gpu-runner": [
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "moonshotai/Kimi-Linear-48B-A3B-Instruct",
        "Qwen/Qwen2-57B-A14B-Instruct",
        "Qwen/Qwen2.5-VL-7B-Instruct",
        "Qwen/Qwen3-VL-30B-A3B-Instruct",
        "neuralmagic/Qwen2-72B-Instruct-FP8",
        "zai-org/GLM-4.5-Air-FP8",
    ],
    "8-gpu-h200": [
        "deepseek-ai/DeepSeek-V3-0324",
        "deepseek-ai/DeepSeek-V3.2-Exp",
        "moonshotai/Kimi-K2-Thinking",
    ],
    "8-gpu-b200": ["deepseek-ai/DeepSeek-V3.1", "deepseek-ai/DeepSeek-V3.2-Exp"],
    "4-gpu-b200": ["nvidia/DeepSeek-V3-0324-FP4"],
    "4-gpu-gb200": ["nvidia/DeepSeek-V3-0324-FP4"],
    "4-gpu-h100": [
        "lmsys/sglang-ci-dsv3-test",
        "lmsys/sglang-ci-dsv3-test-NextN",
        "lmsys/gpt-oss-120b-bf16",
    ],
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
    # Use recursive glob to support Diffusers models with weights in subdirectories
    for file_path in model_path.glob("**/*"):
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


def validate_model_shards(model_path: Path) -> Tuple[bool, Optional[str], List[Path]]:
    """
    Validate that all model shards are present and complete.

    Args:
        model_path: Path to model's snapshot directory

    Returns:
        Tuple of (is_valid, error_message, corrupted_files)
        - corrupted_files: List of paths to corrupted shard files that should be removed
    """
    # Pattern for sharded files: model-00001-of-00009.safetensors, pytorch_model-00001-of-00009.bin,
    # or diffusion_pytorch_model-00001-of-00009.safetensors (for Diffusers models)
    # Use word boundary to prevent matching files like tokenizer_model-* or optimizer_model-*
    shard_pattern = re.compile(
        r"(?:^|/)(?:model|pytorch_model|diffusion_pytorch_model)-(\d+)-of-(\d+)\.(safetensors|bin)"
    )

    # Find all shard files recursively (both .safetensors and .bin)
    # This supports both standard models (weights in root) and Diffusers models (weights in subdirs)
    shard_files = list(model_path.glob("**/*-*-of-*.safetensors")) + list(
        model_path.glob("**/*-*-of-*.bin")
    )

    if not shard_files:
        # No sharded files - check for any safetensors or bin files recursively
        # Exclude non-model files like tokenizer, config, optimizer, etc.
        all_safetensors = list(model_path.glob("**/*.safetensors"))
        all_bins = list(model_path.glob("**/*.bin"))

        # Filter out non-model files
        excluded_prefixes = ["tokenizer", "optimizer", "training_", "config"]
        single_files = [
            f
            for f in (all_safetensors or all_bins)
            if not any(f.name.startswith(prefix) for prefix in excluded_prefixes)
            and not f.name.endswith(".index.json")
        ]

        if single_files:
            # Validate all safetensors files, not just the first one
            for model_file in single_files:
                if model_file.suffix == ".safetensors":
                    is_valid, error_msg = validate_safetensors_file(model_file)
                    if not is_valid:
                        return (
                            False,
                            f"Corrupted file {model_file.name}: {error_msg}",
                            [model_file],
                        )
            return True, None, []
        return False, "No model weight files found (safetensors or bin)", []

    # Group shards by subdirectory and total count
    # This handles Diffusers models where different components (transformer/, vae/)
    # have different numbers of shards
    shard_groups = {}
    for shard_file in shard_files:
        # Match against the full path string to get proper path separation
        match = shard_pattern.search(str(shard_file))
        if match:
            shard_num = int(match.group(1))
            total = int(match.group(2))
            parent = shard_file.parent
            key = (str(parent.relative_to(model_path)), total)

            if key not in shard_groups:
                shard_groups[key] = set()
            shard_groups[key].add(shard_num)

    if not shard_groups:
        return False, "Could not determine shard groups from filenames", []

    # Validate each group separately
    for (parent_path, total_shards), found_shards in shard_groups.items():
        expected_shards = set(range(1, total_shards + 1))
        missing_shards = expected_shards - found_shards

        if missing_shards:
            missing_list = sorted(missing_shards)
            location = f" in {parent_path}" if parent_path != "." else ""
            # Missing shards - nothing to remove, let download handle it
            return (
                False,
                f"Missing shards{location}: {missing_list} (expected {total_shards} total)",
                [],
            )

    # Check for index file (look for specific patterns matching the shard prefixes)
    # Standard models: model.safetensors.index.json or pytorch_model.safetensors.index.json
    # Diffusers models: diffusion_pytorch_model.safetensors.index.json in subdirs
    valid_index_patterns = [
        "model.safetensors.index.json",
        "pytorch_model.safetensors.index.json",
        "**/diffusion_pytorch_model.safetensors.index.json",
    ]

    index_files = []
    for pattern in valid_index_patterns:
        index_files.extend(model_path.glob(pattern))

    if not index_files:
        return (
            False,
            "Missing required index file (model/pytorch_model/diffusion_pytorch_model.safetensors.index.json)",
            [],
        )

    # Validate each safetensors shard file for corruption
    print(f"  Validating {len(shard_files)} shard file(s) for corruption...")
    corrupted_files = []
    for shard_file in shard_files:
        if shard_file.suffix == ".safetensors":
            is_valid, error_msg = validate_safetensors_file(shard_file)
            if not is_valid:
                corrupted_files.append(shard_file)
                print(f"    ✗ Corrupted: {shard_file.name} - {error_msg}")

    if corrupted_files:
        return (
            False,
            f"Corrupted shards: {[f.name for f in corrupted_files]}",
            corrupted_files,
        )

    return True, None, []


def validate_model(
    model_id: str, cache_dir: str
) -> Tuple[bool, Optional[str], List[Path]]:
    """
    Validate a model's cache integrity.

    Args:
        model_id: Model identifier
        cache_dir: HuggingFace cache directory

    Returns:
        Tuple of (is_valid, error_message, corrupted_files)
        - corrupted_files: List of paths to corrupted files that should be removed
    """
    print(f"Validating model: {model_id}")

    # Find model in cache
    model_path = get_model_cache_path(model_id, cache_dir)
    if model_path is None:
        return False, "Model not found in cache", []

    print(f"  Found in cache: {model_path}")

    # Check for incomplete files
    incomplete_files = check_incomplete_files(model_path, cache_dir)
    if incomplete_files:
        return (
            False,
            f"Found incomplete download files: {len(incomplete_files)} files",
            [],
        )

    # Validate shards
    is_valid, error_msg, corrupted_files = validate_model_shards(model_path)
    if not is_valid:
        return False, error_msg, corrupted_files

    print(f"  ✓ Model validated successfully")
    return True, None, []


def download_model(model_id: str, cache_dir: str, corrupted_files: List[Path]) -> bool:
    """
    Download a model from HuggingFace.

    Completely removes the model cache directory before downloading to ensure a clean download.

    Args:
        model_id: Model identifier
        cache_dir: HuggingFace cache directory
        corrupted_files: List of specific file paths that are corrupted (unused, kept for compatibility)

    Returns:
        True if download succeeded, False otherwise
    """
    if not HF_HUB_AVAILABLE:
        print(f"ERROR: Cannot download model - huggingface_hub not available")
        return False

    print(f"Downloading model: {model_id}")

    # Completely remove the model directory from cache
    cache_model_name = "models--" + model_id.replace("/", "--")
    model_cache_path = Path(cache_dir) / cache_model_name

    if model_cache_path.exists():
        print(f"  Removing entire model directory: {model_cache_path}")
        try:
            shutil.rmtree(model_cache_path)
            print(f"    ✓ Successfully removed model directory")
        except Exception as e:
            print(f"    ✗ Failed to remove model directory: {e}")
            print(f"    Attempting download anyway...")
    else:
        print(f"  Model directory not found in cache (will download fresh)")

    print(f"  Downloading from HuggingFace (this may take a while for large models)...")

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
        0 if all models are valid, successfully downloaded, or runner doesn't need validation
        1 only if download attempts fail
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
    # Maps model_id -> (error_msg, corrupted_files)
    models_needing_download: Dict[str, Tuple[str, List[Path]]] = {}

    # Validate each required model
    for model_id in required_models:
        is_valid, error_msg, corrupted_files = validate_model(model_id, cache_dir)

        if not is_valid:
            print(f"  ✗ Validation failed: {error_msg}")
            models_needing_download[model_id] = (error_msg, corrupted_files)

    print("-" * 70)

    # If all models are valid, exit successfully
    if not models_needing_download:
        print("✓ All models validated successfully!")
        return 0

    # Models need to be downloaded
    print(f"⚠ Cache validation failed for {len(models_needing_download)} model(s)")
    for model_id, (error_msg, _) in models_needing_download.items():
        print(f"  - {model_id}: {error_msg}")

    print("-" * 70)
    print("Attempting to download missing/corrupted models...")
    print("-" * 70)

    download_failed = False
    for model_id, (error_msg, corrupted_files) in models_needing_download.items():
        if not download_model(model_id, cache_dir, corrupted_files):
            download_failed = True

    print("-" * 70)

    if download_failed:
        print("✗ FAILED: Some models could not be downloaded")
        return 1

    # All downloads succeeded - now validate them again
    print("✓ All models downloaded successfully!")
    print("-" * 70)
    print("Validating downloaded models...")
    print("-" * 70)

    validation_failed = False
    for model_id in models_needing_download.keys():
        is_valid, error_msg, _ = validate_model(model_id, cache_dir)
        if not is_valid:
            print(f"  ✗ Post-download validation failed for {model_id}: {error_msg}")
            validation_failed = True

    print("-" * 70)

    if validation_failed:
        print("✗ FAILED: Some models failed validation after download")
        return 1

    # All validations passed - emit warning but exit successfully
    print("✓ All downloaded models validated successfully!")
    print("⚠ WARNING: Models were missing/corrupted in cache and have been repaired.")
    print(f"  Repaired models: {', '.join(models_needing_download.keys())}")

    # Emit GitHub Actions warning annotation for visibility
    print(
        f"::warning file=scripts/ci/validate_and_download_models.py::"
        f"Cache validation failed for {len(models_needing_download)} model(s). "
        f"Models were re-downloaded and validated successfully. "
        f"This may indicate cache corruption or infrastructure issues."
    )

    return 0


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
