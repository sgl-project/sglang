import ctypes
import glob
import importlib.util
import logging
import os
import shutil
from pathlib import Path
from typing import List

import torch

logger = logging.getLogger(__name__)


def _get_compute_capability():
    """Get the compute capability of the current GPU."""
    if not torch.cuda.is_available():
        return None

    # Get the current device
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)

    # Return as integer (major * 10 + minor)
    return properties.major * 10 + properties.minor


def _filter_compiled_extensions(file_list):
    """Filter and prioritize compiled extensions over Python source files."""
    compiled_extensions = [".so", ".pyd", ".dll"]  # Common compiled extension suffixes
    compiled_files = []
    other_files = []

    for file_path in file_list:
        path = Path(file_path)
        # Check if it's a compiled extension (including complex names like .abi3.so, .cpython-312.so)
        if any(
            str(path).endswith(ext) or ext in str(path) for ext in compiled_extensions
        ):
            compiled_files.append(file_path)
        else:
            other_files.append(file_path)

    # Return compiled files first, then others
    return compiled_files + other_files


def _load_architecture_specific_ops():
    """Load the appropriate common_ops library based on GPU architecture."""
    compute_capability = _get_compute_capability()
    logger.debug(
        f"[sgl_kernel] GPU Detection: compute_capability = {compute_capability}"
    )

    # Get the directory where sgl_kernel is installed
    sgl_kernel_dir = Path(__file__).parent
    logger.debug(f"[sgl_kernel] sgl_kernel directory: {sgl_kernel_dir}")

    # Determine which version to load based on GPU architecture
    if compute_capability == 90:
        ops_subdir = "sm90"
        variant_name = "SM90 (Hopper/H100 with fast math optimization)"
    elif compute_capability is not None:
        ops_subdir = "sm100"
        variant_name = f"SM{compute_capability} (precise math for compatibility)"
    else:
        ops_subdir = "sm100"
        variant_name = "CPU/No GPU detected (using precise math)"

    # Look for the compiled module with any valid extension

    ops_pattern = str(sgl_kernel_dir / ops_subdir / "common_ops.*")
    raw_matching_files = glob.glob(ops_pattern)
    matching_files = _filter_compiled_extensions(raw_matching_files)

    logger.debug(f"[sgl_kernel] Attempting to load {variant_name}")
    logger.debug(f"[sgl_kernel] Looking for library matching pattern: {ops_pattern}")
    logger.debug(f"[sgl_kernel] Found files: {raw_matching_files}")
    logger.debug(f"[sgl_kernel] Prioritized files: {matching_files}")

    previous_import_errors: List[Exception] = []

    # Try to load from the architecture-specific directory
    if matching_files:
        ops_path = Path(matching_files[0])  # Use the first prioritized file
        logger.debug(f"[sgl_kernel] Found architecture-specific library: {ops_path}")
        try:
            # Load the module from specific path using importlib
            spec = importlib.util.spec_from_file_location("common_ops", str(ops_path))
            if spec is None:
                raise ImportError(f"Could not create module spec for {ops_path}")

            common_ops = importlib.util.module_from_spec(spec)
            if spec.loader is None:
                raise ImportError(f"Module spec has no loader for {ops_path}")

            logger.debug(f"[sgl_kernel] Loading module from {ops_path}...")
            spec.loader.exec_module(common_ops)
            logger.debug(f"[sgl_kernel] ✓ Successfully loaded {variant_name}")
            logger.debug(f"[sgl_kernel] ✓ Module file: {common_ops.__file__}")
            return common_ops

        except Exception as e:
            previous_import_errors.append(e)
            logger.debug(
                f"[sgl_kernel] ✗ Failed to load from {ops_path}: {type(e).__name__}: {e}"
            )
            # Continue to fallback
    else:
        logger.debug(
            f"[sgl_kernel] ✗ Architecture-specific library not found matching pattern: {ops_pattern}"
        )

    # Try alternative directory (in case installation structure differs)
    alt_pattern = str(sgl_kernel_dir / "common_ops.*")
    raw_alt_files = glob.glob(alt_pattern)
    alt_matching_files = _filter_compiled_extensions(raw_alt_files)
    logger.debug(f"[sgl_kernel] Attempting fallback: looking for pattern {alt_pattern}")
    logger.debug(f"[sgl_kernel] Found fallback files: {raw_alt_files}")
    logger.debug(f"[sgl_kernel] Prioritized fallback files: {alt_matching_files}")

    if alt_matching_files:
        alt_path = Path(alt_matching_files[0])  # Use the first prioritized file
        logger.debug(f"[sgl_kernel] Found fallback library: {alt_path}")
        try:
            spec = importlib.util.spec_from_file_location("common_ops", str(alt_path))
            if spec is None:
                raise ImportError(f"Could not create module spec for {alt_path}")

            common_ops = importlib.util.module_from_spec(spec)
            if spec.loader is None:
                raise ImportError(f"Module spec has no loader for {alt_path}")

            logger.debug(f"[sgl_kernel] Loading fallback module from {alt_path}...")
            spec.loader.exec_module(common_ops)
            logger.debug(f"[sgl_kernel] ✓ Successfully loaded fallback library")
            logger.debug(f"[sgl_kernel] ✓ Module file: {common_ops.__file__}")
            return common_ops

        except Exception as e:
            previous_import_errors.append(e)
            logger.debug(
                f"[sgl_kernel] ✗ Failed to load fallback from {alt_path}: {type(e).__name__}: {e}"
            )
    else:
        logger.debug(
            f"[sgl_kernel] ✗ Fallback library not found matching pattern: {alt_pattern}"
        )

    # Final attempt: try standard Python import (for backward compatibility)
    logger.debug(
        f"[sgl_kernel] Final attempt: trying standard Python import 'common_ops'"
    )
    try:
        import common_ops

        logger.debug(f"[sgl_kernel] ✓ Successfully imported via standard Python import")
        logger.debug(f"[sgl_kernel] ✓ Module file: {common_ops.__file__}")
        return common_ops
    except ImportError as e:
        previous_import_errors.append(e)
        logger.debug(f"[sgl_kernel] ✗ Standard Python import failed: {e}")

    attempt_error_msg = "\n".join(
        f"- {type(err).__name__}: {err}" for err in previous_import_errors
    )

    # All attempts failed
    error_msg = f"""
[sgl_kernel] CRITICAL: Could not load any common_ops library!

Attempted locations:
1. Architecture-specific pattern: {ops_pattern} - found files: {matching_files}
2. Fallback pattern: {alt_pattern} - found files: {alt_matching_files}
3. Standard Python import: common_ops - failed

GPU Info:
- Compute capability: {compute_capability}
- Expected variant: {variant_name}

Please ensure sgl_kernel is properly installed with:
pip install --upgrade sgl_kernel

Error details from previous import attempts:
{attempt_error_msg}
"""
    logger.debug(error_msg)
    raise ImportError(error_msg)


# copy & modify from torch/utils/cpp_extension.py
def _find_cuda_home():
    """Find the CUDA install path."""
    # Guess #1
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home is None:
        # Guess #2
        nvcc_path = shutil.which("nvcc")
        if nvcc_path is not None:
            cuda_home = os.path.dirname(os.path.dirname(nvcc_path))
        else:
            # Guess #3
            cuda_home = "/usr/local/cuda"
    return cuda_home


def _preload_cuda_library():
    """Preload the CUDA runtime library to help avoid 'libcudart.so.12 not found' issues."""
    cuda_home = Path(_find_cuda_home())

    candidate_dirs = [
        cuda_home / "lib",
        cuda_home / "lib64",
        Path("/usr/lib/x86_64-linux-gnu"),
        Path("/usr/lib/aarch64-linux-gnu"),
        Path("/usr/lib64"),
        Path("/usr/lib"),
    ]

    for base in candidate_dirs:
        candidate = base / "libcudart.so.12"
        if candidate.exists():
            try:
                cuda_runtime_lib = candidate.resolve()
                ctypes.CDLL(str(cuda_runtime_lib), mode=ctypes.RTLD_GLOBAL)
                logger.debug(f"Preloaded CUDA runtime under {cuda_runtime_lib}")
                return
            except Exception as e:
                logger.debug(f"Failed to load {candidate}: {e}")
                continue

    logger.debug("[sgl_kernel] Could not preload CUDA runtime library")
