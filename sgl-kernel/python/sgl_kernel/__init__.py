import ctypes
import logging
import os
import platform
import shutil
from pathlib import Path

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
    import importlib.util
    import sys
    from pathlib import Path

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
    import glob

    ops_pattern = str(sgl_kernel_dir / ops_subdir / "common_ops.*")
    raw_matching_files = glob.glob(ops_pattern)
    matching_files = _filter_compiled_extensions(raw_matching_files)

    logger.debug(f"[sgl_kernel] Attempting to load {variant_name}")
    logger.debug(f"[sgl_kernel] Looking for library matching pattern: {ops_pattern}")
    logger.debug(f"[sgl_kernel] Found files: {raw_matching_files}")
    logger.debug(f"[sgl_kernel] Prioritized files: {matching_files}")

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
        logger.debug(f"[sgl_kernel] ✗ Standard Python import failed: {e}")

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
"""
    logger.debug(error_msg)
    raise ImportError(error_msg)


# Initialize the ops library based on current GPU
logger.debug("[sgl_kernel] Initializing architecture-specific operator library...")
common_ops = _load_architecture_specific_ops()
logger.debug("[sgl_kernel] ✓ Operator library initialization complete")


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


if torch.version.cuda is not None:
    cuda_home = Path(_find_cuda_home())

    if (cuda_home / "lib").is_dir():
        cuda_path = cuda_home / "lib"
    elif (cuda_home / "lib64").is_dir():
        cuda_path = cuda_home / "lib64"
    else:
        # Search for 'libcudart.so.12' in subdirectories
        for path in cuda_home.rglob("libcudart.so.12"):
            cuda_path = path.parent
            break
        else:
            raise RuntimeError("Could not find CUDA lib directory.")

    cuda_include = (cuda_path / "libcudart.so.12").resolve()
    if cuda_include.exists():
        ctypes.CDLL(str(cuda_include), mode=ctypes.RTLD_GLOBAL)

from sgl_kernel.allreduce import *
from sgl_kernel.attention import (
    cutlass_mla_decode,
    cutlass_mla_get_workspace_size,
    lightning_attention_decode,
    merge_state,
    merge_state_v2,
)
from sgl_kernel.cutlass_moe import cutlass_w4a8_moe_mm, get_cutlass_w4a8_moe_mm_data
from sgl_kernel.elementwise import (
    FusedSetKVBufferArg,
    apply_rope_with_cos_sin_cache_inplace,
    concat_mla_absorb_q,
    concat_mla_k,
    copy_to_gpu_no_ce,
    downcast_fp8,
    fused_add_rmsnorm,
    gelu_and_mul,
    gelu_tanh_and_mul,
    gemma_fused_add_rmsnorm,
    gemma_rmsnorm,
    rmsnorm,
    silu_and_mul,
)
from sgl_kernel.expert_specialization import es_fp8_blockwise_scaled_grouped_mm
from sgl_kernel.fused_moe import fused_marlin_moe
from sgl_kernel.gemm import (
    awq_dequantize,
    bmm_fp8,
    cutlass_scaled_fp4_mm,
    dsv3_fused_a_gemm,
    dsv3_router_gemm,
    fp8_blockwise_scaled_mm,
    fp8_scaled_mm,
    gptq_gemm,
    gptq_marlin_gemm,
    gptq_shuffle,
    int8_scaled_mm,
    qserve_w4a8_per_chn_gemm,
    qserve_w4a8_per_group_gemm,
    scaled_fp4_experts_quant,
    scaled_fp4_grouped_quant,
    scaled_fp4_quant,
    sgl_per_tensor_quant_fp8,
    sgl_per_token_group_quant_8bit,
    sgl_per_token_group_quant_fp8,
    sgl_per_token_group_quant_int8,
    sgl_per_token_quant_fp8,
    shuffle_rows,
    silu_and_mul_scaled_fp4_grouped_quant,
)
from sgl_kernel.grammar import apply_token_bitmask_inplace_cuda
from sgl_kernel.hadamard import (
    hadamard_transform,
    hadamard_transform_12n,
    hadamard_transform_20n,
    hadamard_transform_28n,
    hadamard_transform_40n,
)
from sgl_kernel.kvcacheio import (
    transfer_kv_all_layer,
    transfer_kv_all_layer_mla,
    transfer_kv_per_layer,
    transfer_kv_per_layer_mla,
)
from sgl_kernel.mamba import causal_conv1d_fwd, causal_conv1d_update
from sgl_kernel.marlin import (
    awq_marlin_moe_repack,
    awq_marlin_repack,
    gptq_marlin_repack,
)
from sgl_kernel.memory import set_kv_buffer_kernel
from sgl_kernel.moe import (
    apply_shuffle_mul_sum,
    cutlass_fp4_group_mm,
    fp8_blockwise_scaled_grouped_mm,
    moe_align_block_size,
    moe_fused_gate,
    moe_sum,
    moe_sum_reduce,
    prepare_moe_input,
    topk_softmax,
)
from sgl_kernel.quantization import (
    ggml_dequantize,
    ggml_moe_a8,
    ggml_moe_a8_vec,
    ggml_moe_get_block_size,
    ggml_mul_mat_a8,
    ggml_mul_mat_vec_a8,
)
from sgl_kernel.sampling import (
    min_p_sampling_from_probs,
    top_k_mask_logits,
    top_k_renorm_prob,
    top_k_top_p_sampling_from_logits,
    top_k_top_p_sampling_from_probs,
    top_p_renorm_prob,
    top_p_sampling_from_probs,
)
from sgl_kernel.speculative import (
    build_tree_kernel_efficient,
    reconstruct_indices_from_tree_mask,
    segment_packbits,
    tree_speculative_sampling_target_only,
    verify_tree_greedy,
)
from sgl_kernel.top_k import (
    fast_topk,
    fast_topk_transform_fused,
    fast_topk_transform_ragged_fused,
    fast_topk_v2,
)
from sgl_kernel.version import __version__

if torch.version.hip is not None:
    from sgl_kernel.elementwise import gelu_quick


def create_greenctx_stream_by_value(*args, **kwargs):
    from sgl_kernel.spatial import create_greenctx_stream_by_value as _impl

    return _impl(*args, **kwargs)


def get_sm_available(*args, **kwargs):
    from sgl_kernel.spatial import get_sm_available as _impl

    return _impl(*args, **kwargs)
