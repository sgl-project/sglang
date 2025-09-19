import ctypes
import os
import platform
import shutil
from pathlib import Path

import torch


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

from sgl_kernel import common_ops
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
    sgl_per_token_group_quant_fp8,
    sgl_per_token_group_quant_int8,
    sgl_per_token_quant_fp8,
    shuffle_rows,
    silu_and_mul_scaled_fp4_grouped_quant,
)
from sgl_kernel.grammar import apply_token_bitmask_inplace_cuda
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
    moe_sum_reduce,
    prepare_moe_input,
    topk_softmax,
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
from sgl_kernel.top_k import fast_topk
from sgl_kernel.version import __version__

if torch.version.hip is not None:
    from sgl_kernel.elementwise import gelu_quick


def create_greenctx_stream_by_value(*args, **kwargs):
    from sgl_kernel.spatial import create_greenctx_stream_by_value as _impl

    return _impl(*args, **kwargs)


def get_sm_available(*args, **kwargs):
    from sgl_kernel.spatial import get_sm_available as _impl

    return _impl(*args, **kwargs)
