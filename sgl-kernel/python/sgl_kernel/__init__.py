import ctypes
import os
import platform

import torch

SYSTEM_ARCH = platform.machine()

cuda_path = f"/usr/local/cuda/targets/{SYSTEM_ARCH}-linux/lib/libcudart.so.12"
if os.path.exists(cuda_path):
    ctypes.CDLL(cuda_path, mode=ctypes.RTLD_GLOBAL)

from sgl_kernel import common_ops
from sgl_kernel.allreduce import *
from sgl_kernel.attention import (
    cutlass_mla_decode,
    cutlass_mla_get_workspace_size,
    lightning_attention_decode,
    merge_state,
    merge_state_v2,
)
from sgl_kernel.elementwise import (
    apply_rope_with_cos_sin_cache_inplace,
    fused_add_rmsnorm,
    gelu_and_mul,
    gelu_tanh_and_mul,
    gemma_fused_add_rmsnorm,
    gemma_rmsnorm,
    rmsnorm,
    silu_and_mul,
)
from sgl_kernel.gemm import (
    awq_dequantize,
    bmm_fp8,
    cutlass_scaled_fp4_mm,
    fp8_blockwise_scaled_mm,
    fp8_scaled_mm,
    int8_scaled_mm,
    qserve_w4a8_per_chn_gemm,
    qserve_w4a8_per_group_gemm,
    scaled_fp4_experts_quant,
    scaled_fp4_quant,
    sgl_per_tensor_quant_fp8,
    sgl_per_token_group_quant_fp8,
    sgl_per_token_group_quant_int8,
    sgl_per_token_quant_fp8,
    shuffle_rows,
)
from sgl_kernel.grammar import apply_token_bitmask_inplace_cuda
from sgl_kernel.moe import (
    apply_shuffle_mul_sum,
    cutlass_fp4_group_mm,
    ep_moe_post_reorder,
    ep_moe_pre_reorder,
    ep_moe_silu_and_mul,
    fp8_blockwise_scaled_grouped_mm,
    moe_align_block_size,
    moe_fused_gate,
    prepare_moe_input,
    topk_softmax,
)
from sgl_kernel.sampling import (
    min_p_sampling_from_probs,
    top_k_renorm_prob,
    top_k_top_p_sampling_from_probs,
    top_p_renorm_prob,
    top_p_sampling_from_probs,
)
from sgl_kernel.speculative import (
    build_tree_kernel_efficient,
    segment_packbits,
    tree_speculative_sampling_target_only,
    verify_tree_greedy,
)
from sgl_kernel.top_k import fast_topk
from sgl_kernel.version import __version__

build_tree_kernel = (
    None  # TODO(ying): remove this after updating the sglang python code.
)
