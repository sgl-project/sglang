import ctypes
import os

if os.path.exists("/usr/local/cuda/targets/x86_64-linux/lib/libcudart.so.12"):
    ctypes.CDLL(
        "/usr/local/cuda/targets/x86_64-linux/lib/libcudart.so.12",
        mode=ctypes.RTLD_GLOBAL,
    )

from sgl_kernel.ops import (
    apply_rope_with_cos_sin_cache_inplace,
    bmm_fp8,
    build_tree_kernel,
    build_tree_kernel_efficient,
    cublas_grouped_gemm,
    custom_dispose,
    custom_reduce,
    fp8_blockwise_scaled_mm,
    fp8_scaled_mm,
    fused_add_rmsnorm,
    gelu_and_mul,
    gelu_tanh_and_mul,
    gemma_fused_add_rmsnorm,
    gemma_rmsnorm,
    get_graph_buffer_ipc_meta,
    init_custom_reduce,
    int8_scaled_mm,
    lightning_attention_decode,
    min_p_sampling_from_probs,
    moe_align_block_size,
    register_graph_buffers,
    rmsnorm,
    sampling_scaling_penalties,
    sgl_per_token_group_quant_fp8,
    silu_and_mul,
    top_k_renorm_prob,
    top_k_top_p_sampling_from_probs,
    top_p_renorm_prob,
    tree_speculative_sampling_target_only,
)

from .version import __version__

__all__ = [
    "apply_rope_with_cos_sin_cache_inplace",
    "bmm_fp8",
    "cublas_grouped_gemm",
    "custom_dispose",
    "custom_reduce",
    "fp8_blockwise_scaled_mm",
    "fp8_scaled_mm",
    "fused_add_rmsnorm",
    "gelu_and_mul",
    "gelu_tanh_and_mul",
    "gemma_fused_add_rmsnorm",
    "gemma_rmsnorm",
    "get_graph_buffer_ipc_meta",
    "init_custom_reduce",
    "int8_scaled_mm",
    "lightning_attention_decode",
    "min_p_sampling_from_probs",
    "moe_align_block_size",
    "register_graph_buffers",
    "rmsnorm",
    "sampling_scaling_penalties",
    "silu_and_mul",
    "top_k_renorm_prob",
    "top_k_top_p_sampling_from_probs",
    "top_p_renorm_prob",
    "tree_speculative_sampling_target_only",
    "build_tree_kernel_efficient",
    "build_tree_kernel",
    "sgl_per_token_group_quant_fp8",
]
