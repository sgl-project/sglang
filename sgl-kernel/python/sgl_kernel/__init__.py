import ctypes
import os


if os.path.exists("/usr/local/cuda/targets/x86_64-linux/lib/libcudart.so.12"):
    ctypes.CDLL(
        "/usr/local/cuda/targets/x86_64-linux/lib/libcudart.so.12",
        mode=ctypes.RTLD_GLOBAL,
    )

    apply_rope_with_cos_sin_cache_inplace,
    fused_add_rmsnorm,
    gelu_and_mul,
    gelu_tanh_and_mul,
    gemma_fused_add_rmsnorm,
    gemma_rmsnorm,
    rmsnorm,
    silu_and_mul,
)
    awq_dequantize,
    bmm_fp8,
    cublas_grouped_gemm,
    fp8_blockwise_scaled_mm,
    fp8_scaled_mm,
    int8_scaled_mm,
    sgl_per_tensor_quant_fp8,
    sgl_per_token_group_quant_fp8,
    sgl_per_token_quant_fp8,
)
    min_p_sampling_from_probs,
    top_k_renorm_prob,
    top_k_top_p_sampling_from_probs,
    top_p_renorm_prob,
    top_p_sampling_from_probs,
)
    build_tree_kernel_efficient,
    segment_packbits,
    tree_speculative_sampling_target_only,
    verify_tree_greedy,
)

build_tree_kernel = (
    None  # TODO(ying): remove this after updating the sglang python code.
)