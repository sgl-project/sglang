import torch
from sgl_kernel.debug_utils import maybe_wrap_debug_kernel
from sgl_kernel.load_utils import _load_architecture_specific_ops, _preload_cuda_library

# Initialize the ops library based on current GPU
common_ops = _load_architecture_specific_ops()

# Preload the CUDA library to avoid the issue of libcudart.so.12 not found
if torch.version.cuda is not None:
    _preload_cuda_library()


from sgl_kernel.allreduce import *
from sgl_kernel.attention import (
    cutlass_mla_decode,
    cutlass_mla_get_workspace_size,
    merge_state,
    merge_state_v2,
)
from sgl_kernel.cutlass_moe import cutlass_w4a8_moe_mm, get_cutlass_w4a8_moe_mm_data
from sgl_kernel.elementwise import (
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
    rotary_embedding,
    silu_and_mul,
)
from sgl_kernel.expert_specialization import (
    es_fp8_blockwise_scaled_grouped_mm,
    es_sm100_mxfp8_blockscaled_grouped_mm,
    es_sm100_mxfp8_blockscaled_grouped_quant,
)
from sgl_kernel.gemm import (
    awq_dequantize,
    bmm_fp8,
    dsv3_fused_a_gemm,
    dsv3_router_gemm,
    fp8_blockwise_scaled_mm,
    fp8_scaled_mm,
    gptq_gemm,
    gptq_shuffle,
    int8_scaled_mm,
    qserve_w4a8_per_chn_gemm,
    qserve_w4a8_per_group_gemm,
    sgl_per_token_group_quant_8bit,
    sgl_per_token_group_quant_fp8,
    sgl_per_token_group_quant_int8,
    sgl_per_token_quant_fp8,
    shuffle_rows,
)
from sgl_kernel.grammar import apply_token_bitmask_inplace_cuda
from sgl_kernel.kvcacheio import (
    transfer_kv_all_layer,
    transfer_kv_all_layer_mla,
    transfer_kv_per_layer,
    transfer_kv_per_layer_mla,
)
from sgl_kernel.mamba import (
    causal_conv1d_fn_cpu,
    causal_conv1d_fwd,
    causal_conv1d_update,
    causal_conv1d_update_cpu,
    chunk_gated_delta_rule_cpu,
)
from sgl_kernel.memory import weak_ref_tensor
from sgl_kernel.moe import (
    apply_shuffle_mul_sum,
    fp8_blockwise_scaled_grouped_mm,
    fused_qk_norm_rope,
    kimi_k2_moe_fused_gate,
    moe_align_block_size,
    moe_fused_gate,
    moe_sum,
    moe_sum_reduce,
    prepare_moe_input,
    topk_sigmoid,
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
    top_k_mask_logits,
    top_k_renorm_prob,
    top_p_renorm_prob,
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


_DEBUG_EXPORT_NAMES = [
    "apply_shuffle_mul_sum",
    "apply_token_bitmask_inplace_cuda",
    "awq_dequantize",
    "bmm_fp8",
    "build_tree_kernel_efficient",
    "causal_conv1d_fwd",
    "causal_conv1d_update",
    "concat_mla_absorb_q",
    "concat_mla_k",
    "copy_to_gpu_no_ce",
    "cutlass_mla_decode",
    "cutlass_mla_get_workspace_size",
    "downcast_fp8",
    "dsv3_fused_a_gemm",
    "dsv3_router_gemm",
    "es_fp8_blockwise_scaled_grouped_mm",
    "es_sm100_mxfp8_blockscaled_grouped_mm",
    "es_sm100_mxfp8_blockscaled_grouped_quant",
    "fast_topk",
    "fast_topk_transform_fused",
    "fast_topk_transform_ragged_fused",
    "fast_topk_v2",
    "fp8_blockwise_scaled_grouped_mm",
    "fp8_blockwise_scaled_mm",
    "fp8_scaled_mm",
    "fused_add_rmsnorm",
    "fused_qk_norm_rope",
    "gelu_and_mul",
    "gelu_tanh_and_mul",
    "gemma_fused_add_rmsnorm",
    "gemma_rmsnorm",
    "gptq_gemm",
    "gptq_shuffle",
    "int8_scaled_mm",
    "kimi_k2_moe_fused_gate",
    "merge_state",
    "merge_state_v2",
    "moe_align_block_size",
    "moe_fused_gate",
    "moe_sum",
    "moe_sum_reduce",
    "prepare_moe_input",
    "qserve_w4a8_per_chn_gemm",
    "qserve_w4a8_per_group_gemm",
    "reconstruct_indices_from_tree_mask",
    "rmsnorm",
    "rotary_embedding",
    "segment_packbits",
    "sgl_per_token_group_quant_8bit",
    "sgl_per_token_group_quant_fp8",
    "sgl_per_token_group_quant_int8",
    "sgl_per_token_quant_fp8",
    "shuffle_rows",
    "silu_and_mul",
    "top_k_mask_logits",
    "top_k_renorm_prob",
    "top_p_renorm_prob",
    "topk_sigmoid",
    "topk_softmax",
    "transfer_kv_all_layer",
    "transfer_kv_all_layer_mla",
    "transfer_kv_per_layer",
    "transfer_kv_per_layer_mla",
    "tree_speculative_sampling_target_only",
    "verify_tree_greedy",
    "weak_ref_tensor",
]

if torch.version.hip is not None:
    _DEBUG_EXPORT_NAMES.append("gelu_quick")

for _name in _DEBUG_EXPORT_NAMES:
    if _name in globals():
        globals()[_name] = maybe_wrap_debug_kernel(
            globals()[_name], f"sgl_kernel.{_name}"
        )

del _name
del _DEBUG_EXPORT_NAMES


def create_greenctx_stream_by_value(*args, **kwargs):
    from sgl_kernel.spatial import create_greenctx_stream_by_value as _impl

    return _impl(*args, **kwargs)


def get_sm_available(*args, **kwargs):
    from sgl_kernel.spatial import get_sm_available as _impl

    return _impl(*args, **kwargs)
