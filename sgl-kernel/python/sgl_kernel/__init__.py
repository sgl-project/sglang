import torch
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
    rotary_embedding,
    silu_and_mul,
)
from sgl_kernel.expert_specialization import (
    es_fp8_blockwise_scaled_grouped_mm,
    es_sm100_mxfp8_blockscaled_grouped_mm,
    es_sm100_mxfp8_blockscaled_grouped_quant,
)
from sgl_kernel.fused_moe import moe_wna16_marlin_gemm
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
from sgl_kernel.memory import set_kv_buffer_kernel, weak_ref_tensor
from sgl_kernel.moe import (
    apply_shuffle_mul_sum,
    cutlass_fp4_group_mm,
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
