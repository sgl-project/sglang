import platform
import sys

from sgl_kernel.version import __version__  # noqa: F401

# On macOS only the Metal extension is shipped; skip CUDA op loading and
# re-exports so those symbols are not exposed on Apple Silicon.
if sys.platform == "darwin" and platform.machine() == "arm64":
    from sgl_kernel.metal import *
else:
    import torch
    from sgl_kernel.debug_utils import maybe_wrap_debug_kernel
    from sgl_kernel.load_utils import (
        _load_architecture_specific_ops,
        _preload_cuda_library,
    )

    # Initialize the ops library based on current GPU
    common_ops = _load_architecture_specific_ops()

    # Preload the CUDA library to avoid the issue of libcudart.so.12 not found
    if torch.version.cuda is not None:
        _preload_cuda_library()

    from sgl_kernel.allreduce import *
    from sgl_kernel.attention import (
        cutlass_mla_decode,
        cutlass_mla_get_workspace_size,
        merge_state_v2,
    )
    from sgl_kernel.cutlass_moe import (
        cutlass_w4a8_moe_mm,
        get_cutlass_w4a8_moe_mm_data,
    )
    from sgl_kernel.elementwise import (
        concat_mla_absorb_q,
        concat_mla_k,
        copy_to_gpu_no_ce,
        dsv4_fused_k_norm_rope_flashmla,
        dsv4_fused_q_indexer_rope_hadamard_quant,
        dsv4_fused_q_norm_rope,
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
        dsv3_fused_a_gemm,
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
    from sgl_kernel.infllm_v2 import (
        infllmv2_attn_stage1,
        max_pooling_1d_varlen,
    )
    from sgl_kernel.kvcacheio import (
        copy_all_layer_kv_cache_cpu,
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
        moe_align_block_size,
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
        top_k_renorm_prob,
        top_p_renorm_prob,
    )
    from sgl_kernel.speculative import (
        assign_draft_cache_locs_contiguous_cpu,
        assign_extend_cache_locs_cpu,
        assign_req_to_token_pool_cpu,
        build_draft_decode_metadata_cpu,
        build_tree_kernel_efficient,
        build_tree_kernel_efficient_cpu,
        fill_accept_out_cache_loc_cpu,
        fill_bonus_tokens_cpu,
        reconstruct_indices_from_tree_mask,
        rotate_input_ids_cpu,
        segment_packbits,
        tree_speculative_sampling_target_only,
        verify_tree_greedy,
        verify_tree_greedy_cpu,
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
        from sgl_kernel.top_k import deepseek_v4_topk_transform_512

    if hasattr(torch.version, "musa") and torch.version.musa is not None:
        from sgl_kernel.musa import (
            min_p_sampling_from_probs,
            musa_batched_rotary_embedding_contiguous,
            musa_fused_gemv,
            musa_fused_moe_gemv,
            musa_fused_mul_add,
            musa_rotary_embedding_contiguous,
            top_k_top_p_sampling_from_probs,
        )

    _DEBUG_EXPORT_NAMES = [
        "apply_shuffle_mul_sum",
        "apply_token_bitmask_inplace_cuda",
        "awq_dequantize",
        "build_tree_kernel_efficient",
        "causal_conv1d_fwd",
        "causal_conv1d_update",
        "concat_mla_absorb_q",
        "concat_mla_k",
        "copy_to_gpu_no_ce",
        "cutlass_mla_decode",
        "cutlass_mla_get_workspace_size",
        "dsv3_fused_a_gemm",
        "dsv3_router_gemm",
        "dsv4_fused_k_norm_rope_flashmla",
        "dsv4_fused_q_indexer_rope_hadamard_quant",
        "dsv4_fused_q_norm_rope",
        "es_fp8_blockwise_scaled_grouped_mm",
        "es_sm100_mxfp8_blockscaled_grouped_mm",
        "es_sm100_mxfp8_blockscaled_grouped_quant",
        "fast_topk",
        "fast_topk_transform_fused",
        "fast_topk_transform_ragged_fused",
        "fast_topk_v2",
        "fp8_blockwise_scaled_grouped_mm",
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
        "merge_state_v2",
        "moe_align_block_size",
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
        _DEBUG_EXPORT_NAMES.append("deepseek_v4_topk_transform_512")

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
