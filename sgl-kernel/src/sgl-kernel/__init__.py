import torch
if torch.version.hip is not None:
    # TODO (hubert): check this for custom_reduce
    # init_custom_ar
    # all_reduce_reg
    # all_reduce_unreg
    # dispose
    # meta_size
    # register_buffer
    # get_graph_buffer_ipc_meta
    # register_graph_buffers
    # allocate_meta_buffer
    # get_meta_buffer_ipc_handle
    from sgl_kernel.ops import (
        all_reduce_reg,        # TODO (hubert), ok
        all_reduce_unreg,      # TODO (hubert), ok
        allocate_meta_buffer,  # TODO (hubert), ok
        apply_rope_with_cos_sin_cache_inplace,
        bmm_fp8,
        dispose,              # TODO (hubert), ok
        fp8_scaled_mm,
        fused_add_rmsnorm,
        gelu_and_mul,
        gelu_tanh_and_mul,
        gemma_fused_add_rmsnorm,
        gemma_rmsnorm,
        get_device_bdf,             # TODO (hubert), ok
        get_graph_buffer_ipc_meta,  # TODO (hubert), ok
        get_meta_buffer_ipc_handle, # TODO (hubert), ok
        init_custom_ar,             # TODO (hubert), ok
        int8_scaled_mm,
        lightning_attention_decode,
        meta_size,                  # TODO (hubert), ok
        min_p_sampling_from_probs,
        moe_align_block_size,
        register_buffer,            # TODO (hubert), ok
        register_graph_buffers,     # TODO (hubert), ok
        rmsnorm,
        sampling_scaling_penalties,
        silu_and_mul,
        top_k_renorm_prob,
        top_k_top_p_sampling_from_probs,
        top_p_renorm_prob,
    )

    __all__ = [
        "all_reduce_reg",       # TODO (hubert), ok
        "all_reduce_unreg",     # TODO (hubert), ok
        "allocate_meta_buffer", # TODO (hubert), ok
        "apply_rope_with_cos_sin_cache_inplace",
        "bmm_fp8",
        "dispose",              # TODO (hubert), ok
        "fp8_scaled_mm",
        "fused_add_rmsnorm",
        "gelu_and_mul",
        "gelu_tanh_and_mul",
        "gemma_fused_add_rmsnorm",
        "gemma_rmsnorm",
        "get_device_bdf",             # TODO (hubert), ok
        "get_graph_buffer_ipc_meta",  # TODO (hubert), ok
        "get_meta_buffer_ipc_handle", # TODO (hubert), ok
        "init_custom_ar",             # TODO (hubert), ok
        "int8_scaled_mm",
        "lightning_attention_decode",
        "meta_size",                  # TODO (hubert), ok
        "min_p_sampling_from_probs",
        "moe_align_block_size",
        "register_buffer",            # TODO (hubert), ok
        "register_graph_buffers",     # TODO (hubert), ok
        "rmsnorm",
        "sampling_scaling_penalties",
        "silu_and_mul",
        "top_k_renorm_prob",
        "top_k_top_p_sampling_from_probs",
        "top_p_renorm_prob",
    ]
else:
    # get_graph_buffer_ipc_meta, # TODO (hubert)
    # init_custom_reduce,        # TODO (hubert)
    # get_graph_buffer_ipc_meta, # TODO (hubert)
    # init_custom_reduce,        # TODO (hubert)
    # register_graph_buffers,    # TODO (hubert)
    from sgl_kernel.ops import (
        apply_rope_with_cos_sin_cache_inplace,
        bmm_fp8,
        custom_dispose, # TODO (hubert)
        custom_reduce,  # TODO (hubert)
        fp8_scaled_mm,
        fused_add_rmsnorm,
        gelu_and_mul,
        gelu_tanh_and_mul,
        gemma_fused_add_rmsnorm,
        gemma_rmsnorm,
        get_graph_buffer_ipc_meta, # TODO (hubert)
        init_custom_reduce,        # TODO (hubert)
        int8_scaled_mm,
        lightning_attention_decode,
        min_p_sampling_from_probs,
        moe_align_block_size,
        register_graph_buffers,    # TODO (hubert)
        rmsnorm,
        sampling_scaling_penalties,
        silu_and_mul,
        top_k_renorm_prob,
        top_k_top_p_sampling_from_probs,
        top_p_renorm_prob,
    )

    __all__ = [
        "apply_rope_with_cos_sin_cache_inplace",
        "bmm_fp8",
        "custom_dispose",    # TODO (hubert)
        "custom_reduce",     # TODO (hubert)
        "fp8_scaled_mm",
        "fused_add_rmsnorm",
        "gelu_and_mul",
        "gelu_tanh_and_mul",
        "gemma_fused_add_rmsnorm",
        "gemma_rmsnorm",
        "get_graph_buffer_ipc_meta",    # TODO (hubert)
        "init_custom_reduce",           # TODO (hubert)
        "int8_scaled_mm",
        "lightning_attention_decode",
        "min_p_sampling_from_probs",
        "moe_align_block_size",
        "register_graph_buffers",       # TODO (hubert)
        "rmsnorm",
        "sampling_scaling_penalties",
        "silu_and_mul",
        "top_k_renorm_prob",
        "top_k_top_p_sampling_from_probs",
        "top_p_renorm_prob",
    ]
