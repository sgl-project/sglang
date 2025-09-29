ATTENTION_BACKENDS = {}


def register_attention_backend(name):
    def decorator(fn):
        ATTENTION_BACKENDS[name] = fn
        return fn

    return decorator


@register_attention_backend("flashinfer")
def create_flashinfer_backend(runner):
    import torch

    if not runner.use_mla_backend:
        from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend

        # Init streams
        if runner.server_args.speculative_algorithm == "EAGLE":
            if (
                not hasattr(runner, "plan_stream_for_flashinfer")
                or not runner.plan_stream_for_flashinfer
            ):
                runner.plan_stream_for_flashinfer = torch.cuda.Stream()
        return FlashInferAttnBackend(runner)
    else:
        from sglang.srt.layers.attention.flashinfer_mla_backend import (
            FlashInferMLAAttnBackend,
        )

        return FlashInferMLAAttnBackend(runner)


@register_attention_backend("trtllm_mla")
def create_trtllm_mla_backend(runner):
    if not runner.use_mla_backend:
        raise ValueError("trtllm_mla backend can only be used with MLA models.")
    from sglang.srt.layers.attention.trtllm_mla_backend import TRTLLMMLABackend

    return TRTLLMMLABackend(runner)


@register_attention_backend("aiter")
def create_aiter_backend(runner):
    from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend

    return AiterAttnBackend(runner)


@register_attention_backend("wave")
def create_wave_backend(runner):
    from sglang.srt.layers.attention.wave_backend import WaveAttnBackend

    return WaveAttnBackend(runner)


@register_attention_backend("ascend")
def create_ascend_backend(runner):
    from sglang.srt.layers.attention.ascend_backend import AscendAttnBackend

    return AscendAttnBackend(runner)


@register_attention_backend("nsa")
def create_nsa_backend(runner):
    from sglang.srt.layers.attention.nsa_backend import NativeSparseAttnBackend

    return NativeSparseAttnBackend(runner)


@register_attention_backend("triton")
def create_triton_backend(runner):
    assert not runner.model_config.is_encoder_decoder, (
        "Cross attention is not supported in the triton attention backend. "
        "Please use `--attention-backend flashinfer`."
    )
    if runner.server_args.enable_double_sparsity:
        from sglang.srt.layers.attention.double_sparsity_backend import (
            DoubleSparseAttnBackend,
        )

        return DoubleSparseAttnBackend(runner)
    else:
        from sglang.srt.layers.attention.triton_backend import TritonAttnBackend

        return TritonAttnBackend(runner)


@register_attention_backend("torch_native")
def create_torch_native_backend(runner):
    from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend

    return TorchNativeAttnBackend(runner)


@register_attention_backend("flex_attention")
def create_flex_attention_backend(runner):
    from sglang.srt.layers.attention.torch_flex_backend import TorchFlexAttnBackend

    return TorchFlexAttnBackend(runner)


@register_attention_backend("flashmla")
def create_flashmla_backend(runner):
    from sglang.srt.layers.attention.flashmla_backend import FlashMLABackend

    return FlashMLABackend(runner)


@register_attention_backend("fa3")
def create_flashattention_v3_backend(runner):
    import torch

    assert (
        torch.cuda.get_device_capability()[0] == 8 and not runner.use_mla_backend
    ) or torch.cuda.get_device_capability()[0] == 9, (
        "FlashAttention v3 Backend requires SM>=80 and SM<=90. "
        "Please use `--attention-backend flashinfer`."
    )
    from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend

    return FlashAttentionBackend(runner)


@register_attention_backend("fa4")
def create_flashattention_v4_backend(runner):
    assert (
        runner.use_mla_backend
    ), "FlashAttention v4 Support is at an early stage, only MLA model supported now"
    from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend

    return FlashAttentionBackend(runner, fa_impl_ver=4)


@register_attention_backend("cutlass_mla")
def create_cutlass_mla_backend(runner):
    from sglang.srt.layers.attention.cutlass_mla_backend import CutlassMLABackend

    return CutlassMLABackend(runner)


@register_attention_backend("trtllm_mha")
def create_trtllm_mha_backend(runner):
    if runner.use_mla_backend:
        raise ValueError("trtllm_mha backend can only be used with non-MLA models.")
    from sglang.srt.layers.attention.trtllm_mha_backend import TRTLLMHAAttnBackend

    return TRTLLMHAAttnBackend(runner)


@register_attention_backend("intel_amx")
def create_intel_amx_backend(runner):
    from sglang.srt.layers.attention.intel_amx_backend import IntelAMXAttnBackend

    return IntelAMXAttnBackend(runner)


@register_attention_backend("dual_chunk_flash_attn")
def create_dual_chunk_flash_attn_backend(runner):
    from sglang.srt.layers.attention.dual_chunk_flashattention_backend import (
        DualChunkFlashAttentionBackend,
    )

    return DualChunkFlashAttentionBackend(runner)


@register_attention_backend("hybrid_linear_attn")
def create_hybrid_linear_attn_backend(runner):
    assert (
        runner.is_hybrid_gdn
    ), "hybrid_linear_attn backend can only be used with hybrid GDN models."
    from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
        HybridLinearAttnBackend,
        MambaAttnBackend,
    )
    from sglang.srt.utils import is_blackwell, is_npu

    if is_npu():
        from sglang.srt.layers.attention.ascend_backend import AscendAttnBackend

        full_attn_backend = AscendAttnBackend(runner)
    elif is_blackwell():
        from sglang.srt.layers.attention.triton_backend import TritonAttnBackend

        full_attn_backend = TritonAttnBackend(runner)
    else:
        from sglang.srt.layers.attention.flashattention_backend import (
            FlashAttentionBackend,
        )

        full_attn_backend = FlashAttentionBackend(runner)

    linear_attn_backend = MambaAttnBackend(runner)
    full_attn_layers = runner.model_config.hf_config.full_attention_layer_ids

    return HybridLinearAttnBackend(
        full_attn_backend, linear_attn_backend, full_attn_layers
    )
