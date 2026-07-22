from sglang.srt.layers.attention.tbo_backend import TboAttnBackend
from sglang.srt.layers.utils.cp_utils import mla_use_prefill_cp
from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    is_in_breakable_cuda_graph,
)
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    is_in_tc_piecewise_cuda_graph,
)
from sglang.srt.models.deepseek_common.attention_forward_methods.forward_methods import (
    AttnForwardMethod,
)
from sglang.srt.models.deepseek_common.utils import _is_hip
from sglang.srt.runtime_context import get_server_args
from sglang.srt.utils import use_intel_amx_backend

MHA_ONE_SHOT_SUPPORTED_BACKENDS = ["fa3", "flashinfer", "flashmla"]


class AttentionBackendRegistry:
    _handlers = {}

    @classmethod
    def register(cls, backend_name, handler_func):
        cls._handlers[backend_name] = handler_func

    @classmethod
    def get_handler(cls, backend_name):
        return cls._handlers.get(backend_name, cls._handlers.get("triton"))


def _dispatch_mla_subtype(attn, forward_batch):
    if _is_hip:
        if (
            attn.rocm_fused_decode_mla
            and forward_batch.forward_mode.is_decode()
            and attn.current_attention_backend == "aiter"
        ):
            return AttnForwardMethod.MLA_FUSED_ROPE_ROCM
        else:
            return AttnForwardMethod.MLA
    else:
        if hasattr(attn, "fused_qkv_a_proj_with_mqa") and use_intel_amx_backend(attn):
            return AttnForwardMethod.MLA_FUSED_ROPE_CPU
        else:
            return AttnForwardMethod.MLA


def handle_attention_ascend(attn, forward_batch):
    if (
        forward_batch.forward_mode.is_extend()
        and not forward_batch.forward_mode.is_target_verify()
        and not forward_batch.forward_mode.is_draft_extend_v2()
    ):
        if hasattr(attn, "use_dsa") and attn.use_dsa:
            return AttnForwardMethod.DSA_NPU
        else:
            return AttnForwardMethod.MHA_NPU
    else:
        if hasattr(attn, "use_dsa") and attn.use_dsa:
            return AttnForwardMethod.DSA_NPU
        else:
            return AttnForwardMethod.MLA_NPU


def _get_sum_extend_prefix_lens(forward_batch):
    return (
        sum(forward_batch.extend_prefix_lens_cpu)
        if forward_batch.extend_prefix_lens_cpu is not None
        else 0
    )


def _support_mha_one_shot(attn, forward_batch, backend_name):
    attn_supported = backend_name in MHA_ONE_SHOT_SUPPORTED_BACKENDS
    sum_seq_lens = (
        sum(forward_batch.seq_lens_cpu) if forward_batch.seq_lens_cpu is not None else 0
    )
    return attn_supported and sum_seq_lens <= forward_batch.get_max_chunk_capacity()


def _handle_attention_backend(attn, forward_batch, backend_name):
    if is_in_tc_piecewise_cuda_graph():
        return AttnForwardMethod.MLA

    # MLA prefill CP forces absorbed MLA regardless of prefix length: the
    # CP path gathers latent KV via rebuild_cp_kv_cache and feeds the
    # backend's absorbed-MLA kernel.
    if mla_use_prefill_cp(forward_batch):
        return _dispatch_mla_subtype(attn, forward_batch)

    sum_extend_prefix_lens = _get_sum_extend_prefix_lens(forward_batch)
    disable_ragged = (
        backend_name in ["flashinfer", "flashmla"]
    ) and attn.flashinfer_mla_disable_ragged

    if (
        not disable_ragged
        and forward_batch.forward_mode.is_extend_without_speculative()
        and (
            (
                sum_extend_prefix_lens >= attn.chunked_prefix_cache_threshold
                and not attn.disable_chunked_prefix_cache
            )
            or sum_extend_prefix_lens == 0
        )
    ):
        if _support_mha_one_shot(attn, forward_batch, backend_name):
            return AttnForwardMethod.MHA_ONE_SHOT
        return AttnForwardMethod.MHA_CHUNKED_KV
    else:
        return _dispatch_mla_subtype(attn, forward_batch)


def handle_attention_flashinfer(attn, forward_batch):
    return _handle_attention_backend(attn, forward_batch, "flashinfer")


def handle_attention_fa3(attn, forward_batch):
    # when deterministic inference is enabled, use MLA
    if get_server_args().enable_deterministic_inference:
        return _dispatch_mla_subtype(attn, forward_batch)
    else:
        return _handle_attention_backend(attn, forward_batch, "fa3")


def handle_attention_flashmla(attn, forward_batch):
    return _handle_attention_backend(attn, forward_batch, "flashmla")


def handle_attention_fa4(attn, forward_batch):
    # TODO(cicirori): use FA4 MHA for DeepSeekV3 for now
    return AttnForwardMethod.MHA_CHUNKED_KV


def handle_attention_trtllm_mla(attn, forward_batch):
    if is_in_tc_piecewise_cuda_graph():
        return AttnForwardMethod.MLA

    sum_extend_prefix_lens = _get_sum_extend_prefix_lens(forward_batch)
    if forward_batch.forward_mode.is_extend_without_speculative() and (
        not attn.disable_chunked_prefix_cache or sum_extend_prefix_lens == 0
    ):
        return AttnForwardMethod.MHA_CHUNKED_KV
    else:
        return _dispatch_mla_subtype(attn, forward_batch)


def handle_attention_tokenspeed_mla(attn, forward_batch):
    # tokenspeed_mla shares the trtllm_mla dispatch pattern: pure prefill goes
    # via MHA chunked KV (TRT-LLM ragged), spec decode / decode goes via MLA.
    return handle_attention_trtllm_mla(attn, forward_batch)


def handle_attention_aiter(attn, forward_batch):
    # During PCG/BCG capture on ROCm, aiter fp8 MLA prefill has no capture
    # kernels; route through the MHA path (radix_attention swaps attn_mqa for
    # its attn_mha companion) so capture/replay use valid head/dim metadata.
    if is_in_tc_piecewise_cuda_graph() or is_in_breakable_cuda_graph():
        return AttnForwardMethod.MHA
    if forward_batch.forward_mode.is_extend_without_speculative():
        return AttnForwardMethod.MHA
    else:
        return AttnForwardMethod.MLA


def handle_attention_dsa(attn, forward_batch):
    """
    Dispatch logic is centralized in DeepseekSparseAttnBackend.set_dsa_prefill_impl and executed
    in init_forward_metadata. Read the decision from backend.use_mha.
    """

    backend = get_attn_backend()
    if isinstance(backend, TboAttnBackend):  # if enable tbo, get primary backend
        backend = backend.primary
    if hasattr(backend, "use_mha") and backend.use_mha:
        return AttnForwardMethod.MHA_ONE_SHOT
    return AttnForwardMethod.MLA


def handle_attention_triton(attn, forward_batch):
    if is_in_tc_piecewise_cuda_graph():
        return AttnForwardMethod.MLA

    # when deterministic inference is enabled, use MLA
    if get_server_args().enable_deterministic_inference:
        return _dispatch_mla_subtype(attn, forward_batch)

    if (
        forward_batch.forward_mode.is_extend_without_speculative()
        and sum(forward_batch.extend_prefix_lens_cpu) == 0
    ):
        return AttnForwardMethod.MHA
    else:
        return _dispatch_mla_subtype(attn, forward_batch)


def handle_attention_intel_xpu(attn, forward_batch):
    return _handle_attention_backend(attn, forward_batch, "intel_xpu")


AttentionBackendRegistry.register("ascend", handle_attention_ascend)
AttentionBackendRegistry.register("flashinfer", handle_attention_flashinfer)
AttentionBackendRegistry.register("fa3", handle_attention_fa3)
AttentionBackendRegistry.register("flashmla", handle_attention_flashmla)
AttentionBackendRegistry.register("fa4", handle_attention_fa4)
AttentionBackendRegistry.register("trtllm_mla", handle_attention_trtllm_mla)
AttentionBackendRegistry.register("tokenspeed_mla", handle_attention_tokenspeed_mla)
AttentionBackendRegistry.register("aiter", handle_attention_aiter)
AttentionBackendRegistry.register("dsa", handle_attention_dsa)
AttentionBackendRegistry.register(
    "nsa", handle_attention_dsa
)  # Deprecated alias; use "dsa"
AttentionBackendRegistry.register("triton", handle_attention_triton)
AttentionBackendRegistry.register("intel_xpu", handle_attention_intel_xpu)
