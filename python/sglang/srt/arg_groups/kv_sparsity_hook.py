"""Validation for HBM-resident post-hoc KV-cache sparsity."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


def validate_kv_cache_sparsity(server_args: ServerArgs) -> None:
    if not server_args.enable_kv_cache_sparsity:
        if server_args.kv_cache_sparsity_config is not None:
            raise ValueError(
                "--kv-cache-sparsity-config requires --enable-kv-cache-sparsity"
            )
        return

    from sglang.srt.arg_groups.overrides import attention_backends_of, resolved_view
    from sglang.srt.configs.hybrid_arch import mambaish_config
    from sglang.srt.configs.model_config import AttentionArch
    from sglang.srt.environ import envs
    from sglang.srt.mem_cache.sparsity.factory import parse_kv_sparsity_config
    from sglang.srt.model_executor.cuda_graph_config import Backend, Phase
    from sglang.srt.server_args import is_cuda, is_hip

    view = resolved_view(server_args)

    if view.enable_hisparse:
        raise ValueError(
            "HBM-resident KV sparsity and --enable-hisparse are separate placement "
            "modes and cannot be enabled together"
        )
    if is_hip() or not is_cuda():
        raise ValueError("The initial KV sparsity runtime requires NVIDIA CUDA FA3")
    if envs.SGLANG_USE_HND_KVCACHE.get():
        raise ValueError("KV sparsity currently requires the NHD KV-cache layout")

    config = parse_kv_sparsity_config(server_args)
    if config.policy != "streaming_llm":
        raise ValueError(
            "PR 1 supports only policy='streaming_llm'; Quest is provided by PR 2"
        )
    if config.backend != "fa3":
        raise ValueError("The initial KV sparsity runtime supports backend='fa3'")

    prefill_backend, decode_backend = attention_backends_of(view)
    allowed_backends = {"fa3", "flashattention"}
    if (
        prefill_backend not in allowed_backends
        or decode_backend not in allowed_backends
    ):
        raise ValueError(
            "KV sparsity requires FA3 for prefill and decode, but got "
            f"prefill={prefill_backend!r}, decode={decode_backend!r}"
        )

    unsupported = (
        (view.speculative_algorithm is not None, "speculative decoding"),
        (view.dllm_algorithm is not None, "diffusion LLM inference"),
        (view.enable_pdmux, "PD multiplexing"),
        (view.disaggregation_mode != "null", "PD disaggregation"),
        (view.enable_two_batch_overlap, "two-batch overlap"),
        (view.enable_mixed_chunk, "mixed chunked prefill"),
        (view.enable_torch_compile, "torch.compile"),
        (view.enable_dp_attention, "DP attention"),
        (view.enable_prefill_cp, "prefill context parallelism"),
        (view.attn_cp_size > 1, "attention context parallelism"),
        (view.dcp_size > 1, "decode context parallelism"),
    )
    for enabled, label in unsupported:
        if enabled:
            raise ValueError(f"KV sparsity does not yet support {label}")

    model_config = view.get_model_config()
    if model_config.attention_arch != AttentionArch.MHA:
        raise ValueError("KV sparsity currently supports standard MHA/GQA models only")
    if model_config.is_encoder_decoder or model_config.is_multimodal:
        raise ValueError("KV sparsity currently supports decoder-only text models")
    if not model_config.is_generation:
        raise ValueError("KV sparsity is supported only for generation models")
    if model_config.num_attention_layers != model_config.num_hidden_layers:
        raise ValueError("KV sparsity requires one attention layer per hidden layer")

    sliding_window_size = model_config.sliding_window_size
    has_sliding_window = isinstance(sliding_window_size, (int, float)) and (
        sliding_window_size > -1
    )
    if (
        model_config.is_hybrid_swa
        or has_sliding_window
        or model_config.attention_chunk_size is not None
        or mambaish_config(model_config) is not None
    ):
        raise ValueError("KV sparsity does not yet support local or hybrid attention")
    num_kv_shared_layers = (
        getattr(model_config.hf_text_config, "num_kv_shared_layers", 0) or 0
    )
    if num_kv_shared_layers > 0:
        raise ValueError("KV sparsity does not yet support cross-layer KV sharing")

    locked = getattr(server_args, "_cuda_graph_config_locked", set())
    for phase, phase_config in (
        (Phase.DECODE, server_args.cuda_graph_config.decode),
        (Phase.PREFILL, server_args.cuda_graph_config.prefill),
    ):
        if (phase, "backend") in locked and phase_config.backend != Backend.DISABLED:
            raise ValueError(
                f"KV sparsity PR 1 requires {phase} CUDA graph to be disabled"
            )
        phase_config.backend = Backend.DISABLED
    logger.info("Enabled HBM-resident StreamingLLM visibility with FA3 in eager mode")
