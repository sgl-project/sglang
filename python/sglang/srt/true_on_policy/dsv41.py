"""Server-start checks for DeepSeek-V4.1-Flash's TOP forward.

The program is the release checkpoint on Hopper: 32x32 UE8M0 FP8 dense weights, MXFP4 experts cast losslessly to
128x128 FP8 at load (SGLANG_DSV4_FP4_DEQUANT) and run by DS4's FP8 expert program behind DeepEP, an FP8 E4M3 KV
cache on the dsv4 backend at attention TP1, previous-sublayer mHC, Engram, and the V4.1 attention program. Every
other serving path is refused here by name.
"""


def validate_dsv41(config, quant_config):
    """Refuse every DS4.1 serving path outside the qualified TOP program."""
    from sglang.srt.environ import envs
    from sglang.srt.layers.moe.utils import get_deepep_mode, is_tbo_enabled
    from sglang.srt.model_executor.cuda_graph_config import Backend
    from sglang.srt.runtime_context import (
        get_disagg,
        get_exec,
        get_memory,
        get_model,
        get_parallel,
        get_schedule,
        get_server_args,
        get_serving,
        get_spec,
        process_model_config,
    )
    from sglang.srt.true_on_policy import fp8
    from sglang.srt.true_on_policy.moe_ep import validate_ep_runtime

    if config.model_type != "deepseek_v41":
        raise NotImplementedError(
            f"[top] DeepSeek-V4.1 TOP covers deepseek_v41 only, got {config.model_type!r}"
        )
    if config.vision_n_layers > 0 and get_disagg().language_model_only:
        # Without its tower the engine routes every row by the text gate; the trainer, reading the checkpoint,
        # would route them through the VL-biased gate (serves_images).
        raise NotImplementedError(
            "[top] DeepSeek-V4.1 serves its vision checkpoint with the tower: drop --language-model-only"
        )
    if not fp8.enabled_b32():
        raise NotImplementedError(
            "[top] DeepSeek-V4.1 requires its 32x32 UE8M0 block-FP8 checkpoint"
        )
    fp8.validate_quant_config(quant_config)
    if not getattr(quant_config, "is_fp4_experts", False):
        raise NotImplementedError(
            "[top] DeepSeek-V4.1 requires the release's MXFP4 experts"
        )
    if not envs.SGLANG_DSV4_FP4_DEQUANT.get():
        raise NotImplementedError(
            "[top] DeepSeek-V4.1 runs its MXFP4 experts cast to 128x128 FP8 on Hopper (DeepGEMM's FP4 GEMM is "
            "SM100/SM120-only); set SGLANG_DSV4_FP4_DEQUANT=1"
        )
    args, kernel, exec_ = get_server_args(), get_exec().kernel, get_exec()
    if get_model().kv_cache_dtype != "fp8_e4m3":
        raise NotImplementedError("[top] DeepSeek-V4.1 requires an FP8 E4M3 KV cache")
    if kernel.attention_backend != "dsv4" or kernel.dsv4_attn_backend not in (
        "auto",
        "flashmla",
    ):
        raise NotImplementedError(
            "[top] DeepSeek-V4.1 requires the dsv4 FlashMLA backend, got "
            f"{kernel.attention_backend!r}/{kernel.dsv4_attn_backend!r}"
        )
    validate_ep_runtime()
    parallel = get_parallel()
    if (
        parallel.attn_tp_size != 1
        or parallel.pp_size != 1
        or parallel.attn_cp_size != 1
    ):
        raise NotImplementedError(
            "[top] DeepSeek-V4.1 requires attention TP1, PP1 and no context parallelism"
        )
    if config.n_routed_experts % parallel.moe_ep_size:
        raise NotImplementedError("[top] DeepSeek-V4.1 EP must divide the expert count")
    if get_deepep_mode().value not in ("auto", "normal"):
        raise NotImplementedError("[top] DeepSeek-V4.1 prefill requires normal DeepEP")
    if is_tbo_enabled():
        raise NotImplementedError("[top] DeepSeek-V4.1 forward does not admit TBO")
    if exec_.graph.cuda_graph_config.prefill.backend != Backend.DISABLED:
        raise NotImplementedError(
            "[top] DeepSeek-V4.1 prefill runs eager (the prefill graph takes the captured indexer); "
            "pass --cuda-graph-backend-prefill disabled"
        )
    validate_dsv41_radix(
        compress_ratios=config.compress_ratios,
        page_size=get_schedule().page_size,
        images=serves_images(config),
        streaming_session=get_serving().enable_streaming_session,
        memory=get_memory(),
        disagg=get_disagg(),
    )
    chunk = get_schedule().chunked_prefill_size
    if chunk is not None and 0 < chunk < process_model_config().context_len:
        raise NotImplementedError(
            "[top] DeepSeek-V4.1 chunked prefill splits are not qualified; the per-rank chunk "
            f"({chunk}) must cover the context length ({process_model_config().context_len}) or be -1"
        )
    for feature, enabled in (
        (
            "decoder SWA bounded replay",
            exec_.features.enable_decoder_swa_bounded_replay,
        ),
        (
            "encoder SWA bounded replay",
            exec_.features.enable_encoder_swa_bounded_replay,
        ),
        ("mixed prefill/decode chunks", get_schedule().enable_mixed_chunk),
        ("HiSparse", get_memory().enable_hisparse),
        ("shared-expert fusion", exec_.moe.enforce_shared_experts_fusion),
        ("FlashInfer mHC", envs.SGLANG_OPT_USE_FLASHINFER_MHC.get()),
        ("the BF16 KV quant source", envs.SGLANG_DSV4_USE_BF16_KV_QUANT_SOURCE.get()),
        ("the unified KV layout", _unified_kv()),
    ):
        if enabled:
            raise NotImplementedError(f"[top] DeepSeek-V4.1 does not admit {feature}")
    from sglang.srt.true_on_policy.dsv41_dspark import validate_dspark

    validate_dspark(config, get_spec(), get_model().model_path)
    if (
        config.hidden_act != "silu"
        or config.scoring_func != "sqrtsoftplus"
        or not config.norm_topk_prob
        or not config.hc_pre_from_prev_sublayer
        or config.hc_mult != 4
        or config.hidden_size != 5120
        or getattr(config, "num_hash_layers", 0)
    ):
        raise NotImplementedError(
            "[top] DeepSeek-V4.1 requires SwiGLU, sqrtsoftplus score routing (no hash layers) and 4-stream "
            "previous-sublayer mHC at hidden 5120"
        )


def validate_dsv41_radix(
    *, compress_ratios, page_size, images, streaming_session, memory, disagg
):
    """Admit radix reuse from the engine's own device tree at page boundaries; refuse every other prefix source.

    Hits are whole pages (the tree keys are page-aligned). A page of whole compression blocks leaves no pending pair in
    the ratio-2 ring at a hit, so the new request starts with nothing to carry; ratio 1 and the indexer keep nothing
    across forwards beyond the paged KV and index keys, which a hit shares.
    """
    if memory.disable_radix_cache:
        return
    block = max(compress_ratios)
    if page_size % block:
        raise NotImplementedError(
            f"[top] DeepSeek-V4.1 radix reuse needs pages of whole {block}-token compression blocks (a hit inside a "
            f"block resumes a pending pair the new request's ring does not hold), got --page-size {page_size}"
        )
    for source, enabled in (
        ("HiCache", memory.enable_hierarchical_cache),
        ("LMCache", memory.enable_lmcache),
        ("FlexKV", memory.enable_flexkv),
        ("a custom radix-cache backend", memory.radix_cache_backend is not None),
        ("an external cache linker", memory.enable_unified_cache_external_linker),
        ("session radix caching", memory.enable_session_radix_cache),
        ("streaming sessions", streaming_session),
        ("PD disaggregation", disagg.disaggregation_mode != "null"),
        (
            "host-pool retraction",
            disagg.disaggregation_decode_retraction_backup == "host_pool",
        ),
        ("the vision tower", images),
    ):
        if enabled:
            raise NotImplementedError(
                f"[top] DeepSeek-V4.1 radix reuse with {source} is NOT_QUALIFIED; pass --disable-radix-cache"
            )


def serves_images(config) -> bool:
    """Whether a V4.1 engine built from ``config`` loads the vision tower; such an engine routes every row through
    the VL-biased gate (DeepseekV2MoE.top_vision_routing), so the trainer's router must too."""
    return (
        config.model_type == "deepseek_v41"
        and config.vision_n_layers > 0
        and not getattr(config, "language_model_only", False)
    )


def _unified_kv():
    from sglang.srt.layers.attention.dsv4.unified_kv_gate import (
        is_unified_kv_triton,
    )

    return is_unified_kv_triton()
