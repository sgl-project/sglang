"""Admission for the Qwen GDN TOP program; the shared no-split rule (``contracts``) owns the engine layout."""

from sglang.srt.true_on_policy.gdn_chunked import (
    CHUNKED,
    require_chunked_gdn_kernels,
    top_gdn_mode,
    validate_gdn_chunked_contract,
)

# Admitted text architectures. Qwen3.5/3.6/3.8 sizes differ only in layer, head and expert counts,
# so every size is admitted by the per-head program below; Qwen3.8-Flash-Next (qwen4_exp_text: hidden 2560,
# 16 key / 48 value GDN heads of 128) runs the same GDN layer.
GDN_TOP_ARCHITECTURES = frozenset(
    {"qwen3_5_text", "qwen3_5_moe_text", "qwen4_exp_text"}
)
GDN_TOP_MOE_ARCHITECTURES = frozenset({"qwen3_5_moe_text", "qwen4_exp_text"})
# Full-attention layers run QSA (true_on_policy.qsa) instead of FA4, so there is no FA decode split to bind.
GDN_TOP_QSA_ARCHITECTURES = frozenset({"qwen4_exp_text"})


def _validate_per_head_program(config):
    if (
        config.linear_key_head_dim != 128
        or config.linear_value_head_dim != 128
        or config.linear_conv_kernel_dim != 4
        or config.linear_num_value_heads % config.linear_num_key_heads
        or config.head_dim != 256
        or config.hidden_act != "silu"
    ):
        raise ValueError(
            "GDN TOP requires the Qwen3.5 per-head program: 128-wide GDN heads in whole value groups, "
            "a width-4 convolution, 256-wide attention heads and SiLU"
        )


def _validate_decode_kv_split(model_type, *, page_size, context_len):
    """The decode split's limits, checked at launch rather than at graph capture or the first long decode."""
    from sglang.srt.true_on_policy.attention import (
        MAX_DECODE_KV_PIECES,
        decode_kv_split,
    )

    split = decode_kv_split(model_type)
    if not split:
        return
    if split % page_size:
        raise ValueError(
            f"GDN TOP decode KV split {split} must be a multiple of the page size {page_size}"
        )
    if -(-context_len // split) > MAX_DECODE_KV_PIECES:
        raise ValueError(
            f"GDN TOP decode KV split {split} cuts a {context_len}-token context into more than "
            f"{MAX_DECODE_KV_PIECES} pieces: raise SGLANG_TOP_DECODE_KV_SPLIT or lower --context-length"
        )


def validate_gdn_contract(args):
    from sglang.srt.arg_groups.model_override_base import model_config_of

    model_config = model_config_of(args)
    config = getattr(model_config.hf_config, "text_config", model_config.hf_config)
    if config.model_type not in GDN_TOP_ARCHITECTURES:
        raise ValueError(
            f"GDN TOP admits {sorted(GDN_TOP_ARCHITECTURES)}, got {config.model_type!r}"
        )
    _validate_per_head_program(config)
    if config.model_type in GDN_TOP_QSA_ARCHITECTURES and (
        args.qsa_indexer_dtype not in ("auto", "bfloat16")
    ):
        # Serving would score blocks from the rounded compressed keys; the trainer scores the BF16 keys.
        raise ValueError(
            f"GDN TOP scores QSA blocks from BF16 compressed keys; got --qsa-indexer-dtype {args.qsa_indexer_dtype}"
        )
    if top_gdn_mode() != CHUNKED:
        raise ValueError(
            "GDN TOP runs the chunked GDN program: the engine selects it at launch (SGLANG_TOP_GDN_MODE=chunked); "
            "a process that starts schedulers itself sets it"
        )
    require_chunked_gdn_kernels()
    if args.speculative_algorithm is not None and (
        args.speculative_algorithm != "EAGLE" or (args.speculative_eagle_topk or 1) != 1
    ):
        # Chain MTP/EAGLE only: verify runs the root and its drafts as one multi-row step of the row engine, and the
        # race draws every emitted token independently of the draft, so speculation changes speed, not tokens.
        raise ValueError("GDN TOP admits chain EAGLE/MTP speculation (top-k 1) only")
    validate_gdn_chunked_contract(args, config)
    if args.page_size % 64 or args.mamba_track_interval % 64:
        raise ValueError("GDN TOP requires 64-aligned pages and Mamba tracking")
    _validate_decode_kv_split(
        config.model_type,
        page_size=args.page_size,
        context_len=model_config.context_len,
    )
    if args.enable_int8_mamba_checkpoint:
        raise ValueError("GDN TOP requires lossless FP32 prefix checkpoints")
    # ReplaySSM advances the state one token at a time, unlike the chunk prefill the row engine follows.
    if args.enable_linear_replayssm:
        raise ValueError("GDN TOP decodes through its own row engine, not ReplaySSM")
    if args.enable_page_major_kv_layout:
        raise ValueError("GDN TOP requires contiguous state pools")
    # -1 disables chunked prefill: a whole prompt is one prefill segment.
    if args.chunked_prefill_size is not None and args.chunked_prefill_size > 0:
        if args.chunked_prefill_size % 64:
            raise ValueError("GDN TOP requires 64-aligned chunked prefill")
    # Every GDN state must be this engine's own chunk-prefill or row-engine state: its own prefill/decode or a
    # radix node it wrote.
    if (
        args.enable_streaming_session
        or args.disaggregation_mode != "null"
        or args.disaggregation_decode_retraction_backup == "host_pool"
        or args.enable_hierarchical_cache
        or args.enable_unified_cache_external_linker
    ):
        raise ValueError(
            "GDN TOP builds every state itself: streaming sessions, disaggregation, host state "
            "offload and external cache linkers are not supported"
        )
    # External caches copy mamba states in from outside the engine.
    if args.enable_lmcache or args.enable_flexkv or args.radix_cache_backend:
        raise ValueError(
            "GDN TOP builds every state itself: LMCache, FlexKV and custom "
            "radix-cache backends are not supported"
        )
    if (
        not args.disable_radix_cache
        and args.mamba_radix_cache_strategy != "extra_buffer"
    ):
        raise ValueError(
            "GDN TOP radix reuse requires --mamba-radix-cache-strategy extra_buffer"
        )
    # The row engine plans decode and extend rows per forward; mixed batches are not supported.
    if args.enable_mixed_chunk:
        raise ValueError("GDN TOP forbids mixed prefill/decode batches")
    # The row engine replays FLA's Triton chunk program on the FP32 state; CuteDSL and FlashInfer GDN are other programs.
    if args.linear_attn_backend != "triton" or args.mamba_ssm_dtype != "float32":
        raise ValueError("GDN TOP requires Triton GDN and FP32 recurrent state")
    for name in ("linear_attn_prefill_backend", "linear_attn_decode_backend"):
        if getattr(args, name) not in (None, "triton"):
            raise ValueError(f"GDN TOP requires Triton GDN kernels; got {name}")
    if (
        config.model_type in GDN_TOP_MOE_ARCHITECTURES
        and not args.disable_shared_experts_fusion
    ):
        raise ValueError("GDN TOP requires separate gated shared experts")


def validate_qwen4_exp_top(config, *, quant_config):
    """Qwen3.8-Flash-Next under TOP: the GDN program plus the model's hyper-connection, PLE and QSA bindings.
    The checkpoint's MTP layer (the EAGLE draft) is built from the same classes and runs the same programs."""
    if quant_config is not None:
        raise NotImplementedError(
            "[top] Qwen4-Exp TOP admits the BF16 checkpoint only (no quantized checkpoint)"
        )
    if (
        config.hc_count != 4
        or config.hc_lowrank % 16
        or (config.hc_count * config.hidden_size) % 128
        or config.ple_embed_dim % 128
    ):
        raise NotImplementedError(
            "[top] Qwen4-Exp TOP requires 4 branches and BI-GEMM-aligned PLE/HC widths"
        )
