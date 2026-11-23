"""Server-start checks and MoE bindings for DeepSeek-V4-Flash's TOP forward.

The program is block-FP8 (128x128) with FP8 experts and FP8 KV, the dsv4
FlashMLA backend and DeepEP/DeepGEMM experts at attention TP1. V4.1 is a
separate program (true_on_policy.dsv41).
"""


def fp8_wo_a_program() -> bool:
    """Whether serving runs wo_a as the FP8 GEMM; otherwise it dequantizes an FP8 wo_a to BF16 at load.

    Mirrors the resolution of SGLANG_OPT_FP8_WO_A_GEMM in arg_groups/serving_hook.py (on unless disabled, explicit
    opt-in on SM90, DeepGEMM support), so the trainer, which runs no serving hooks, reads the same decision.
    """
    from sglang.srt.environ import envs
    from sglang.srt.layers import deep_gemm_wrapper
    from sglang.srt.runtime_context import get_platform

    flag = envs.SGLANG_OPT_FP8_WO_A_GEMM
    if not flag.get():
        return False
    return bool(
        deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0
        or (
            deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
            and get_platform().is_sm90
            and flag.is_set()
        )
    )


def validate_top_dsv4(config, quant_config):
    from sglang.srt.true_on_policy.config import is_true_on_policy_enabled

    if not is_true_on_policy_enabled():
        return
    if config.model_type == "deepseek_v41":
        from sglang.srt.true_on_policy.dsv41 import validate_dsv41

        validate_dsv41(config, quant_config)
        return
    validate_dsv4(config, quant_config)


def validate_dsv4(config, quant_config):
    """Refuse every DS4 serving path outside the qualified TOP program."""
    from sglang.srt.environ import envs
    from sglang.srt.layers.moe.utils import (
        get_deepep_mode,
        is_sbo_enabled,
        is_tbo_enabled,
    )
    from sglang.srt.runtime_context import (
        get_exec,
        get_model,
        get_parallel,
        get_server_args,
    )
    from sglang.srt.true_on_policy import fp8
    from sglang.srt.true_on_policy.moe_ep import validate_ep_runtime

    if config.model_type != "deepseek_v4":
        raise NotImplementedError(
            f"[top] DeepSeek-V4 TOP covers deepseek_v4 only, got {config.model_type!r}"
        )
    if not fp8.enabled():
        raise NotImplementedError(
            "[top] DeepSeek-V4 requires a 128x128 block-FP8 checkpoint"
        )
    fp8.validate_quant_config(quant_config)
    fp8.validate_device()
    if (
        getattr(quant_config, "is_fp4_experts", False)
        or envs.SGLANG_DSV4_FP4_EXPERTS.get()
    ):
        raise NotImplementedError(
            "[top] DeepSeek-V4 requires FP8 experts; set SGLANG_DSV4_FP4_EXPERTS=0"
        )
    # Resolved values: get_server_args() keeps only what the user passed.
    args, kernel = get_server_args(), get_exec().kernel
    if get_model().kv_cache_dtype != "fp8_e4m3":
        raise NotImplementedError("[top] DeepSeek-V4 requires an FP8 E4M3 KV cache")
    if kernel.attention_backend != "dsv4" or kernel.dsv4_attn_backend not in (
        "auto",
        "flashmla",
    ):
        raise NotImplementedError(
            "[top] DeepSeek-V4 requires the dsv4 FlashMLA backend, got "
            f"{kernel.attention_backend!r}/{kernel.dsv4_attn_backend!r}"
        )
    if kernel.dsv4_prefill_backend not in ("auto", "flashmla_sparse"):
        raise NotImplementedError("[top] DeepSeek-V4 requires BF16 sparse prefill")
    # DeepEP transport, DeepGEMM experts and one expert domain; the op registry admits the EP degree.
    validate_ep_runtime()
    parallel = get_parallel()
    if parallel.attn_tp_size != 1 or parallel.pp_size != 1:
        raise NotImplementedError("[top] DeepSeek-V4 requires attention TP1 and PP1")
    if config.n_routed_experts % parallel.moe_ep_size:
        raise NotImplementedError("[top] DeepSeek-V4 EP must divide the expert count")
    if get_deepep_mode().value not in ("auto", "normal"):
        raise NotImplementedError("[top] DeepSeek-V4 prefill requires normal DeepEP")
    if is_tbo_enabled() or is_sbo_enabled():
        raise NotImplementedError(
            "[top] DeepSeek-V4 forward does not admit TBO or SBO; TOP's slot transport "
            "runs the shared expert and the combine itself"
        )
    _validate_kv_layout()
    for feature, enabled in (
        ("HiSparse", args.enable_hisparse),
        ("the FP4 indexer", kernel.enable_deepseek_v4_fp4_indexer),
        ("PD disaggregation", args.disaggregation_mode != "null"),
        ("HiCache", args.enable_hierarchical_cache),
        ("a ragged verify mode other than static", _ragged_verify()),
        # Fused, the shared expert is an FP32-weighted combine slot, not the BF16 add after the combine the trainer runs.
        ("shared-expert fusion", args.enforce_shared_experts_fusion),
        ("online c128 compression", envs.SGLANG_OPT_USE_ONLINE_COMPRESS.get()),
        ("FP4 expert dequant", envs.SGLANG_DSV4_FP4_DEQUANT.get()),
        # The trainer calls the TileLang mhc_pre/mhc_post and the pinned-split DeepGEMM prenorm (required below).
        ("FlashInfer mHC", envs.SGLANG_OPT_USE_FLASHINFER_MHC.get()),
        ("the BF16 KV quant source", envs.SGLANG_DSV4_USE_BF16_KV_QUANT_SOURCE.get()),
        ("the non-paged indexer", envs.SGLANG_OPT_DSV4_NONPAGED_INDEXER.get()),
        ("the TileLang indexer", envs.SGLANG_OPT_USE_TILELANG_INDEXER.get()),
        ("the aiter indexer", envs.SGLANG_OPT_USE_AITER_INDEXER.get()),
        ("torch indexer logits", envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.get()),
        (
            "the unified KV layout (its c4 state ring skips radix page writes)",
            _unified_kv(),
        ),
    ):
        if enabled:
            raise NotImplementedError(f"[top] DeepSeek-V4 does not admit {feature}")
    from sglang.srt.true_on_policy.mtp import validate_mtp

    # (3, 1, 4) is the cookbook low-latency recipe for DS4-Flash.
    validate_mtp(config, args, family="DeepSeek-V4", extra_recipes=((3, 1, 4),))
    from sglang.srt.layers.deep_gemm_wrapper.configurer import ENABLE_JIT_DEEPGEMM

    if not (
        envs.SGLANG_OPT_USE_TILELANG_MHC_PRE.get()
        and envs.SGLANG_OPT_USE_TILELANG_MHC_POST.get()
        and envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM.get()
        and ENABLE_JIT_DEEPGEMM
    ):
        raise NotImplementedError(
            "[top] DeepSeek-V4 mHC requires TileLang pre/post and the DeepGEMM prenorm"
        )
    if (
        config.hidden_act != "silu"
        or config.scoring_func != "sqrtsoftplus"
        or not config.norm_topk_prob
        or config.hc_pre_from_prev_sublayer
    ):
        raise NotImplementedError(
            "[top] DeepSeek-V4 requires SwiGLU, sqrtsoftplus routing and same-sublayer mHC"
        )
    # DeepseekV2MoE reads num_hash_layers with a 0 default; a missing key would score-route every layer.
    if getattr(config, "num_hash_layers", 0) <= 0:
        raise NotImplementedError(
            "[top] DeepSeek-V4 requires num_hash_layers in the config"
        )


def validate_top_dsv4_topk(topk, *, is_deepseek_v4, is_deepseek_v41=False):
    """Admit DS4's native per-row gates as the TOP router; the combine applies routed scaling.

    DeepSeek-V4.1's router (op dsv41_router) is the same gate with sqrtsoftplus evaluated through log1p."""
    from sglang.srt.true_on_policy.config import is_true_on_policy_enabled

    if not is_deepseek_v4 or not is_true_on_policy_enabled():
        return
    from sglang.srt.layers.moe.hash_topk import HashTopK
    from sglang.srt.true_on_policy.moe import refuse_routed_experts_capture

    refuse_routed_experts_capture("DeepSeek-V4.1" if is_deepseek_v41 else "DeepSeek-V4")

    if isinstance(topk, HashTopK):
        if (
            topk.apply_routed_scaling_factor_on_output
            or topk.num_shared_topk_slots
            or topk.enable_waterfill
            or topk.score_func != "sqrtsoftplus"
        ):
            raise NotImplementedError(
                "[top] DeepSeek-V4 hash routing requires unscaled sqrtsoftplus weights"
            )
        return
    cfg = topk.topk_config
    if (
        cfg.scoring_func != "sqrtsoftplus"
        or cfg.correction_bias is None
        or not cfg.renormalize
        or cfg.use_grouped_topk
        or cfg.num_shared_topk_slots
        or cfg.custom_routing_function is not None
        or cfg.apply_routed_scaling_factor_on_output
        or cfg.sqrtsoftplus_log1p != is_deepseek_v41
        or cfg.torch_native
    ):
        raise NotImplementedError(
            "[top] DeepSeek-V4 requires ungrouped sqrtsoftplus/bias routing with "
            "unscaled FP32 weights and no fused shared slots"
        )


# Arbitrary fixed split of the mHC pre-norm GEMM; any constant makes it token-count invariant.
MHC_PRE_NUM_SPLITS = 16


def top_mhc_pre_num_splits():
    """The pinned mHC pre-norm split count under TOP, None otherwise.

    None also where no server args are published (kernel tests, benchmarks, offline scripts).
    """
    from sglang.srt.runtime_context import get_server_args
    from sglang.srt.true_on_policy.config import is_true_on_policy_enabled

    try:
        get_server_args()
    except ValueError:
        return None
    return MHC_PRE_NUM_SPLITS if is_true_on_policy_enabled() else None


def _validate_kv_layout():
    """The qualified program reads the 584-byte V4 pools; V4.1 quantizes SWA and compressed KV differently."""
    from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import select_dsv4_kv_layout

    layout, compressed = select_dsv4_kv_layout()
    if layout is not KVLayout.V4 or compressed is not None:
        raise NotImplementedError(
            f"[top] DeepSeek-V4 requires the V4 KV layout, got {layout.value} with "
            f"compressed option {compressed!r}; set SGLANG_DSV4_KV_LAYOUT=v4 and leave "
            "SGLANG_DSV4_COMPRESSED_KV_LAYOUT unset"
        )


def _ragged_verify():
    from sglang.srt.speculative.ragged_verify import (
        RaggedVerifyMode,
        read_ragged_verify_mode,
    )

    return read_ragged_verify_mode() is not RaggedVerifyMode.STATIC


def _unified_kv():
    from sglang.srt.layers.attention.dsv4.unified_kv_gate import (
        is_unified_kv_triton,
    )

    return is_unified_kv_triton()
