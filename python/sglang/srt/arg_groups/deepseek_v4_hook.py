from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sglang.srt.arg_groups.overrides import (
    _deepseek_v4_kv_cache_dtype,
    declare_resolution,
    resolving_view,
    run_post_process_pass,
)
from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_platform

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


def validate_deepseek_v4_mega_moe_token_budget(
    server_args: ServerArgs,
) -> None:
    """Ensure the DSV4 prefill budget fits MegaMoE's per-rank buffer."""
    cfg = resolving_view(server_args)
    mega_moe_enabled = cfg.moe_a2a_backend == "megamoe"
    if not mega_moe_enabled or cfg.disaggregation_mode == "decode":
        # decode node will skip the check because decode bs is not relevant with --chunk-prefill-size
        return

    if cfg.pp_size > 1 and cfg.enable_dynamic_chunking:
        return

    if cfg.chunked_prefill_size is None or cfg.chunked_prefill_size <= 0:
        raise ValueError(
            "DeepSeekV4 with MegaMoE requires chunked prefill to be enabled. "
            "Set --chunked-prefill-size to a positive value; "
            "--chunked-prefill-size=-1 is unsafe because MegaMoE's per-rank "
            "token requirement would not have a strict prefill-forward bound."
        )

    if cfg.enable_prefill_cp:
        token_partition_size = cfg.attn_cp_size
        token_partition_name = "attn_cp_size"
        token_alignment = 1
        local_chunked_prefill_size = (
            cfg.chunked_prefill_size + token_partition_size - 1
        ) // token_partition_size
    elif cfg.enable_dp_attention:
        token_partition_size = cfg.dp_size
        token_partition_name = "dp_size"
        token_alignment = max(
            cfg.tp_size // cfg.dp_size // cfg.attn_cp_size,
            1,
        )
        local_chunked_prefill_size = cfg.chunked_prefill_size // token_partition_size
    else:
        # Pure TP and PP with static chunking are handled here.
        token_partition_size = 1
        token_partition_name = "none"
        # global_num_tokens will ceil_align to attn_tp_size so the validation needs to do alignment as well
        token_alignment = max(
            cfg.tp_size // token_partition_size // cfg.attn_cp_size,
            1,
        )
        local_chunked_prefill_size = cfg.chunked_prefill_size

    if local_chunked_prefill_size <= 0:
        raise ValueError(
            "DeepSeekV4 with MegaMoE requires a positive effective per-rank "
            "chunked prefill size. "
            f"Current values: chunked_prefill_size="
            f"{cfg.chunked_prefill_size}, "
            f"token_partition={token_partition_name}, "
            f"token_partition_size={token_partition_size}."
        )

    required_tokens_per_rank = (
        (local_chunked_prefill_size + token_alignment - 1)
        // token_alignment
        * token_alignment
    )
    max_tokens_per_rank = (
        envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK.get()
    )
    if max_tokens_per_rank < required_tokens_per_rank:
        raise ValueError(
            "DeepSeekV4 with MegaMoE requires "
            "SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK to "
            "cover each rank's effective prefill token budget. "
            f"Current values: chunked_prefill_size="
            f"{cfg.chunked_prefill_size}, "
            f"token_partition={token_partition_name}, "
            f"token_partition_size={token_partition_size}, "
            f"token_alignment={token_alignment}, "
            f"required_per_rank={required_tokens_per_rank}, "
            "SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK="
            f"{max_tokens_per_rank}. Set "
            "SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK to at "
            f"least {required_tokens_per_rank}, or lower "
            "--chunked-prefill-size until the effective per-rank budget fits. "
            "Otherwise MegaMoE falls back to the fused MoE path at runtime."
        )


def apply_deepseek_v4_defaults(server_args: ServerArgs, model_arch: str) -> None:
    """Residual imperative arm of the DeepSeek V4 defaults.

    The attention/page/window/MoE-runner declarations moved to the override
    registry (arg_groups/overrides.py: _deepseek_v4_overrides) and the
    kv-cache dtype default to the resolution pipeline
    (_deepseek_v4_kv_cache_dtype, invoked below at its legacy slot). This
    keeps, at the legacy slot: the ROCm env fill (env-write policy), the
    max_running_requests fill (the speculative hook is a later writer of
    that field) and the validations.
    """
    cfg = resolving_view(server_args)

    # FlashMLA sparse prefill (SGLANG_OPT_FLASHMLA_SPARSE_PREFILL, default on)
    # currently returns incorrect output for DeepSeek-V4-Flash on ROCm/HIP
    # (MI355X), which breaks the disaggregation nightly. Keep the previous
    # (dense prefill) behavior on ROCm until the sparse kernel is validated
    # there;
    if get_platform().is_hip:
        logger.warning(
            "Disabling SGLANG_OPT_FLASHMLA_SPARSE_PREFILL by default on ROCm/HIP "
            f"for {model_arch}; set it explicitly to override."
        )
        envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL.set(False)

    # The kv-cache dtype default moved to the resolution pipeline
    # (arg_groups/overrides.py: _deepseek_v4_kv_cache_dtype), invoked here at
    # its legacy slot.

    run_post_process_pass(server_args, _deepseek_v4_kv_cache_dtype)

    if cfg.max_running_requests is None:
        declare_resolution(
            server_args,
            "apply_deepseek_v4_defaults",
            max_running_requests=256,
        )
        logger.warning(
            f"Setting max_running_requests to {cfg.max_running_requests} for {model_arch}."
        )

    if cfg.speculative_algorithm is not None:
        assert cfg.speculative_algorithm in (
            "EAGLE",
            "DSPARK",
        ), (
            f"Only EAGLE and DSPARK speculative algorithms are supported for {model_arch}"
        )
        if cfg.speculative_algorithm == "EAGLE":
            assert cfg.speculative_eagle_topk == 1, (
                f"Only EAGLE speculative algorithm with topk == 1 is supported for {model_arch}"
            )


def validate_deepseek_v4_cp(server_args: ServerArgs) -> None:
    """Validate DeepSeek V4 context-parallel configuration."""
    cfg = resolving_view(server_args)
    if not cfg.enable_prefill_cp:
        return

    if cfg.cp_strategy != "interleave":
        raise ValueError(
            f"DeepSeekV4 only supports interleave CP strategy, got {cfg.cp_strategy}"
        )

    if get_platform().is_hip or get_platform().is_npu:
        # Protected platform implementations still consume the legacy runtime
        # fields. Generic backends use enable_prefill_cp/cp_strategy directly.
        declare_resolution(
            server_args,
            "validate_deepseek_v4_cp",
            enable_dsa_prefill_context_parallel=True,
        )
        declare_resolution(
            server_args,
            "validate_deepseek_v4_cp",
            enable_prefill_context_parallel=False,
        )
        declare_resolution(
            server_args,
            "validate_deepseek_v4_cp",
            dsa_prefill_cp_mode="round-robin-split",
        )
    declare_resolution(
        server_args,
        "validate_deepseek_v4_cp",
        enable_dp_attention=True,
    )
    declare_resolution(
        server_args,
        "validate_deepseek_v4_cp",
        moe_dense_tp_size=1,
    )
    declare_resolution(
        server_args,
        "validate_deepseek_v4_cp",
        attn_cp_size=cfg.tp_size // cfg.dp_size,
    )
    assert cfg.dp_size == 1, (
        "For round-robin split mode, dp attention is not supported."
    )
    assert cfg.tp_size <= 8, (
        "Context parallel only supports single machine (tp_size <= 8). Cross-machine CP has precision issues."
    )
    supported_a2a_backends = ("none", "deepep", "megamoe", "mori")
    if cfg.moe_a2a_backend not in supported_a2a_backends:
        raise ValueError(
            f"DeepSeekV4 CP supports moe_a2a_backend in {supported_a2a_backends}, "
            f"got {cfg.moe_a2a_backend!r}."
        )
    logger.warning(
        "Disabling SGLANG_OPT_FLASHMLA_SPARSE_PREFILL because DeepSeekV4 "
        "context parallelism is enabled."
    )
    envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL.set(False)
    logger.warning(
        f"Enable Context Parallel for DeepSeekV4, "
        f"dp_size={cfg.dp_size}, moe_dense_tp_size={cfg.moe_dense_tp_size}, "
        f"attn_cp_size={cfg.attn_cp_size}, ep_size={cfg.ep_size}, tp_size={cfg.tp_size}"
    )
