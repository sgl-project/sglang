from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


def validate_deepseek_v4_mega_moe_token_budget(
    server_args: ServerArgs,
) -> None:
    """Ensure the DSV4 prefill budget fits MegaMoE's per-rank buffer."""
    mega_moe_enabled = (
        server_args.moe_a2a_backend == "megamoe"
        or envs.SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE.get()
    )
    if not mega_moe_enabled or server_args.disaggregation_mode == "decode":
        # decode node will skip the check because decode bs is not relevant with --chunk-prefill-size
        return

    if server_args.pp_size > 1 and server_args.enable_dynamic_chunking:
        return

    if (
        server_args.chunked_prefill_size is None
        or server_args.chunked_prefill_size <= 0
    ):
        raise ValueError(
            "DeepSeekV4 with MegaMoE requires chunked prefill to be enabled. "
            "Set --chunked-prefill-size to a positive value; "
            "--chunked-prefill-size=-1 is unsafe because MegaMoE's per-rank "
            "token requirement would not have a strict prefill-forward bound."
        )

    if server_args.enable_prefill_cp:
        token_partition_size = server_args.attn_cp_size
        token_partition_name = "attn_cp_size"
        token_alignment = 1
        local_chunked_prefill_size = (
            server_args.chunked_prefill_size + token_partition_size - 1
        ) // token_partition_size
    elif server_args.enable_dp_attention:
        token_partition_size = server_args.dp_size
        token_partition_name = "dp_size"
        token_alignment = max(
            server_args.tp_size // server_args.dp_size // server_args.attn_cp_size,
            1,
        )
        local_chunked_prefill_size = (
            server_args.chunked_prefill_size // token_partition_size
        )
    else:
        # Pure TP and PP with static chunking are handled here.
        token_partition_size = 1
        token_partition_name = "none"
        # global_num_tokens will ceil_align to attn_tp_size so the validation needs to do alignment as well
        token_alignment = max(
            server_args.tp_size // token_partition_size // server_args.attn_cp_size,
            1,
        )
        local_chunked_prefill_size = server_args.chunked_prefill_size

    if local_chunked_prefill_size <= 0:
        raise ValueError(
            "DeepSeekV4 with MegaMoE requires a positive effective per-rank "
            "chunked prefill size. "
            f"Current values: chunked_prefill_size="
            f"{server_args.chunked_prefill_size}, "
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
            f"{server_args.chunked_prefill_size}, "
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
    from sglang.srt.utils import is_hip

    # FlashMLA sparse prefill (SGLANG_OPT_FLASHMLA_SPARSE_PREFILL, default on)
    # currently returns incorrect output for DeepSeek-V4-Flash on ROCm/HIP
    # (MI355X), which breaks the disaggregation nightly. Keep the previous
    # (dense prefill) behavior on ROCm until the sparse kernel is validated
    # there;
    if is_hip():
        logger.warning(
            "Disabling SGLANG_OPT_FLASHMLA_SPARSE_PREFILL by default on ROCm/HIP "
            f"for {model_arch}; set it explicitly to override."
        )
        envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL.set(False)

    # The kv-cache dtype default moved to the resolution pipeline
    # (arg_groups/overrides.py: _deepseek_v4_kv_cache_dtype), invoked here at
    # its legacy slot.
    from sglang.srt.arg_groups.overrides import (
        _deepseek_v4_kv_cache_dtype,
        run_post_process_pass,
    )

    run_post_process_pass(server_args, _deepseek_v4_kv_cache_dtype)

    if server_args.max_running_requests is None:
        server_args.max_running_requests = 256
        logger.warning(
            f"Setting max_running_requests to {server_args.max_running_requests} for {model_arch}."
        )

    if server_args.speculative_algorithm is not None:
        assert server_args.speculative_algorithm in (
            "EAGLE",
            "DSPARK",
        ), f"Only EAGLE and DSPARK speculative algorithms are supported for {model_arch}"
        if server_args.speculative_algorithm == "EAGLE":
            assert (
                server_args.speculative_eagle_topk == 1
            ), f"Only EAGLE speculative algorithm with topk == 1 is supported for {model_arch}"


def validate_deepseek_v4_cp(server_args: ServerArgs) -> None:
    """Validate DeepSeek V4 context-parallel configuration."""
    if not server_args.enable_prefill_cp:
        return

    if server_args.cp_strategy != "interleave":
        raise ValueError(
            "DeepSeekV4 only supports interleave CP strategy, "
            f"got {server_args.cp_strategy}"
        )

    server_args.enable_dsa_prefill_context_parallel = True
    server_args.dsa_prefill_cp_mode = "round-robin-split"
    server_args.enable_dp_attention = True
    server_args.moe_dense_tp_size = 1
    server_args.attn_cp_size = server_args.tp_size // server_args.dp_size
    assert (
        server_args.dp_size == 1
    ), "For round-robin split mode, dp attention is not supported."
    assert (
        server_args.tp_size <= 8
    ), "Context parallel only supports single machine (tp_size <= 8). Cross-machine CP has precision issues."
    if server_args.moe_a2a_backend not in ("none", "deepep", "megamoe"):
        raise ValueError(
            "DeepSeekV4 CP supports moe_a2a_backend in "
            "('none', 'deepep', 'megamoe'), "
            f"got {server_args.moe_a2a_backend!r}."
        )
    logger.warning(
        "Disabling SGLANG_OPT_FLASHMLA_SPARSE_PREFILL because DeepSeekV4 "
        "context parallelism is enabled."
    )
    envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL.set(False)
    logger.warning(
        f"Enable Context Parallel for DeepSeekV4, "
        f"dp_size={server_args.dp_size}, moe_dense_tp_size={server_args.moe_dense_tp_size}, "
        f"attn_cp_size={server_args.attn_cp_size}, ep_size={server_args.ep_size}, tp_size={server_args.tp_size}"
    )
