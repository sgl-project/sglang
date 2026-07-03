from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


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

    validate_deepseek_v4_dcp(server_args)


def validate_deepseek_v4_dcp(server_args: ServerArgs) -> None:
    """Validate DeepSeek V4 DCP (decode context parallel) compatibility."""
    if server_args.dcp_size <= 1:
        return

    if not envs.SGLANG_DSV4_ENABLE_DCP.get():
        raise ValueError(
            "DeepSeekV4 DCP (--dcp-size > 1) is gated behind "
            "SGLANG_DSV4_ENABLE_DCP=1. Set the env var explicitly after "
            "verifying numerical equivalence vs. dcp_size=1 on your cluster."
        )

    # DSV4 DCP reduce-scatters attention output across the attention-TP head
    # dimension, so the DCP group must match the attention-TP group.
    attn_tp = server_args.tp_size // max(server_args.dp_size, 1)
    if attn_tp != server_args.dcp_size:
        raise ValueError(
            f"DeepSeekV4 DCP currently requires dcp_size ({server_args.dcp_size}) "
            f"== attn_tp ({attn_tp}); configure tp/dp_size/dcp_size accordingly."
        )

    # Online c128 + DCP: untested combination; warn but allow.
    if envs.SGLANG_OPT_USE_ONLINE_COMPRESS.get():
        logger.warning(
            "DeepSeekV4 DCP + SGLANG_OPT_USE_ONLINE_COMPRESS combo is "
            "experimental; numerical correctness has not been validated."
        )

    # HiSparse + DCP: HiSparse C4 device pool uses index translation that is
    # not yet DCP-aware in the fused C++ kernels; only the Triton fallback
    # write path supports DCP.
    if server_args.enable_hisparse:
        logger.warning(
            "DeepSeekV4 DCP + enable_hisparse falls back to Triton write "
            "path for KV writes; expect throughput regression vs. fused C++."
        )

    logger.info(
        f"DeepSeekV4 DCP enabled: dcp_size={server_args.dcp_size}, "
        f"attn_tp={attn_tp}"
    )


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
