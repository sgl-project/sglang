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

    if cfg.dsv4_attn_backend == "trtllm":
        from sglang.srt.utils.common import is_sm100_supported

        assert cfg.device == "cuda" and is_sm100_supported(), (
            "--dsv4-attn-backend trtllm requires an SM100/SM103 (Blackwell) GPU."
        )
        # The resolution pipeline materializes "auto" as fp8_e4m3 on CUDA.
        assert cfg.kv_cache_dtype in ("auto", "fp8_e4m3"), (
            "--dsv4-attn-backend trtllm requires kv_cache_dtype=fp8_e4m3, "
            f"got {cfg.kv_cache_dtype}."
        )
        assert not cfg.enable_hisparse, (
            "--dsv4-attn-backend trtllm does not support enable_hisparse."
        )
        assert not (
            cfg.attn_cp_size > 1 or cfg.dcp_size > 1 or cfg.enable_prefill_cp
        ), (
            "--dsv4-attn-backend trtllm does not support context parallelism "
            "(prefill CP, attention CP, or decode CP)."
        )
        # The trtllm backend stores KV in a 512-byte uniform-FP8 layout while
        # FlashMLA uses the 584-byte packed layout; the PD handshake only
        # compares kv_cache_dtype, so mismatched prefill/decode backends would
        # pass the check and transfer garbage. Reject until the handshake
        # carries a layout identifier and the path is tested (#37838).
        assert cfg.disaggregation_mode == "null", (
            "--dsv4-attn-backend trtllm does not support PD disaggregation yet "
            "(uniform-FP8 KV layout is not part of the PD handshake; see "
            "https://github.com/sgl-project/sglang/issues/37838)."
        )
        # The trtllm-gen semaphore buffer is sized from the prefill chunk
        # bound; with chunking disabled a single long request has no bound.
        assert cfg.chunked_prefill_size is not None and cfg.chunked_prefill_size > 0, (
            "--dsv4-attn-backend trtllm requires chunked prefill "
            "(--chunked-prefill-size > 0)."
        )
        logger.info(
            "DeepSeek V4 attention: trtllm backend enabled "
            "(uniform-FP8 KV pool, decode + sparse prefill)."
        )

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
