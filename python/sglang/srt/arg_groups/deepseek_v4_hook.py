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

    dsv4_decode_backend = envs.SGLANG_DSV4_ATTN_DECODE_BACKEND.get()
    if dsv4_decode_backend not in ("auto", "flashmla", "trtllm_gen"):
        raise ValueError(
            f"Invalid SGLANG_DSV4_ATTN_DECODE_BACKEND={dsv4_decode_backend!r}; "
            "expected 'auto', 'flashmla', or 'trtllm_gen'."
        )
    if dsv4_decode_backend == "trtllm_gen":
        # Opt-in SM100 trtllm-gen decode + sparse varlen prefill on a uniform
        # 512-dim FP8 KV cache. kv_cache_dtype stays fp8_e4m3; only the
        # in-pool layout changes.
        assert (
            server_args.device == "cuda"
        ), "SGLANG_DSV4_ATTN_DECODE_BACKEND=trtllm_gen is CUDA (SM100) only."
        assert server_args.kv_cache_dtype == "fp8_e4m3", (
            "SGLANG_DSV4_ATTN_DECODE_BACKEND=trtllm_gen requires "
            f"kv_cache_dtype=fp8_e4m3, got {server_args.kv_cache_dtype}."
        )
        assert server_args.speculative_algorithm is None, (
            "SGLANG_DSV4_ATTN_DECODE_BACKEND=trtllm_gen does not support "
            "speculative decoding (MTP) yet."
        )
        assert not server_args.enable_hisparse, (
            "SGLANG_DSV4_ATTN_DECODE_BACKEND=trtllm_gen does not support "
            "enable_hisparse (the hisparse c4 pool uses the packed layout)."
        )
        logger.info(
            "DeepSeek V4 attention (decode + sparse prefill): trtllm-gen "
            "uniform-FP8 path enabled "
            "(SGLANG_DSV4_ATTN_DECODE_BACKEND=trtllm_gen)."
        )

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
