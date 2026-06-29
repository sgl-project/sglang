from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


def apply_deepseek_v4_defaults(server_args: ServerArgs, model_arch: str) -> None:
    """Apply DeepSeek V4 model-specific server arg defaults and constraints."""
    from sglang.srt.server_args import ServerArgs

    server_args.attention_backend = "dsv4"
    server_args.page_size = 256
    if server_args.kv_cache_dtype == "auto":
        server_args.kv_cache_dtype = "fp8_e4m3"
        logger.warning(
            f"Setting KV cache dtype to {server_args.kv_cache_dtype} for {model_arch}."
        )

    if server_args.device == "npu":
        # NPU keeps the device-aware "dsv4" backend (the registry routes it to
        # the Ascend V4 subclass); only the pool geometry / dtype differ.
        # set_default_server_args() pins all three backends to "ascend" for
        # generic NPU models; undo that here so V4 stays consistently on dsv4.
        server_args.prefill_attention_backend = "dsv4"
        server_args.decode_attention_backend = "dsv4"
        server_args.page_size = 128
        server_args.kv_cache_dtype = "bfloat16"

    logger.info(
        f"Use dsv4 attention backend for {model_arch}, setting page_size to {server_args.page_size}."
    )
    assert server_args.kv_cache_dtype in [
        "fp8_e4m3",
        "bfloat16",
    ], f"{server_args.kv_cache_dtype} is not supported for {model_arch}"

    if server_args.max_running_requests is None:
        server_args.max_running_requests = 256
        logger.warning(
            f"Setting max_running_requests to {server_args.max_running_requests} for {model_arch}."
        )

    if server_args.speculative_algorithm is not None:
        assert (
            server_args.speculative_algorithm == "EAGLE"
        ), f"Only EAGLE speculative algorithm is supported for {model_arch}"
        assert (
            server_args.speculative_eagle_topk == 1
        ), f"Only EAGLE speculative algorithm with topk == 1 is supported for {model_arch}"

    if server_args.swa_full_tokens_ratio == ServerArgs.swa_full_tokens_ratio:
        server_args.swa_full_tokens_ratio = 0.1
        logger.info(
            f"Setting swa_full_tokens_ratio to {server_args.swa_full_tokens_ratio} for {model_arch}."
        )

    # nvidia/DeepSeek-V4-Pro-NVFP4 uses flashinfer_trtllm_routed MoE runner backend.
    if (
        server_args.moe_runner_backend == "auto"
        and server_args.get_model_config().nvfp4_moe_meta is not None
    ):
        server_args.moe_runner_backend = "flashinfer_trtllm_routed"
        logger.info(
            "Use flashinfer_trtllm_routed as MoE runner backend for "
            f"{model_arch} hybrid FP8+NVFP4 checkpoint."
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
        server_args.nnodes == 1
    ), "Context parallel only supports single-machine runs. Cross-machine CP has precision issues."
    logger.warning(
        f"Enable Context Parallel for DeepSeekV4, "
        f"dp_size={server_args.dp_size}, moe_dense_tp_size={server_args.moe_dense_tp_size}, "
        f"attn_cp_size={server_args.attn_cp_size}, ep_size={server_args.ep_size}, tp_size={server_args.tp_size}"
    )
