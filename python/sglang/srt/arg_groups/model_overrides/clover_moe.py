"""Config-time override declarations for Clover."""

import logging
from typing import Any

from sglang.srt.arg_groups.model_override_base import _register_for, resolving_view
from sglang.srt.utils.common import get_device_sm

logger = logging.getLogger(__name__)


@_register_for("CloverMoeModelLM")
def _clover_overrides(server_args: Any, hf_config: Any) -> dict:
    cfg = resolving_view(server_args)
    pure = cfg.dllm_algorithm == "CloverPureBlock" and cfg.speculative_algorithm is None
    speculative = (
        cfg.speculative_algorithm or ""
    ).upper() == "CLOVER" and cfg.dllm_algorithm is None
    if not (pure or speculative):
        raise ValueError(
            "CloverMoeModelLM requires --speculative-algorithm clover or --dllm-algorithm CloverPureBlock, exclusively"
        )
    if cfg.device != "cuda" or cfg.dtype not in ("auto", "bfloat16"):
        raise ValueError(
            "Clover preview kernels require CUDA; serving dtypes other than BF16 are unvalidated"
        )
    supported_sm = (9, 10, 12) if pure else (9, 10)
    if get_device_sm() // 10 not in supported_sm:
        raise ValueError(
            f"Clover requires compute capability in {supported_sm} for the selected mode"
        )
    if (
        cfg.linear_attn_backend not in ("triton", "flashinfer")
        or cfg.linear_attn_prefill_backend not in (None, "triton", "flashinfer")
        or cfg.linear_attn_decode_backend not in (None, "triton", "flashinfer")
    ):
        raise ValueError(
            "Clover GDN backends other than Triton and FlashInfer are unvalidated"
        )
    if any(
        backend not in (None, "flashinfer")
        for backend in (
            cfg.attention_backend,
            cfg.prefill_attention_backend,
            cfg.decode_attention_backend,
        )
    ):
        raise ValueError(
            "Clover full-attention backends other than FlashInfer are unvalidated"
        )
    if cfg.dcp_size != 1 or cfg.attn_cp_size != 1 or cfg.attn_dp_size != 1:
        raise ValueError(
            "Clover with DCP or attention DP/CP greater than 1 is unvalidated"
        )
    if cfg.pp_size != 1:
        raise ValueError("Clover requires pipeline parallel size 1")
    # A2A backends can expand EP during later argument resolution.
    if cfg.moe_a2a_backend != "none":
        raise ValueError("Clover requires --moe-a2a-backend none")
    if cfg.ep_size > 1 and cfg.moe_runner_backend not in ("auto", "triton"):
        raise ValueError("Clover EP requires --moe-runner-backend triton")
    if cfg.enable_eplb:
        raise ValueError("Clover does not support --enable-eplb")
    if cfg.quantization is not None:
        raise ValueError("Clover supports only unquantized models")
    if cfg.disaggregation_mode != "null" or cfg.enable_hierarchical_cache:
        raise ValueError("Clover with disaggregation or HiCache is unvalidated")
    strategy = "no_buffer" if pure or cfg.disable_radix_cache else "extra_buffer"
    if cfg.mamba_radix_cache_strategy not in ("auto", strategy):
        logger.warning(
            "Clover requires mamba_radix_cache_strategy=%s, overriding %s.",
            strategy,
            cfg.mamba_radix_cache_strategy,
        )
    overrides = {
        "attention_backend": "flashinfer",
        "mamba_radix_cache_strategy": strategy,
    }
    if cfg.ep_size > 1:
        overrides["moe_runner_backend"] = "triton"
    return overrides
