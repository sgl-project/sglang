"""Config-time override declarations for Clover."""

from typing import Any

from sglang.srt.arg_groups.model_override_base import _register_for, resolving_view


@_register_for("CloverMoeModelLM")
def _clover_overrides(server_args: Any, hf_config: Any) -> dict:
    cfg = resolving_view(server_args)
    if (
        cfg.speculative_algorithm is None
        or cfg.speculative_algorithm.upper() != "CLOVER"
    ):
        raise ValueError("CloverMoeModelLM requires --speculative-algorithm clover")
    if cfg.dllm_algorithm is not None:
        raise ValueError(
            "Clover does not support --dllm-algorithm; use --speculative-algorithm clover"
        )
    # Keep configurations outside the current validation scope gated pending
    # targeted tests. An unvalidated configuration is not necessarily unsupported.
    # CUDA is required by the preview kernels; the model wrapper requires PP=1.
    if cfg.device != "cuda" or cfg.dtype not in ("auto", "bfloat16"):
        raise ValueError(
            "Clover preview kernels require CUDA; serving dtypes other than BF16 are unvalidated"
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
    if cfg.quantization is not None:
        raise ValueError("Clover supports only unquantized models")
    if cfg.disaggregation_mode != "null" or cfg.enable_hierarchical_cache:
        raise ValueError("Clover with disaggregation or HiCache is unvalidated")
    overrides = {
        "attention_backend": "flashinfer",
        "mamba_radix_cache_strategy": (
            "no_buffer" if cfg.disable_radix_cache else "extra_buffer"
        ),
    }
    if cfg.ep_size > 1:
        overrides["moe_runner_backend"] = "triton"
    return overrides
