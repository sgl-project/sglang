import os
from typing import Any

from sglang.srt.arg_groups.model_override_base import (
    _register_for,
    is_attention_backend_not_set,
    resolving_view,
)
from sglang.srt.runtime_context import get_platform

_SUPPORTED_MOE_RUNNERS = ("auto", "triton", "flashinfer_trtllm_routed")
_SINK_ATTENTION_BACKENDS = ("fa3", "fa4", "triton", "trtllm_mha")


@_register_for("FlintForCausalLM")
def _flint_overrides(server_args: Any, hf_config: Any) -> dict:
    cfg = resolving_view(server_args)
    if cfg.moe_runner_backend not in _SUPPORTED_MOE_RUNNERS:
        raise ValueError(
            f"Flint requires --moe-runner-backend in {_SUPPORTED_MOE_RUNNERS}, "
            f"got {cfg.moe_runner_backend}."
        )
    if cfg.moe_a2a_backend != "none":
        raise ValueError("Flint does not support --moe-a2a-backend.")
    requested = (
        cfg.attention_backend,
        cfg.prefill_attention_backend,
        cfg.decode_attention_backend,
    )
    if any(b not in (None, *_SINK_ATTENTION_BACKENDS) for b in requested):
        raise ValueError(
            f"Flint needs attention sinks; use one of {_SINK_ATTENTION_BACKENDS}."
        )
    if "trtllm_mha" in requested and len(set(requested) - {None}) > 1:
        raise ValueError("Flint cannot mix trtllm_mha with other attention backends.")
    os.environ.setdefault("FLASHINFER_USE_CUDA_NORM", "1")

    overrides = {}
    default_backend = (
        "trtllm_mha" if "trtllm_mha" in requested else _default_attention_backend()
    )
    if is_attention_backend_not_set(cfg):
        overrides["attention_backend"] = default_backend
    elif cfg.attention_backend is None:
        if cfg.prefill_attention_backend is None:
            overrides["prefill_attention_backend"] = default_backend
        if cfg.decode_attention_backend is None:
            overrides["decode_attention_backend"] = default_backend
    if cfg.moe_runner_backend == "auto":
        overrides["moe_runner_backend"] = (
            "flashinfer_trtllm_routed" if get_platform().is_sm100 else "triton"
        )
    return overrides


def _default_attention_backend() -> str:
    if get_platform().is_sm100:
        return "fa4"
    if get_platform().is_sm90:
        return "fa3"
    return "triton"
