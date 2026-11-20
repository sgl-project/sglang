"""Config-time override declarations for Step5."""

from typing import Any

from sglang.srt.arg_groups.model_override_base import (
    _register_for,
    is_attention_backend_not_set,
    resolving_view,
)
from sglang.srt.runtime_context import get_platform

_STEP5_PAGE_SIZE = 64


def _validate_features(cfg: Any) -> None:
    if cfg.kv_cache_dtype not in ("auto", "bf16", "bfloat16"):
        raise ValueError("Step5 supports --kv-cache-dtype auto or bfloat16")
    if cfg.speculative_algorithm is not None:
        raise ValueError("Step5 does not support speculative decoding yet")
    if cfg.enable_hierarchical_cache:
        raise ValueError("Step5 does not support --enable-hierarchical-cache yet")
    if cfg.disaggregation_mode != "null":
        raise ValueError("Step5 does not support PD disaggregation yet")
    if cfg.attn_dp_size > 1:
        raise ValueError("Step5 does not support attention data parallelism yet")
    if cfg.moe_a2a_backend != "none":
        raise ValueError("Step5 supports only --moe-a2a-backend none")
    if cfg.page_size not in (None, _STEP5_PAGE_SIZE):
        raise ValueError(f"Step5 requires --page-size {_STEP5_PAGE_SIZE}")
    if cfg.boundary_reduction not in ("auto", "ar"):
        raise ValueError("Step5 supports only --boundary-reduction ar")
    if not is_attention_backend_not_set(cfg) and cfg.attention_backend != "triton":
        raise ValueError("Step5 supports only --attention-backend triton")


def _moe_runner_backend(cfg: Any) -> str:
    if cfg.moe_runner_backend == "auto":
        return "flashinfer_trtllm_routed" if get_platform().is_sm100 else "triton"
    if cfg.moe_runner_backend not in ("triton", "flashinfer_trtllm_routed"):
        raise ValueError(
            "Step5 supports --moe-runner-backend triton or flashinfer_trtllm_routed"
        )
    return cfg.moe_runner_backend


@_register_for("Step5ForCausalLM", "Step5ForConditionalGeneration")
def _step5_overrides(server_args: Any, hf_config: Any) -> dict:
    cfg = resolving_view(server_args)
    _validate_features(cfg)
    return {
        "attention_backend": "triton",
        "page_size": _STEP5_PAGE_SIZE,
        "boundary_reduction": "ar",
        "moe_runner_backend": _moe_runner_backend(cfg),
    }
