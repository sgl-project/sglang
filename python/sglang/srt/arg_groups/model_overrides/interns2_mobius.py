"""Config-time override declarations for interns2_mobius."""

from typing import Any

from sglang.srt.arg_groups.model_override_base import (
    _register_for,
    resolving_view,
)
from sglang.srt.runtime_context import get_platform


@_register_for("InternS2MobiusForConditionalGeneration")
def _interns2_mobius_baseline_overrides(server_args: Any, hf_config: Any) -> dict:
    """Select the only MoE runner validated for the 2,560-expert baseline."""
    cfg = resolving_view(server_args)
    # triton_kernel makes FusedMoE load w13/w2 transposed, but the CPU FP8
    # MoE method allocates them untransposed, so leave auto unchanged on CPU.
    if cfg.moe_runner_backend == "auto" and not get_platform().is_cpu:
        return {"moe_runner_backend": "triton_kernel"}
    return {}
