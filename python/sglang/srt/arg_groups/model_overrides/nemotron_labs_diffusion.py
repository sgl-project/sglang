"""Config-time attention defaults for Nemotron Labs Diffusion."""

from typing import Any

from sglang.srt.arg_groups.model_override_base import (
    register_model_override,
    resolving_view,
)
from sglang.srt.runtime_context import get_platform


@register_model_override("NemotronLabsDiffusionModel")
def _nemotron_labs_diffusion_overrides(server_args: Any, hf_config: Any) -> dict:
    cfg = resolving_view(server_args)
    # A supported default keeps dLLM validation from replacing split backends.
    if (
        cfg.dllm_algorithm in ("NemotronFastDiffuser", "NemotronLinearSpec")
        and get_platform().is_sm100
        and cfg.attention_backend is None
    ):
        return {"attention_backend": "fa4"}
    return {}
