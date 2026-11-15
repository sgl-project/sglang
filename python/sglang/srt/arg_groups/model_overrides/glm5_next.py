"""Config-time override declarations for glm5_next."""

from typing import Any, Dict

from sglang.srt.arg_groups.model_override_base import (
    _register_for,
    resolving_view,
)


@_register_for("Glm5NextForConditionalGeneration")
def _glm5_next_overrides(server_args: Any, hf_config: Any) -> dict:
    """Size the KDA state pool for multi-turn agentic traffic: a small state
    pool with lazily allocated tracking slots, leaving the rest to KV, keeps
    the sessions' prefixes resident. Each default yields to an explicit value."""
    from sglang.srt.arg_groups.overrides import supports_mamba_cache_extra_buffer

    cfg = resolving_view(server_args)
    overrides: Dict[str, Any] = {}
    if cfg.mamba_full_memory_ratio is None:
        overrides["mamba_full_memory_ratio"] = 0.25
    # Replaces the extra_buffer pick "auto" would make, where the lazy variant
    # is supported; the mamba radix-cache pass then leaves it alone.
    if (
        cfg.mamba_radix_cache_strategy == "auto"
        and supports_mamba_cache_extra_buffer(cfg, hf_config)
        and cfg.disaggregation_mode == "null"
        and not cfg.disable_overlap_schedule
        and cfg.pp_size == 1
    ):
        overrides["mamba_radix_cache_strategy"] = "extra_buffer_lazy"
    return overrides
