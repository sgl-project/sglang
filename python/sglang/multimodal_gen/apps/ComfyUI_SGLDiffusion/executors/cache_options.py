# SPDX-License-Identifier: Apache-2.0
"""Per-run cache options carried in ``model_options["transformer_options"]``.

They ride on a cloned MODEL, so changing them never touches the loader options
and never restarts the worker.
"""

from __future__ import annotations

from typing import Any

CACHE_OPTIONS_KEY = "sgld_cache_options"


def build_cache_options(
    enable_cache_dit: bool | None, cache_dit_params: dict[str, Any] | None
) -> dict[str, Any]:
    """Plain, pickle-safe options; unknown cache_dit_params keys raise."""
    from sglang.multimodal_gen.runtime.cache.cache_dit_integration import (
        resolve_cache_dit_request_overrides,
    )

    params = resolve_cache_dit_request_overrides(cache_dit_params)
    if params and enable_cache_dit is not True:
        raise ValueError(
            "cache_dit_params were given but enable_cache_dit is not on; "
            "they would be ignored."
        )
    return {"enable_cache_dit": enable_cache_dit, "cache_dit_params": params or None}


def check_cache_options_supported(executor, options: dict[str, Any]) -> None:
    """Raise if the options ask for a cache the executor's model cannot use."""
    if options.get("enable_cache_dit") is True and not executor.supports_cache_dit:
        raise ValueError(
            f"{type(executor).__name__} does not support Cache-DiT through "
            "ComfyUI: this model runs one independent step per request, so "
            "there is no step history to reuse. Supported: MiniMax-H3."
        )


def with_cache_options(model, options: dict[str, Any]):
    """Return a clone of ``model`` carrying ``options``; ``model`` is untouched."""
    executor = model.model.diffusion_model
    check_cache_options_supported(executor, options)
    cloned = model.clone()
    transformer_options = dict(cloned.model_options.get("transformer_options") or {})
    transformer_options[CACHE_OPTIONS_KEY] = options
    cloned.model_options["transformer_options"] = transformer_options
    return cloned


def read_cache_options(transformer_options: dict[str, Any] | None) -> dict[str, Any]:
    return (transformer_options or {}).get(CACHE_OPTIONS_KEY) or {}
