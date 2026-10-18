# SPDX-License-Identifier: Apache-2.0
"""Per-run cache options carried in ``model_options["transformer_options"]``.

They ride on a cloned MODEL, so changing them never touches the loader options
and never restarts the worker.
"""

from __future__ import annotations

from typing import Any

CACHE_OPTIONS_KEY = "sgld_cache_options"

# Measured on Flux.1-dev, 25 steps, 1024px, H100 vs cache off: 1.37x at
# 22.5 dB PSNR / 0.81 SSIM; the server default (warmup 4, threshold 0.24) gives
# 1.81x at 18.1 dB / 0.62.
CONSERVATIVE_PRESET = {
    "max_warmup_steps": 8,
    "residual_diff_threshold": 0.08,
}


def build_cache_options(
    enable_cache_dit: bool | None, cache_dit_params: dict[str, Any] | None
) -> dict[str, Any]:
    """Plain, pickle-safe options; unknown cache_dit_params keys raise.

    Knobs are kept when caching is on or left to the loader/server default, and
    dropped when it is off, so a tuned node can be switched off without resetting
    every knob.
    """
    from sglang.multimodal_gen.runtime.cache.cache_dit_integration import (
        resolve_cache_dit_request_overrides,
    )

    params = resolve_cache_dit_request_overrides(cache_dit_params)
    if enable_cache_dit is False:
        params = {}
    return {"enable_cache_dit": enable_cache_dit, "cache_dit_params": params or None}


def request_cache_dit_for_run(packed, transformer_options, separate_cfg: bool) -> None:
    """Ask the worker to keep one Cache-DiT context for the whole sampler run.

    ComfyUI sends one request per DiT call. With CFG the cond and uncond calls
    alternate, which Cache-DiT counts as one step when mounted with separate CFG.
    """
    options = transformer_options or {}
    if 1 in (options.get("cond_or_uncond") or ()) and not separate_cfg:
        raise ValueError(
            "Cache-DiT through ComfyUI got an uncond call in a run whose guider "
            "does not run CFG; set cfg to 1 or turn Cache-DiT off for this run."
        )
    sigmas = options.get("sample_sigmas")
    if sigmas is None or len(sigmas) < 2:
        raise ValueError("Cache-DiT through ComfyUI needs the sampler's sample_sigmas")
    packed.extra_req["comfyui_num_steps"] = len(sigmas) - 1
    packed.extra_req["comfyui_separate_cfg"] = separate_cfg


def guider_runs_cfg(guider) -> bool:
    """Whether this sampler run evaluates cond and uncond every step.

    ComfyUI skips the uncond branch at cfg == 1; guiders other than CFGGuider
    (BasicGuider, custom ones) carry no cfg and run a single branch.
    """
    cfg = getattr(guider, "cfg", 1.0)
    conds = getattr(guider, "conds", None) or {}
    return cfg != 1.0 and "negative" in conds


def with_cache_options(model, options: dict[str, Any]):
    """Return a clone of ``model`` carrying ``options``; ``model`` is untouched."""
    cloned = model.clone()
    transformer_options = dict(cloned.model_options.get("transformer_options") or {})
    transformer_options[CACHE_OPTIONS_KEY] = options
    cloned.model_options["transformer_options"] = transformer_options
    return cloned


def read_cache_options(transformer_options: dict[str, Any] | None) -> dict[str, Any]:
    return (transformer_options or {}).get(CACHE_OPTIONS_KEY) or {}
