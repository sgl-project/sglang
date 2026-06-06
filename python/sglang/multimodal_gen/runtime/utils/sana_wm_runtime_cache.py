# SPDX-License-Identifier: Apache-2.0

from typing import Any

from sglang.multimodal_gen.runtime.managers.forward_context import (
    get_forward_context_or_none,
)

SANA_WM_REQUEST_RUNTIME_CACHE_KEY = "_sana_wm_runtime_cache"


def get_sana_wm_request_cache(name: str) -> dict | None:
    """Return a named cache bucket scoped to the current SANA-WM request.

    Returns ``None`` when called outside an active forward context (e.g.
    during model initialisation or tests), which signals to the caller that
    caching is unavailable and it should fall back to direct computation.
    """
    ctx = get_forward_context_or_none()
    if ctx is None:
        return None

    forward_batch = ctx.forward_batch
    extra = getattr(forward_batch, "extra", None)
    if extra is None:
        return None

    runtime_cache = extra.setdefault(SANA_WM_REQUEST_RUNTIME_CACHE_KEY, {})
    return runtime_cache.setdefault(name, {})


def clear_sana_wm_request_runtime_cache(batch: Any) -> None:
    extra = getattr(batch, "extra", None)
    if extra is not None:
        extra.pop(SANA_WM_REQUEST_RUNTIME_CACHE_KEY, None)
