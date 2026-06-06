# SPDX-License-Identifier: Apache-2.0

from typing import Any

from sglang.multimodal_gen.runtime.managers.forward_context import get_forward_context

SANA_WM_REQUEST_RUNTIME_CACHE_KEY = "_sana_wm_runtime_cache"


def get_sana_wm_request_cache(name: str) -> dict | None:
    """Return a named cache bucket scoped to the current SANA-WM request."""
    try:
        forward_batch = get_forward_context().forward_batch
    except AssertionError:
        return None

    extra = getattr(forward_batch, "extra", None)
    if extra is None:
        return None

    runtime_cache = extra.setdefault(SANA_WM_REQUEST_RUNTIME_CACHE_KEY, {})
    return runtime_cache.setdefault(name, {})


def clear_sana_wm_request_runtime_cache(batch: Any) -> None:
    extra = getattr(batch, "extra", None)
    if extra is not None:
        extra.pop(SANA_WM_REQUEST_RUNTIME_CACHE_KEY, None)
