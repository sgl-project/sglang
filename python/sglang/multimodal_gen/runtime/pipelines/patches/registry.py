# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

from .base import DiffusionRolloutAdapter
from .contracts import RolloutRequest

_ADAPTERS: list[DiffusionRolloutAdapter] = []
_DEFAULT_ADAPTERS_REGISTERED = False


def register_adapter(adapter: DiffusionRolloutAdapter) -> None:
    for existing in _ADAPTERS:
        if existing.name == adapter.name:
            return
    _ADAPTERS.append(adapter)


def _ensure_default_adapters() -> None:
    global _DEFAULT_ADAPTERS_REGISTERED

    if _DEFAULT_ADAPTERS_REGISTERED:
        return

    from .sd3_adapter import SD3LogprobAdapter

    register_adapter(SD3LogprobAdapter())
    _DEFAULT_ADAPTERS_REGISTERED = True


def resolve_adapter(
    pipe: Any, request: RolloutRequest
) -> DiffusionRolloutAdapter | None:
    _ensure_default_adapters()

    requested = str(request.get("adapter", "auto")).lower()
    if requested != "auto":
        for adapter in _ADAPTERS:
            if adapter.name.lower() == requested and adapter.can_handle(pipe, request):
                return adapter
        return None

    for adapter in _ADAPTERS:
        if adapter.can_handle(pipe, request):
            return adapter
    return None
