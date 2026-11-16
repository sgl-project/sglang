"""Declarations for models that cannot accept in-place weight updates."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from typing import Callable, Iterator, Optional

import torch

_PLUGIN_REASON = "_sglang_plugin_weight_update_unsupported_reason"
_declaration_revision = 0


class _RequestChecks:
    def __init__(self) -> None:
        # Keep the model alive until the request ends so its id cannot be reused.
        self.models: Optional[dict[int, tuple[torch.nn.Module, Optional[str]]]] = {}
        self.revision = _declaration_revision


_request: ContextVar[Optional[_RequestChecks]] = ContextVar(
    "weight_update_request", default=None
)


@contextmanager
def weight_update_request() -> Iterator[None]:
    """Share a model check between preflight and loading in one scheduler call."""
    scope = _RequestChecks()
    token = _request.set(scope)
    try:
        yield
    finally:
        # A plugin may copy the current context. End reuse there as well.
        scope.models = None
        _request.reset(token)


def declare_weight_update_unsupported(
    module: torch.nn.Module, reason: Optional[str]
) -> None:
    """Declare before serving, identically on every rank. None clears this declaration."""
    global _declaration_revision
    if reason is not None and not isinstance(reason, str):
        raise TypeError("weight update refusal reason must be a string or None")
    if reason is None:
        vars(module).pop(_PLUGIN_REASON, None)
    else:
        vars(module)[_PLUGIN_REASON] = reason
    # Also invalidate a suspended outer request if a plugin declares in a nested call.
    _declaration_revision += 1


def plugin_weight_update_reason(module: torch.nn.Module) -> Optional[str]:
    return vars(module).get(_PLUGIN_REASON)


def cache_weight_update_check(
    check: Callable[[Optional[torch.nn.Module]], Optional[str]],
):
    """Reuse the runner's refusal check within a single scheduler request."""

    @wraps(check)
    def checked(model: Optional[torch.nn.Module] = None) -> Optional[str]:
        scope = _request.get()
        if scope is None or scope.models is None or model is None:
            return check(model)
        if scope.revision != _declaration_revision:
            scope.models.clear()
            scope.revision = _declaration_revision
        key = id(model)
        if key not in scope.models:
            scope.models[key] = model, check(model)
        return scope.models[key][1]

    return checked
