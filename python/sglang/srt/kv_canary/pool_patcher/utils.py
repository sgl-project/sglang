from __future__ import annotations

import functools
from typing import Any, Callable

_WRAPPED_MARKER_ATTR = "_kv_canary_wrapped_by"


def wrap_method(
    obj: object,
    method_name: str,
    *,
    wrapper: Callable[..., Any],
) -> None:
    """Replace ``obj.method_name`` with a closure that delegates to ``wrapper``.

    ``wrapper(original, *args, **kwargs)`` receives the original bound method as its first arg and the
    call-site args/kwargs as the rest. It decides when (and whether) to call ``original`` and what to
    return. The patched callable is installed as a plain function; :func:`functools.wraps` preserves
    ``__name__`` / ``__doc__`` but the bound-method nature of the original is not retained.

    Raises:
        AttributeError: ``obj`` has no attribute ``method_name``.
        RuntimeError: ``obj.method_name`` has already been wrapped by ``wrap_method`` (idempotency
            guard — re-wrapping silently would stack two transforms and corrupt return values).
    """
    if not hasattr(obj, method_name):
        raise AttributeError(
            f"kv-canary: {type(obj).__name__} missing required method {method_name!r}"
        )
    original = getattr(obj, method_name)
    if getattr(original, _WRAPPED_MARKER_ATTR, None) is not None:
        raise RuntimeError(
            f"kv-canary: {type(obj).__name__}.{method_name} already wrapped by kv-canary"
        )

    @functools.wraps(original)
    def patched(*args: Any, **kwargs: Any) -> Any:
        return wrapper(original, *args, **kwargs)

    setattr(patched, _WRAPPED_MARKER_ATTR, method_name)
    setattr(obj, method_name, patched)
