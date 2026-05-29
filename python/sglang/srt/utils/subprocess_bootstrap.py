"""Helpers for lazily bootstrapping multiprocessing subprocess targets."""

import importlib
from collections.abc import Callable
from typing import Any


def resolve_subprocess_target(target: str) -> Callable[..., Any]:
    """Resolve a ``module:function`` target string to a callable."""
    module_name, separator, function_name = target.partition(":")
    if not separator or not module_name or not function_name:
        raise ValueError(
            f"Invalid subprocess target {target!r}. Expected 'module:function'."
        )

    module = importlib.import_module(module_name)
    function = getattr(module, function_name)
    if not callable(function):
        raise TypeError(f"Subprocess target {target!r} is not callable.")
    return function


def run_subprocess_target(target: str, *args: Any, **kwargs: Any) -> Any:
    """Run startup plugins, then import and call a subprocess target.

    ``target`` is intentionally passed as ``module:function`` instead of a function
    object so spawn-mode subprocesses import only this lightweight bootstrap module
    before startup plugins get a chance to install import-time compatibility shims.
    """
    from sglang.srt.plugins import load_startup_plugins

    load_startup_plugins()
    function = resolve_subprocess_target(target)
    return function(*args, **kwargs)
