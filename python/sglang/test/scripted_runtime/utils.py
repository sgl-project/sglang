"""Shared helpers for the scripted-runtime package.

Process-boundary utilities used by both the scheduler-side hook and the
caller-side entry points: forwarding the script module onto ``sys.path``
in spawn-mode subprocesses, and resolving a script to a qualified name.
"""

from __future__ import annotations

import importlib
import sys
from typing import Callable, Optional


def ensure_script_importable(sys_path_entry: Optional[str]) -> None:
    """Forward the script module's directory onto ``sys.path``.

    Spawn-mode mp subprocesses don't inherit the parent's ``sys.path``, so
    the script can't be imported by qualified name without this. No-op when
    the entry is unset or already present.
    """
    if sys_path_entry and sys_path_entry not in sys.path:
        sys.path.insert(0, sys_path_entry)


def resolve_fn(qualified: str) -> Callable:
    """Resolve ``"module.path:qualname"`` to the function object.

    The leaf must be importable across processes — no lambdas / closures.
    """
    module_name, sep, fn_name = qualified.partition(":")
    if not sep or not module_name or not fn_name:
        raise ValueError(
            f"scripted-runtime fn path must be 'module.path:function_name', "
            f"got {qualified!r}"
        )
    obj = importlib.import_module(module_name)
    for part in fn_name.split("."):
        obj = getattr(obj, part)
    if not callable(obj):
        raise TypeError(f"resolved object is not callable: {qualified!r} -> {obj!r}")
    return obj
