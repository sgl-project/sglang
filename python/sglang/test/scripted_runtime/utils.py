from __future__ import annotations

import importlib
import sys
from typing import Callable, Optional

import zmq


def close_zmq_socket(socket: zmq.Socket, ctx: zmq.Context) -> None:
    socket.setsockopt(zmq.LINGER, 0)
    socket.close()
    ctx.term()


def ensure_script_importable(sys_path_entry: Optional[str]) -> None:
    if sys_path_entry and sys_path_entry not in sys.path:
        sys.path.insert(0, sys_path_entry)


def resolve_fn(qualified: str) -> Callable:
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
