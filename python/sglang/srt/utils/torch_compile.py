"""Compilation utilities for torch.compile with debug support."""

from __future__ import annotations

import functools
from typing import Any, Callable, Dict, Optional

import torch
import torch._dynamo.config as dynamo_config
import torch._inductor.config as inductor_config


def _torch_compile(
    fn: Callable,
    compile_kwargs: Optional[Dict[str, Any]] = None,
    dynamo_kwargs: Optional[Dict[str, Any]] = None,
    inductor_kwargs: Optional[Dict[str, Any]] = None,
    debug: bool = False,
) -> Callable:
    """Compile a function with torch.compile and apply localized compilation config."""
    if compile_kwargs is None:
        compile_kwargs = {}
    if inductor_kwargs is None:
        inductor_kwargs = {}
    if dynamo_kwargs is None:
        dynamo_kwargs = {}

    compiled_fn = None

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        nonlocal compiled_fn

        if compiled_fn is None:
            # First call – compile with debug settings and caching disabled
            with dynamo_config.patch(
                verbose=True,
                **dynamo_kwargs,
            ), inductor_config.patch(
                **inductor_kwargs,
                **{
                    "debug": debug,
                    "trace.enabled": debug,
                    "trace.fx_graph": debug,
                    "trace.fx_graph_transformed": debug,
                    "trace.output_code": debug,
                    "fx_graph_cache": False if debug else True,
                    "force_disable_caches": debug,
                },
            ):
                compiled_fn = torch.compile(fn, **compile_kwargs)
                return compiled_fn(*args, **kwargs)

        # Replay with same config
        with dynamo_config.patch(
            **dynamo_kwargs,
        ), inductor_config.patch(**inductor_kwargs):
            return compiled_fn(*args, **kwargs)

    return wrapper


def warmup_compiled_fn(compiled_fn: Callable, *args: Any, **kwargs: Any) -> None:
    """Run warmup iterations on a compiled function and synchronize."""
    for _ in range(10):
        compiled_fn(*args, **kwargs)
    torch.cuda.synchronize()
