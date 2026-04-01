"""Compilation utilities for torch.compile with debug support."""

from __future__ import annotations

import functools
from typing import Any, Callable, Dict, Optional

import torch
import torch._dynamo.config as dynamo_config
import torch._inductor.config as inductor_config


def compile_with_debug(
    fn: Callable,
    compile_kwargs: Optional[Dict[str, Any]] = None,
    dynamo_kwargs: Optional[Dict[str, Any]] = None,
    inductor_kwargs: Optional[Dict[str, Any]] = None,
) -> Callable:
    """Wrap a function with torch.compile and enable debug output.

    Debug config is applied when compilation actually occurs (on first call).
    Caching is disabled to ensure fresh compilation with debug output.

    Args:
        fn: The function to compile.
        compile_kwargs: Dict of arguments passed to torch.compile
            (e.g. mode, fullgraph, dynamic).
        dynamo_kwargs: Dict of extra dynamo config overrides.
        inductor_kwargs: Dict of extra inductor config overrides merged
            with debug defaults.
    """
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
                    "debug": True,
                    "trace.enabled": True,
                    "trace.fx_graph": True,
                    "trace.fx_graph_transformed": True,
                    "trace.output_code": True,
                    "fx_graph_cache": False,
                    "force_disable_caches": True,
                },
            ):
                compiled_fn = torch.compile(fn, **compile_kwargs)
                return compiled_fn(*args, **kwargs)

        return compiled_fn(*args, **kwargs)

    return wrapper


def warmup_compiled_fn(compiled_fn: Callable, *args: Any, **kwargs: Any) -> None:
    """Run warmup iterations on a compiled function and synchronize."""
    for _ in range(10):
        compiled_fn(*args, **kwargs)
    torch.cuda.synchronize()
