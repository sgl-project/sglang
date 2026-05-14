"""Shared graph memory pool used by the speculative-draft cuda graph
runners. The new ``DecodeCudaGraphRunner`` and ``PrefillCudaGraphRunner``
backends each own their pool internally; this global is retained for the
EAGLE / multi-step draft runners that haven't been folded into the new
backend interface.
"""

from __future__ import annotations

from typing import Any, Optional

_global_graph_memory_pool: Optional[Any] = None


def get_global_graph_memory_pool() -> Optional[Any]:
    return _global_graph_memory_pool


def set_global_graph_memory_pool(val: Any) -> None:
    global _global_graph_memory_pool
    _global_graph_memory_pool = val
