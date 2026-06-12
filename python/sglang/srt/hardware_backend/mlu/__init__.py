"""In-tree Cambricon MLU backend registration."""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def register_mlu_backend() -> None:
    """Initialize MLU backend hooks that are not registered elsewhere."""
    pass
