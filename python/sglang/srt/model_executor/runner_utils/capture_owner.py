from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator, Optional

_full_cuda_graph_owners: ContextVar[Optional[list[Any]]] = ContextVar(
    "full_cuda_graph_owners", default=None
)


@contextmanager
def collect_full_cuda_graph_owners() -> Iterator[list[Any]]:
    """Collect tensor owners whose addresses are recorded by one full graph."""

    owners: list[Any] = []
    token = _full_cuda_graph_owners.set(owners)
    try:
        yield owners
    finally:
        _full_cuda_graph_owners.reset(token)


def retain_full_cuda_graph_owner(owner: Any) -> bool:
    """Retain ``owner`` when called from an active full-graph capture."""

    owners = _full_cuda_graph_owners.get()
    if owners is None:
        return False
    owners.append(owner)
    return True
