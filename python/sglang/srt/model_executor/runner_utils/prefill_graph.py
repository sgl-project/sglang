"""Runner-owned prefill graph policy, independent of the attention backend."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class _PrefillGraphState:
    full_graph: bool
    raw_num_tokens: int | None


_current: ContextVar[_PrefillGraphState | None] = ContextVar(
    "prefill_graph", default=None
)


def is_in_prefill_graph_scope() -> bool:
    """Whether the prefill runner owns this execution, regardless of graph backend."""
    # BCG/full prefill runs without torch.compile. Older supported PyTorch
    # builds cannot trace ContextVar.get(), including during decode/verify.
    if torch.compiler.is_compiling():
        return False
    return _current.get() is not None


def is_in_full_prefill_graph() -> bool:
    if torch.compiler.is_compiling():
        return False
    state = _current.get()
    return state is not None and state.full_graph


def get_prefill_raw_num_tokens() -> int | None:
    state = _current.get()
    return state.raw_num_tokens if state is not None else None


@contextmanager
def prefill_graph_scope(*, full_graph: bool, raw_num_tokens: int | None = None):
    reset_token = _current.set(_PrefillGraphState(full_graph, raw_num_tokens))
    try:
        yield
    finally:
        _current.reset(reset_token)
