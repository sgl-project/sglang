"""Per-forward-call control configs.

Holds state that varies per forward call (vs. per process, which lives in
``pool_context.py``). Today carries ``attn_backend``; future fields include
DP-attention buffer sizes, TBO child indices, hybrid routing flags, etc.

Usage:

- ``ModelRunner._forward_raw`` enters a ``forward_context(...)`` block at the
  start of every forward, populating the active ``AttentionBackend``.
- Override sites (PDmux per-stream backend swap, Eagle/Frozen-KV MTP draft
  loops, draft-extend cuda-graph capture) wrap their forward dispatch in
  ``forward_context(replace(get_forward_context(), attn_backend=...))`` to
  publish a different backend just for that scope.
- Model-layer code reads via ``get_attn_backend()`` (sugar on top of
  ``get_forward_context().attn_backend``).

No locking: all runners share one process, forward calls are synchronous, so
the global is mutated and read serially.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend


@dataclass(frozen=True, slots=True)
class ForwardContext:
    """Per-forward-call control configs.

    Read at depth via ``get_forward_context()``; never threaded through
    ``forward_batch``. Extend by adding fields here — no new globals needed.
    """

    attn_backend: "AttentionBackend"


_current: Optional[ForwardContext] = None


def set_forward_context(ctx: Optional[ForwardContext]) -> Optional[ForwardContext]:
    """Set active context; return previous for explicit save/restore."""
    global _current
    prev, _current = _current, ctx
    return prev


def get_forward_context() -> ForwardContext:
    assert _current is not None, "no forward context active"
    return _current


def get_attn_backend() -> "AttentionBackend":
    return get_forward_context().attn_backend


@contextmanager
def forward_context(ctx: ForwardContext):
    """Scoped publication. Save the previous context, install ``ctx``, run
    the suite, then restore the previous context on exit (even on exception).
    """
    prev = set_forward_context(ctx)
    try:
        yield
    finally:
        set_forward_context(prev)
