"""Process-wide so it need not thread through the autograd entry points. Set
before the first kernel compile; it keys the forward compile cache.
"""

from __future__ import annotations

_batch_invariant = False


def set_batch_invariant(enabled: bool) -> None:
    global _batch_invariant
    _batch_invariant = bool(enabled)


def is_batch_invariant() -> bool:
    return _batch_invariant
