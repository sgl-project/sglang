"""Hands aiter's split-MLA stage-1 partials from the attention backend to the
model's absorbed-V epilogue, which combines them together with W_vc and the
output gate (kernels/ops/attention/mla_decode_epilogue.py) instead of the
backend's own split reduce.

The model opens a window around one attention call (``expecting``) only when
its epilogue will run; the backend defers inside that window, and the
epilogue takes the deferred reduce or finishes it the backend's way.
"""

from __future__ import annotations

import contextlib
from typing import Iterator, Optional

import msgspec
import torch


class DeferredMlaReduce(msgspec.Struct, frozen=True):
    # aiter persistent stage 1: [P * max_q, 1, HP, L] fp32 partials and their LSE.
    partial_output: torch.Tensor
    partial_lse: torch.Tensor
    reduce_indptr: torch.Tensor
    reduce_final_map: torch.Tensor
    reduce_partial_map: torch.Tensor
    max_q: int
    # [S, HP, L]; the rows no reduce tile covers are already final.
    o: torch.Tensor
    num_heads: int
    max_splits: int

    def finish(self) -> None:
        """The backend's own split reduce into o, for an epilogue that cannot
        combine the partials itself."""
        from sglang.kernels.ops.attention import mla_split_reduce

        mla_split_reduce.mla_split_reduce(
            self.partial_output,
            self.partial_lse,
            self.reduce_indptr,
            self.reduce_final_map,
            self.reduce_partial_map,
            self.max_q,
            self.o,
            self.num_heads,
        )


class _Handoff(msgspec.Struct):
    # the RadixAttention whose split reduce may be deferred, inside expecting()
    expected: Optional[object] = None
    pending: Optional[DeferredMlaReduce] = None


_handoff = _Handoff()


@contextlib.contextmanager
def expecting(layer: Optional[object]) -> Iterator[None]:
    """Let the attention call of ``layer`` (a RadixAttention, or None for no
    deferral) defer its split reduce to the epilogue that follows it."""
    _handoff.expected = layer
    try:
        yield
    finally:
        _handoff.expected = None
        leftover, _handoff.pending = _handoff.pending, None
    if leftover is not None:
        raise RuntimeError("a deferred MLA split reduce was not taken by its epilogue")


def defers(layer: object) -> bool:
    return _handoff.expected is not None and layer is _handoff.expected


def defer(reduce: DeferredMlaReduce) -> None:
    assert _handoff.pending is None, "an MLA split reduce is already deferred"
    _handoff.pending = reduce


def take(attn_output: torch.Tensor) -> Optional[DeferredMlaReduce]:
    """The deferred reduce whose output buffer attn_output views, or None."""
    reduce = _handoff.pending
    if reduce is None or reduce.o.data_ptr() != attn_output.data_ptr():
        return None
    _handoff.pending = None
    return reduce
