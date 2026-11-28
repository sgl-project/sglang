"""Runner-owned prefill graph policy, independent of the attention backend."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING

import msgspec
import torch

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


@dataclass(frozen=True)
class _PrefillGraphState:
    full_graph: bool
    raw_num_tokens: int | None
    zero_padded_tail: bool


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


def should_zero_prefill_padding() -> bool:
    state = _current.get()
    return state is None or state.zero_padded_tail


@contextmanager
def prefill_graph_scope(
    *,
    full_graph: bool,
    raw_num_tokens: int | None = None,
    zero_padded_tail: bool = True,
):
    reset_token = _current.set(
        _PrefillGraphState(full_graph, raw_num_tokens, zero_padded_tail)
    )
    try:
        yield
    finally:
        _current.reset(reset_token)


class PrefillAttentionState(msgspec.Struct, frozen=True):
    """ForwardBatch fields an eager attention break reads at replay.

    Model code writes them before each attention call, but that code is inside
    the captured segments and does not rerun at replay.
    """

    mha_one_shot: bool | None = None
    mha_return_lse: bool | None = None
    attn_attend_prefix_cache: bool | None = None

    @classmethod
    def of(cls, forward_batch: "ForwardBatch") -> "PrefillAttentionState":
        return cls(
            mha_one_shot=forward_batch.mha_one_shot,
            mha_return_lse=forward_batch.mha_return_lse,
            attn_attend_prefix_cache=forward_batch.attn_attend_prefix_cache,
        )

    def apply(self, forward_batch: "ForwardBatch") -> None:
        forward_batch.mha_one_shot = self.mha_one_shot
        forward_batch.mha_return_lse = self.mha_return_lse
        forward_batch.set_attn_attend_prefix_cache(self.attn_attend_prefix_cache)


class PrefillAttentionBreaks(msgspec.Struct):
    """Distinct batch states and attention layers seen at eager breaks."""

    states: set[PrefillAttentionState] = set()
    layers: set[object] = set()


_attention_breaks: ContextVar[PrefillAttentionBreaks | None] = ContextVar(
    "prefill_attention_breaks", default=None
)


@contextmanager
def record_prefill_attention_breaks() -> Iterator[PrefillAttentionBreaks]:
    breaks = PrefillAttentionBreaks()
    reset_token = _attention_breaks.set(breaks)
    try:
        yield breaks
    finally:
        _attention_breaks.reset(reset_token)


def note_prefill_attention_break(layer: object, forward_batch: "ForwardBatch") -> None:
    breaks = _attention_breaks.get()
    if breaks is not None:
        breaks.states.add(PrefillAttentionState.of(forward_batch))
        breaks.layers.add(layer)
