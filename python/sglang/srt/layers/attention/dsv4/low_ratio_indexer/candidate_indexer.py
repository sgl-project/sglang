"""The two-level candidate scheme of the DeepSeek V4.1 ratio-1/2 index layers.

The candidate source publishes the blocks of compressed positions it kept, and
the index layers after it select their top-k among those blocks only. What a
publish carries -- a DeepGEMM sparse table, block ids, a mask -- is the backend's
own; the attention backend keeps it alive on the forward metadata and hands it
back to the consumers unread.
"""

from __future__ import annotations

from typing import Optional, Protocol, TypeVar

from .inputs import (
    CandidateMetadata,
    DecodeInputs,
    PrefillInputs,
    Selection,
)

T = TypeVar("T", bound=CandidateMetadata)


class PrefillBackend(Protocol[T]):
    def publish_prefill(
        self,
        inputs: PrefillInputs,
        out: Selection,
    ) -> Optional[T]: ...

    def consume_prefill(
        self,
        inputs: PrefillInputs,
        published: Optional[T],
        out: Selection,
    ) -> None: ...


class DecodeBackend(Protocol[T]):
    def publish_decode(
        self,
        inputs: DecodeInputs,
        out: Selection,
    ) -> Optional[T]: ...

    def consume_decode(
        self,
        inputs: DecodeInputs,
        published: Optional[T],
        out: Selection,
    ) -> None: ...


class CandidateIndexer:
    def __init__(self, *, prefill: PrefillBackend, decode: DecodeBackend):
        self.prefill = prefill
        self.decode = decode

    def publish_prefill(
        self,
        inputs: PrefillInputs,
        out: Selection,
    ) -> Optional[CandidateMetadata]:
        return self.prefill.publish_prefill(inputs, out)

    def consume_prefill(
        self,
        inputs: PrefillInputs,
        published: Optional[CandidateMetadata],
        out: Selection,
    ) -> None:
        self.prefill.consume_prefill(inputs, published, out)

    def publish_decode(
        self,
        inputs: DecodeInputs,
        out: Selection,
    ) -> Optional[CandidateMetadata]:
        return self.decode.publish_decode(inputs, out)

    def consume_decode(
        self,
        inputs: DecodeInputs,
        published: Optional[CandidateMetadata],
        out: Selection,
    ) -> None:
        self.decode.consume_decode(inputs, published, out)
