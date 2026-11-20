"""Host-side transport of captured output-token sampling rows."""

import pickle
import sys
from array import array
from typing import List, Optional, Tuple

import msgspec
import numpy as np


def returns_sampling_output(request) -> bool:
    """Whether a request asked for sampling logprobs or for the sampling mask."""
    return bool(request.return_sampling_logprobs) or bool(request.return_sampling_mask)


class SamplingMaskChunk(
    msgspec.Struct, frozen=True, kw_only=True, array_like=True, gc=False
):
    """One request's captured sampling rows in one output batch, one row per output token."""

    # float32 [num_rows], the sampled token's logprob
    selected_logprobs: np.ndarray
    # int32 [num_rows], 0 for a request that did not ask for the mask
    lengths: np.ndarray
    # int32 [sum(lengths)], row after row
    token_ids: np.ndarray
    # float32 [sum(lengths)], aligned with token_ids
    support_logprobs: np.ndarray

    def __reduce_ex__(self, protocol: int):
        # Read-only buffers unpickle as bytes, which np.frombuffer wraps without the
        # GC-tracked objects that numpy's reducer or a bytearray would leave per array.
        if protocol < 5:
            return super().__reduce_ex__(protocol)
        return (
            _chunk_from_buffers,
            tuple(
                pickle.PickleBuffer(memoryview(values).toreadonly())
                for values in (
                    self.selected_logprobs,
                    self.lengths,
                    self.token_ids,
                    self.support_logprobs,
                )
            ),
        )

    def support_rows(self) -> Tuple[List[List[int]], List[List[float]]]:
        """Each row's support token IDs and their logprobs."""
        token_ids: List[List[int]] = []
        logprobs: List[List[float]] = []
        start = 0
        for length in self.lengths.tolist():
            end = start + length
            token_ids.append(self.token_ids[start:end].tolist())
            logprobs.append(self.support_logprobs[start:end].tolist())
            start = end
        return token_ids, logprobs


def _chunk_from_buffers(
    selected_logprobs, lengths, token_ids, support_logprobs
) -> SamplingMaskChunk:
    return SamplingMaskChunk(
        selected_logprobs=np.frombuffer(selected_logprobs, dtype=np.float32),
        lengths=np.frombuffer(lengths, dtype=np.int32),
        token_ids=np.frombuffer(token_ids, dtype=np.int32),
        support_logprobs=np.frombuffer(support_logprobs, dtype=np.float32),
    )


class _GrowableArray(msgspec.Struct):
    """Append-only 1-D buffer; doubling capacity copies each value O(1) times.

    ``reset`` keeps the buffer as a spare, reused once no view of it is alive,
    so steady-state appends touch no new pages.
    """

    buffer: np.ndarray
    size: int = 0
    spare: Optional[np.ndarray] = None

    def extend(self, values: np.ndarray) -> None:
        end = self.size + len(values)
        if end > len(self.buffer):
            self.buffer = self._grow(end)
        self.buffer[self.size : end] = values
        self.size = end

    def view(self) -> np.ndarray:
        return self.buffer[: self.size]

    def reset(self) -> None:
        self.spare = self.buffer
        self.buffer = np.empty(0, dtype=self.buffer.dtype)
        self.size = 0

    def _grow(self, end: int) -> np.ndarray:
        spare, self.spare = self.spare, None
        # Two references, the local and the argument, mean no view is alive.
        if spare is not None and len(spare) >= end and sys.getrefcount(spare) == 2:
            grown = spare
        else:
            grown = np.empty(max(end, 2 * len(self.buffer)), dtype=self.buffer.dtype)
        grown[: self.size] = self.buffer[: self.size]
        return grown


class SamplingMaskRows(msgspec.Struct):
    """One request's SamplingMaskChunk rows, queued until they are streamed."""

    _selected_logprobs: array = msgspec.field(default_factory=lambda: array("f"))
    _lengths: array = msgspec.field(default_factory=lambda: array("i"))
    _token_ids: _GrowableArray = msgspec.field(
        default_factory=lambda: _GrowableArray(np.empty(0, dtype=np.int32))
    )
    _support_logprobs: _GrowableArray = msgspec.field(
        default_factory=lambda: _GrowableArray(np.empty(0, dtype=np.float32))
    )

    def append(
        self,
        selected_logprob: float,
        token_ids: np.ndarray,
        support_logprobs: np.ndarray,
    ) -> None:
        """Queue one token's sampled logprob, int32 support, and float32 support logprobs."""
        self._selected_logprobs.append(selected_logprob)
        self._lengths.append(len(token_ids))
        self._token_ids.extend(token_ids)
        self._support_logprobs.extend(support_logprobs)

    def view(self) -> SamplingMaskChunk:
        """View the queued support IDs and logprobs without copying them."""
        return SamplingMaskChunk(
            selected_logprobs=np.array(self._selected_logprobs, dtype=np.float32),
            lengths=np.array(self._lengths, dtype=np.int32),
            token_ids=self._token_ids.view(),
            support_logprobs=self._support_logprobs.view(),
        )

    def take(self) -> SamplingMaskChunk:
        """Hand off the queued rows without copying them, leaving the queue empty."""
        chunk = self.view()
        del self._selected_logprobs[:]
        del self._lengths[:]
        self._token_ids.reset()
        self._support_logprobs.reset()
        return chunk
