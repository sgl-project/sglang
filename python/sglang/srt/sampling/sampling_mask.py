"""Host-side transport of output-token sampling masks."""

import pickle
import sys
from array import array
from typing import List, Optional, Tuple, Union

import msgspec
import numpy as np


class SamplingMaskChunk(
    msgspec.Struct, frozen=True, kw_only=True, array_like=True, gc=False
):
    """One request's sampling masks in one output batch, one row per output token."""

    # int32 [num_rows]
    lengths: np.ndarray
    # int32 [sum(lengths)], row after row
    token_ids: np.ndarray
    # float32 [sum(lengths)] in support mode, [num_rows] in selected mode
    logprobs: np.ndarray

    def __reduce_ex__(self, protocol: int):
        # Read-only buffers unpickle as bytes, which np.frombuffer wraps without the
        # GC-tracked objects that numpy's reducer or a bytearray would leave per array.
        if protocol < 5:
            return super().__reduce_ex__(protocol)
        return (
            _chunk_from_buffers,
            tuple(
                pickle.PickleBuffer(memoryview(values).toreadonly())
                for values in (self.lengths, self.token_ids, self.logprobs)
            ),
        )

    def to_lists(
        self, support_logprobs: bool
    ) -> Tuple[List[List[int]], List[Union[float, List[float]]]]:
        masks: List[List[int]] = []
        values: List[Union[float, List[float]]] = (
            [] if support_logprobs else self.logprobs.tolist()
        )
        start = 0
        for length in self.lengths.tolist():
            end = start + length
            masks.append(self.token_ids[start:end].tolist())
            if support_logprobs:
                values.append(self.logprobs[start:end].tolist())
            start = end
        return masks, values


def _chunk_from_buffers(lengths, token_ids, logprobs) -> SamplingMaskChunk:
    return SamplingMaskChunk(
        lengths=np.frombuffer(lengths, dtype=np.int32),
        token_ids=np.frombuffer(token_ids, dtype=np.int32),
        logprobs=np.frombuffer(logprobs, dtype=np.float32),
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

    _lengths: array = msgspec.field(default_factory=lambda: array("i"))
    _token_ids: _GrowableArray = msgspec.field(
        default_factory=lambda: _GrowableArray(np.empty(0, dtype=np.int32))
    )
    _logprobs: _GrowableArray = msgspec.field(
        default_factory=lambda: _GrowableArray(np.empty(0, dtype=np.float32))
    )

    def append(self, token_ids: np.ndarray, logprobs: np.ndarray) -> None:
        """Queue one token's int32 support and float32 logprobs."""
        self._lengths.append(len(token_ids))
        self._token_ids.extend(token_ids)
        self._logprobs.extend(logprobs)

    def view(self) -> SamplingMaskChunk:
        """View the queued token IDs and logprobs without copying them."""
        return SamplingMaskChunk(
            lengths=np.array(self._lengths, dtype=np.int32),
            token_ids=self._token_ids.view(),
            logprobs=self._logprobs.view(),
        )

    def take(self) -> SamplingMaskChunk:
        """Hand off the queued rows without copying them, leaving the queue empty."""
        chunk = self.view()
        del self._lengths[:]
        self._token_ids.reset()
        self._logprobs.reset()
        return chunk
