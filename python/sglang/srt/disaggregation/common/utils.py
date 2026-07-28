import ctypes
import dataclasses
import struct
import threading
from collections import deque
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

from sglang.srt.observability.trace import (
    TraceNullContext,
    TraceReqContext,
)


@dataclasses.dataclass
class TransferKVChunk:
    """Work unit for KV cache transfer from prefill to decode."""

    room: int
    prefill_kv_indices: npt.NDArray[np.int32]
    index_slice: slice
    is_last_chunk: bool
    prefill_aux_index: Optional[int]
    state_indices: Optional[List]
    chunk_id: Optional[int] = None
    kv_sent: bool = False
    pd_hidden_packet_idx: int = 0
    pd_hidden_sent: bool = False
    pd_hidden_ready_sent: bool = False
    pd_hidden_ack_ready: bool = False
    pd_hidden_ack_expected_count: int = 0
    pd_hidden_ack_timed_out: bool = False
    pd_hidden_start: Optional[int] = None
    pd_hidden_row_len: int = 0
    pd_hidden_is_last_chunk: bool = False
    pd_hidden_release_indices: Optional[List[int]] = None
    enqueue_time: float = 0.0
    source_event: Optional[Any] = None
    trace_ctx: Union[TraceReqContext, TraceNullContext] = dataclasses.field(
        default_factory=TraceNullContext
    )


@dataclasses.dataclass
class PDHiddenChunk:
    """Transport-neutral PD hidden chunk descriptor."""

    room: int
    prefill_rank: int
    hidden_start: int
    row_len: int
    is_last_hidden_chunk: bool
    dst_indices: List[int]
    ack_host: Optional[str] = None
    ack_port: Optional[int] = None


@dataclasses.dataclass
class PDHiddenRequestState:
    """Decode-side request state for hidden transfer, separate from KV status."""

    enabled: bool = False
    streaming: bool = False
    start: int = 0
    next_start: int = 0
    end: int = 0
    hidden_done: bool = True
    kv_done: bool = False

    @classmethod
    def disabled(cls) -> "PDHiddenRequestState":
        return cls()

    @classmethod
    def full(cls, start: int, end: int) -> "PDHiddenRequestState":
        return cls(
            enabled=True,
            streaming=False,
            start=int(start),
            next_start=int(start),
            end=int(end),
            hidden_done=True,
        )

    @classmethod
    def streaming_state(cls, start: int, end: int) -> "PDHiddenRequestState":
        return cls(
            enabled=True,
            streaming=True,
            start=int(start),
            next_start=int(start),
            end=int(end),
            hidden_done=False,
        )

    def reset(self) -> None:
        self.enabled = False
        self.streaming = False
        self.start = 0
        self.next_start = 0
        self.end = 0
        self.hidden_done = True
        self.kv_done = False

    def mark_kv_done(self) -> None:
        self.kv_done = True

    def mark_hidden_done(self) -> None:
        self.hidden_done = True

    def hidden_request_done(self) -> bool:
        return self.hidden_done

    def kv_request_done(self) -> bool:
        return self.kv_done

    def request_done(self) -> bool:
        return self.kv_request_done() and self.hidden_request_done()

    def accept_chunk(
        self, chunk: PDHiddenChunk, *, defer_hidden_done: bool = False
    ) -> str:
        """Return accepted/future/stale for a streaming hidden chunk."""
        hidden_start = int(chunk.hidden_start)
        if hidden_start > self.next_start:
            return "future"
        if hidden_start < self.next_start:
            return "stale"
        next_start = hidden_start + int(chunk.row_len)
        if next_start > self.end:
            raise RuntimeError(
                "PD streaming hidden chunk exceeds request range: "
                f"next_start={next_start}, expected_end={self.end}"
            )
        if chunk.is_last_hidden_chunk:
            if next_start != self.end:
                raise RuntimeError(
                    "PD streaming hidden ended at an unexpected offset: "
                    f"next_start={next_start}, expected_end={self.end}"
                )
            if not defer_hidden_done:
                self.mark_hidden_done()
        self.next_start = next_start
        return "accepted"


def pack_list_of_buffers(buffers: List[bytes]) -> bytes:
    if not buffers:
        return b""
    n = len(buffers)
    header = struct.pack(f"<{n+1}I", n, *(len(b) for b in buffers))
    return header + b"".join(buffers)


def unpack_list_of_buffers(buf: bytes) -> List[bytes]:
    if buf == b"":
        return []
    (n,) = struct.unpack("<I", buf[:4])
    lens = struct.unpack(f"<{n}I", buf[4 : 4 + 4 * n])
    out = []
    offset = 4 + 4 * n
    for length in lens:
        out.append(buf[offset : offset + length])
        offset += length
    return out


def pack_int_lists(lists, fmt: str) -> bytes:
    return pack_list_of_buffers([struct.pack(f"<{len(a)}{fmt}", *a) for a in lists])


def unpack_int_lists(buf: bytes, fmt: str) -> List[List[int]]:
    width = struct.calcsize(fmt)
    return [
        list(struct.unpack(f"<{len(b)//width}{fmt}", b))
        for b in unpack_list_of_buffers(buf)
    ]


class FastQueue:
    def __init__(self):
        self._buf = deque()
        self._cond = threading.Condition()

    def put(self, item):
        with self._cond:
            self._buf.append(item)
            # wake up a thread of wait()
            self._cond.notify()

    def get(self):
        with self._cond:
            # if queue is empty  ,block until is notified()
            while not self._buf:
                self._cond.wait()
            return self._buf.popleft()


class AuxDataCodec:
    """Handles serialization and deserialization of auxiliary data buffers."""

    @staticmethod
    def serialize_data_from_buffer(src_addr, data_length):
        """Serialize data from memory buffer to bytes."""
        buffer = (ctypes.c_byte * data_length).from_address(src_addr)
        return bytes(buffer)

    @staticmethod
    def deserialize_data_to_buffer(kv_args, buffer_index, aux_index, data):
        """Deserialize bytes into target memory buffer."""
        dst_aux_ptr = kv_args.aux_data_ptrs[buffer_index]
        item_len = kv_args.aux_item_lens[buffer_index]
        dst_addr = dst_aux_ptr + item_len * aux_index
        buffer = (ctypes.c_byte * len(data)).from_address(dst_addr)
        buffer[:] = data
        return


def group_concurrent_contiguous(
    src_indices: npt.NDArray[np.int32], dst_indices: npt.NDArray[np.int32]
) -> Tuple[List[npt.NDArray[np.int32]], List[npt.NDArray[np.int32]]]:
    """Vectorised NumPy implementation."""
    # src/dst indices are transferred pairwise, so an empty side means there is
    # nothing to transfer. Guarding both sides (not just src) avoids a cryptic
    # NumPy broadcast error from np.diff() below when only one side is empty, e.g.
    # a non-empty prefill DSA/SWA state list paired with an empty decode registration.
    if src_indices.size == 0 or dst_indices.size == 0:
        return [], []

    if src_indices.size != dst_indices.size:
        raise ValueError(
            "group_concurrent_contiguous requires equal-length src/dst index arrays, "
            f"got {src_indices.size} and {dst_indices.size}"
        )

    brk = np.where((np.diff(src_indices) != 1) | (np.diff(dst_indices) != 1))[0] + 1
    src_groups = np.split(src_indices, brk)
    dst_groups = np.split(dst_indices, brk)

    src_groups = [g.tolist() for g in src_groups]
    dst_groups = [g.tolist() for g in dst_groups]

    return src_groups, dst_groups
