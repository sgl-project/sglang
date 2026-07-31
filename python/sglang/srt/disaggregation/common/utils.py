import concurrent.futures
import ctypes
import dataclasses
import struct
import threading
from collections import deque
from typing import Callable, List, Optional, Sequence, Tuple, Union

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
    # Identity of the request that queued this chunk, for backends that must not
    # let a recycled bootstrap_room attach queued work to a new request. Opaque
    # to the queue; see MooncakeKVManager.try_lease_chunk.
    owner: Optional[object] = None
    trace_ctx: Union[TraceReqContext, TraceNullContext] = dataclasses.field(
        default_factory=TraceNullContext
    )


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


def drain_transfer_futures(futures: Sequence[concurrent.futures.Future]) -> int:
    """Wait for every transfer future to leave the running state.

    Returns the first non-zero transfer status, or re-raises the first
    exception. Unlike a plain ``as_completed`` loop that returns on the first
    error, this never returns while a sibling future may still be reading or
    writing KV pages: ``Future.cancel()`` is a no-op once a future is running,
    so returning early would let the caller release pages that are still being
    transferred.
    """
    first_status = 0
    first_exception = None
    for future in concurrent.futures.as_completed(futures):
        try:
            status = future.result()
        except concurrent.futures.CancelledError:
            continue
        except BaseException as e:  # noqa: BLE001 - re-raised after draining
            if first_exception is None:
                first_exception = e
                _cancel_pending(futures)
            continue
        if status != 0 and first_status == 0:
            first_status = status
            _cancel_pending(futures)
    # as_completed() already yielded every future, so this only reaps the ones
    # cancelled above; it is what makes the "no work in flight" guarantee hold.
    concurrent.futures.wait(futures)
    if first_exception is not None:
        raise first_exception
    return first_status


def submit_transfer_calls(
    executor: concurrent.futures.Executor,
    calls: Sequence[Tuple[Callable[..., int], tuple]],
) -> int:
    """Submit transfer work and drain it via ``drain_transfer_futures``.

    A failure part-way through submission still drains the futures that were
    accepted, so the caller never regains KV page ownership early.
    """
    futures: List[concurrent.futures.Future] = []
    try:
        for fn, args in calls:
            futures.append(executor.submit(fn, *args))
    except BaseException:
        _cancel_pending(futures)
        concurrent.futures.wait(futures)
        raise
    return drain_transfer_futures(futures)


def _cancel_pending(futures: Sequence[concurrent.futures.Future]) -> None:
    for future in futures:
        future.cancel()
