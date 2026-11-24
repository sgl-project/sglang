import ctypes
import dataclasses
import struct
import threading
from collections import deque
from typing import List, Optional, Tuple, Union

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
    num_kv_tokens: Optional[int] = None
    trace_ctx: Union[TraceReqContext, TraceNullContext] = dataclasses.field(
        default_factory=TraceNullContext
    )
    # Set when the staging worker first counts this chunk toward the per-room
    # outstanding count; stays set across re-enqueue on a watermark defer.
    staging_counted: bool = False
    # Early-send: GPU event to synchronize before reading KV (optional).
    wait_event: Optional[object] = None
    # Must complete before reading asynchronously copied host indices.
    indices_event: Optional[object] = None
    # The room's send record this chunk was counted in when queued (a
    # `SendRecord`); the worker reports the chunk finished to it only.
    send_record: Optional[object] = None


def bind_thread_to_device(gpu_id: int) -> None:
    """Make ``gpu_id`` the calling thread's current device.

    New threads default to device 0; binding keeps a transfer worker's device
    calls on this rank's GPU instead of opening a context on device 0.
    """
    from sglang.srt.platforms import current_platform

    try:
        current_platform.set_device(current_platform.get_device(gpu_id))
    except NotImplementedError:
        pass


def run_on_device(gpu_id: int, fn, *args) -> None:
    """Thread target: bind the thread to ``gpu_id``, then run ``fn(*args)``."""
    bind_thread_to_device(gpu_id)
    fn(*args)


def pack_list_of_buffers(buffers: List[bytes]) -> bytes:
    if not buffers:
        return b""
    n = len(buffers)
    header = struct.pack(f"<{n + 1}I", n, *(len(b) for b in buffers))
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
        list(struct.unpack(f"<{len(b) // width}{fmt}", b))
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
        # One memcpy through the buffer protocol. Assigning a ctypes slice converts every
        # byte to a Python int while holding the GIL (~14 ms per MiB), which stalls the
        # scheduler thread of the receiving decode rank.
        memoryview(buffer).cast("B")[:] = data
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


@dataclasses.dataclass(frozen=True)
class DCPTokenTransferPlan:
    target_src_token_indices: npt.NDArray[np.int64]
    target_dst_token_indices: npt.NDArray[np.int64]
    draft_src_token_indices: npt.NDArray[np.int64]
    draft_dst_token_indices: npt.NDArray[np.int64]

    def empty(self) -> bool:
        return (
            self.target_src_token_indices.size == 0
            and self.draft_src_token_indices.size == 0
        )


def build_dcp_token_transfer_plan(
    src_page_indices: npt.NDArray[np.int32],
    dst_page_indices: npt.NDArray[np.int32],
    *,
    physical_page_size: int,
    dcp_size: int,
    dcp_rank: int,
    src_page_offset: int = 0,
    decode_prefix_len: int = 0,
    num_kv_tokens: Optional[int] = None,
    draft_rows_collapsed: bool = False,
    src_dcp_size: int = 1,
    src_dcp_rank: int = 0,
    src_draft_rows_collapsed: bool = False,
    send_replicated_draft: bool = True,
) -> DCPTokenTransferPlan:
    if src_dcp_size < 1 or dcp_size < 1:
        raise ValueError(
            f"PD DCP sizes must be >= 1, got src={src_dcp_size}, dst={dcp_size}"
        )
    if not (0 <= src_dcp_rank < src_dcp_size) or not (0 <= dcp_rank < dcp_size):
        raise ValueError(
            f"PD DCP rank out of range: src={src_dcp_rank}/{src_dcp_size}, "
            f"dst={dcp_rank}/{dcp_size}"
        )
    if dcp_size % src_dcp_size != 0 and src_dcp_size % dcp_size != 0:
        raise ValueError(
            f"PD DCP sizes must divide each other, got src={src_dcp_size}, dst={dcp_size}"
        )

    src_pages = np.asarray(src_page_indices, dtype=np.int64)
    dst_pages = np.asarray(dst_page_indices, dtype=np.int64)
    src_virtual_page_size = physical_page_size * src_dcp_size
    dst_virtual_page_size = physical_page_size * dcp_size
    prefix_align = max(src_virtual_page_size, dst_virtual_page_size)
    if decode_prefix_len % prefix_align != 0:
        raise ValueError(
            "PD DCP transfer requires decode_prefix_len to align to the virtual "
            f"DCP page size ({prefix_align}), got {decode_prefix_len}"
        )
    if num_kv_tokens is None:
        num_kv_tokens = src_pages.size * src_virtual_page_size
    empty = np.empty((0,), dtype=np.int64)
    if num_kv_tokens == 0:
        return DCPTokenTransferPlan(empty, empty.copy(), empty.copy(), empty.copy())

    def rows(
        pages: npt.NDArray[np.int64], page_size: int, local: npt.NDArray[np.int64]
    ):
        if local.size == 0:
            return empty.copy()
        return pages[local // page_size] * page_size + local % page_size

    chunk_token_offset = src_page_offset * src_virtual_page_size
    chunk_start = decode_prefix_len + chunk_token_offset
    gcd_dcp = min(src_dcp_size, dcp_size)
    if src_dcp_rank % gcd_dcp != dcp_rank % gcd_dcp:
        target_offsets = empty
    else:
        stride = max(src_dcp_size, dcp_size)
        owner_mod = dcp_rank if dcp_size >= src_dcp_size else src_dcp_rank
        target_offsets = np.arange(
            (owner_mod - chunk_start) % stride,
            num_kv_tokens,
            stride,
            dtype=np.int64,
        )

    target_src = rows(src_pages, physical_page_size, target_offsets // src_dcp_size)
    target_dst = rows(
        dst_pages, physical_page_size, (chunk_token_offset + target_offsets) // dcp_size
    )

    if src_draft_rows_collapsed and draft_rows_collapsed:
        return DCPTokenTransferPlan(
            target_src, target_dst, target_src.copy(), target_dst.copy()
        )
    if draft_rows_collapsed:
        draft_src = rows(src_pages, src_virtual_page_size, target_offsets)
        return DCPTokenTransferPlan(
            target_src, target_dst, draft_src, target_dst.copy()
        )
    if src_draft_rows_collapsed:
        draft_dst = rows(
            dst_pages, dst_virtual_page_size, chunk_token_offset + target_offsets
        )
        return DCPTokenTransferPlan(
            target_src, target_dst, target_src.copy(), draft_dst
        )
    if not send_replicated_draft:
        return DCPTokenTransferPlan(target_src, target_dst, empty.copy(), empty.copy())
    draft_offsets = np.arange(num_kv_tokens, dtype=np.int64)
    draft_src = rows(src_pages, src_virtual_page_size, draft_offsets)
    draft_dst = rows(
        dst_pages, dst_virtual_page_size, chunk_token_offset + draft_offsets
    )
    return DCPTokenTransferPlan(target_src, target_dst, draft_src, draft_dst)
