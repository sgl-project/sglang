# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Zero-SM copy-engine KV prefetch for Dense MLA Decode Context Parallel (DCP).

When ``SGLANG_ENABLE_DCP_KV_PREFETCH=1`` is enabled on ``dcp_size > 1``, this
module overlaps historical prefix KV all-gathers behind compute across all
three Dense MLA extend paths (``MHA_CHUNKED_KV``, ``MHA_ONE_SHOT``, and
absorbed ``MLA``):

  1. ``_pack_dcp_kv_kernel`` packs rank-local prefix rows from the strided MLA
     KV pool into double-buffered symmetric ``own_wire[parity]`` in raw
     KV-cache wire bytes (preserving 50% NVLink savings on FP8 KV caches).
  2. Alternating two-channel symmetric-memory barriers (one barrier per
     layer/chunk on ``channel = parity``) with device-side
     ``_plan_check_kernel`` guarantee both write-read and slot-reuse ordering
     with zero decode overhead.
  3. Dedicated ``copy_stream`` pulls all ``world_size - 1`` remote shards in
     conflict-free staggered ring order via ``cuMemcpyBatchAsync_v2`` (0 SMs).
  4. ``_reorg_dcp_kv_kernel`` reads own rows directly from ``own_wire`` (zero
     local self-copy) and remote rows from ``peer_stage``, casts FP8 on store
     when needed, and writes directly into a single L2-resident output buffer
     (fenced via ``fence_ev`` before ``pack`` once main finishes consuming the
     previous chunk/layer, saving 144 MiB per GPU and preventing SM contention
     with persistent attention kernels).
"""

from __future__ import annotations

import ctypes
import functools
import hashlib
import logging
import struct
import weakref
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import torch
import torch.distributed as dist
import triton
import triton.language as tl

try:
    import torch.distributed._symmetric_memory as symm_mem
except ImportError:
    symm_mem = None

logger = logging.getLogger(__name__)

BARRIER_TIMEOUT_MS = 60_000
PLAN_SLOTS = 64

_PLAN_DOMAIN_TAG = 0x53474450  # "SGDP"
_MODE_CODE = {"mha_chunked": 1, "mha_one_shot": 2, "mla_absorb": 3}
_DTYPE_CODE = {
    torch.bfloat16: 1,
    torch.float16: 2,
    torch.float8_e4m3fn: 3,
    torch.float8_e5m2: 4,
}


# ---------------------------------------------------------------------------
# Zero-SM batched P2P copy helper (cuMemcpyBatchAsync_v2 + cuMemcpyAsync).
# ---------------------------------------------------------------------------


class _MemLocation(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class _MemcpyAttributes(ctypes.Structure):
    _fields_ = [
        ("src_access_order", ctypes.c_int),
        ("src_loc_hint", _MemLocation),
        ("dst_loc_hint", _MemLocation),
        ("flags", ctypes.c_uint),
    ]


class CopyList:
    """``(dst, src, nbytes)`` device copy list with prebuilt driver arrays."""

    __slots__ = ("copies", "count", "destinations", "sources", "sizes", "nbytes")

    def __init__(self, copies: Sequence[tuple[int, int, int]]) -> None:
        self.copies = [copy for copy in copies if copy[2] > 0]
        self.count = count = len(self.copies)
        self.nbytes = sum(size for _, _, size in self.copies)
        dsts, srcs, sizes = zip(*self.copies) if count else ((), (), ())
        self.destinations = (ctypes.c_uint64 * count)(*dsts)
        self.sources = (ctypes.c_uint64 * count)(*srcs)
        self.sizes = (ctypes.c_size_t * count)(*sizes)


class MemcpyDriver:
    """Minimal CUDA Driver wrapper for zero-SM batched device copies."""

    SRC_ACCESS_ORDER_STREAM = 0x1

    def __init__(self) -> None:
        driver = ctypes.CDLL("libcuda.so.1")
        handle, u64, size = ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t
        self._memcpy = driver.cuMemcpyAsync
        self._memcpy.argtypes = [u64, u64, size, handle]
        self._memcpy.restype = ctypes.c_int

        self._memcpy_batch = getattr(driver, "cuMemcpyBatchAsync_v2", None)
        if self._memcpy_batch is not None:
            self._memcpy_batch.argtypes = [
                handle,
                handle,
                handle,
                size,
                handle,
                handle,
                size,
                handle,
            ]
            self._memcpy_batch.restype = ctypes.c_int
            self._attributes = _MemcpyAttributes(self.SRC_ACCESS_ORDER_STREAM)
            self._attributes_ref = ctypes.byref(self._attributes)
            self._attribute_index = (size * 1)(0)

    def memcpy_list(self, copies: CopyList, stream: int) -> None:
        count = copies.count
        if count == 0:
            return
        if count > 1 and self._memcpy_batch is not None:
            rc = self._memcpy_batch(
                copies.destinations,
                copies.sources,
                copies.sizes,
                count,
                self._attributes_ref,
                self._attribute_index,
                1,
                stream,
            )
            if rc == 0:
                return
            self._memcpy_batch = None
        for dst, src, nbytes in copies.copies:
            rc = self._memcpy(dst, src, nbytes, stream)
            if rc != 0:
                raise RuntimeError(f"cuMemcpyAsync failed with error {rc}")


@functools.cache
def _memcpy_driver() -> MemcpyDriver:
    return MemcpyDriver()


# ---------------------------------------------------------------------------
# Pure layout, plan-hash, ring-schedule, and layer-order helpers (CPU-testable).
# ---------------------------------------------------------------------------


def staggered_pull_order(world_size: int, rank: int) -> list[int]:
    """Conflict-free staggered ring pull order across remote DCP peers."""
    return [(rank + step) % world_size for step in range(1, world_size)]


def plan_hash(values: Sequence[int]) -> int:
    """Return signed 64-bit blake2b digest of integer plan parameters."""
    packed = struct.pack(f"<{len(values)}q", *values)
    (word,) = struct.unpack("<q", hashlib.blake2b(packed, digest_size=8).digest())
    return word


@dataclass(frozen=True)
class DcpChunkLayout:
    """Per-chunk token-interleaved DCP pack and reorg layout (interleave_size=1)."""

    total_out: int
    total_local: int
    total_padded: int
    pack_single_span: bool
    pack_single_dst_base: int
    pack_local_cu: tuple[int, ...]
    pack_dst_base: tuple[int, ...]
    reorg_single_req: bool
    reorg_single_start_mod: int
    reorg_out_cu: tuple[int, ...]
    reorg_pad_base: tuple[int, ...]
    reorg_start_mod: tuple[int, ...]


def plan_dcp_chunk_layout(
    seq_lens: Sequence[int],
    starts: Optional[Sequence[int]],
    world_size: int,
    rank: int,
) -> DcpChunkLayout:
    """Compute rank-invariant padded wire layout and per-rank pack/reorg spans."""
    starts_seq: Sequence[int] = [0] * len(seq_lens) if starts is None else starts
    total_out = total_local = total_padded = 0
    all_aligned = True
    pack_local_cu, pack_dst_base = [0], []
    reorg_out_cu, reorg_pad_base, reorg_start_mod = [0], [], []

    for length_raw, start_raw in zip(seq_lens, starts_seq):
        length = int(length_raw)
        if length <= 0:
            continue
        start = int(start_raw)
        start_mod = start % world_size
        end_mod = (start + length - 1) % world_size
        left_pad = int(start_mod > rank)
        right_pad = int(end_mod < rank)
        padded = (length + start_mod + world_size - 1) // world_size
        local = padded - left_pad - right_pad
        if left_pad != 0 or local != padded:
            all_aligned = False

        reorg_pad_base.append(total_padded)
        reorg_start_mod.append(start_mod)
        total_out += length
        reorg_out_cu.append(total_out)
        if local > 0:
            pack_dst_base.append(total_padded + left_pad)
            total_local += local
            pack_local_cu.append(total_local)
        total_padded += padded

    n_pack = len(pack_dst_base)
    if all_aligned or n_pack == 0:
        pack_single_span, pack_single_dst_base = True, 0
    elif n_pack == 1:
        pack_single_span, pack_single_dst_base = True, pack_dst_base[0]
    else:
        pack_single_span, pack_single_dst_base = False, 0

    reorg_single_req = len(reorg_pad_base) == 1
    return DcpChunkLayout(
        total_out=total_out,
        total_local=total_local,
        total_padded=total_padded,
        pack_single_span=pack_single_span,
        pack_single_dst_base=pack_single_dst_base,
        pack_local_cu=tuple(pack_local_cu),
        pack_dst_base=tuple(pack_dst_base),
        reorg_single_req=reorg_single_req,
        reorg_single_start_mod=reorg_start_mod[0] if reorg_single_req else 0,
        reorg_out_cu=tuple(reorg_out_cu),
        reorg_pad_base=tuple(reorg_pad_base),
        reorg_start_mod=tuple(reorg_start_mod),
    )


class LayerOrder:
    """Execution order of MLA layers learned from one complete forward pass."""

    def __init__(self) -> None:
        self.layer_ids: list[int] = []
        self.learning = self.frozen = self.in_order = False
        self.cursor = 0

    def advance(self, layer_id: int, new_forward: bool) -> Optional[int]:
        if new_forward:
            if self.learning and self.layer_ids:
                self.frozen = True
            elif self.in_order and self.cursor != len(self.layer_ids):
                self.frozen = False
            self.learning = not self.frozen
            if self.learning:
                self.layer_ids = []
            self.in_order = self.frozen
            self.cursor = 0

        if self.learning:
            if layer_id in self.layer_ids:
                self.learning = False
                self.layer_ids = []
            else:
                self.layer_ids.append(layer_id)
            return None

        if (
            not self.in_order
            or self.cursor >= len(self.layer_ids)
            or layer_id != self.layer_ids[self.cursor]
        ):
            self.in_order = self.frozen = False
            return None
        pos = self.cursor
        self.cursor += 1
        return pos


# ---------------------------------------------------------------------------
# Triton kernels: raw wire pack, fused zero-self-copy reorg, plan-hash check.
# ---------------------------------------------------------------------------


@triton.jit(
    do_not_specialize=["n_local", "n_spans", "single_dst_base", "buf_stride"],
    do_not_specialize_on_alignment=[
        "kv_buf_ptr",
        "loc_ptr",
        "own_wire_ptr",
        "local_cu_ptr",
        "dst_base_ptr",
    ],
)
def _pack_dcp_kv_kernel(
    kv_buf_ptr,
    loc_ptr,
    own_wire_ptr,
    local_cu_ptr,
    dst_base_ptr,
    n_local,
    n_spans,
    single_dst_base,
    buf_stride: tl.int64,
    KV_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ROWS: tl.constexpr,
    SINGLE_SPAN: tl.constexpr,
):
    base_row = tl.program_id(0) * ROWS
    for r in tl.static_range(0, ROWS):
        pid = base_row + r
        if pid < n_local:
            if SINGLE_SPAN:
                dst_row = single_dst_base + pid
            else:
                lo, hi = 0, n_spans
                while lo + 1 < hi:
                    mid = (lo + hi) >> 1
                    if tl.load(local_cu_ptr + mid) <= pid:
                        lo = mid
                    else:
                        hi = mid
                dst_row = tl.load(dst_base_ptr + lo) + (
                    pid - tl.load(local_cu_ptr + lo)
                )
            loc = tl.load(loc_ptr + pid).to(tl.int64)
            src_base = kv_buf_ptr + loc * buf_stride
            dst_base = own_wire_ptr + dst_row.to(tl.int64) * KV_DIM
            for d0 in tl.static_range(0, KV_DIM, BLOCK_D):
                offs = d0 + tl.arange(0, BLOCK_D)
                mask = offs < KV_DIM
                tl.store(
                    dst_base + offs, tl.load(src_base + offs, mask=mask), mask=mask
                )


@triton.jit(
    do_not_specialize=["total_out", "n_reqs", "single_start_mod", "wire_rows"],
    do_not_specialize_on_alignment=[
        "own_wire_ptr",
        "peer_stage_ptr",
        "out_nope_ptr",
        "out_rope_ptr",
        "out_cu_ptr",
        "pad_base_ptr",
        "start_mod_ptr",
    ],
)
def _reorg_dcp_kv_kernel(
    own_wire_ptr,
    peer_stage_ptr,
    out_nope_ptr,
    out_rope_ptr,
    out_cu_ptr,
    pad_base_ptr,
    start_mod_ptr,
    total_out,
    n_reqs,
    single_start_mod,
    wire_rows: tl.int64,
    RANK: tl.constexpr,
    WORLD: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    KV_DIM: tl.constexpr,
    NOPE_BLOCK: tl.constexpr,
    ROPE_BLOCK: tl.constexpr,
    ROWS: tl.constexpr,
    SINGLE_REQ: tl.constexpr,
    SPLIT_OUT: tl.constexpr,
):
    base_row = tl.program_id(0) * ROWS
    for r in tl.static_range(0, ROWS):
        pid = base_row + r
        if pid < total_out:
            if SINGLE_REQ:
                local_t, pad_base, start_mod = pid, 0, single_start_mod
            else:
                lo, hi = 0, n_reqs
                while lo + 1 < hi:
                    mid = (lo + hi) >> 1
                    if tl.load(out_cu_ptr + mid) <= pid:
                        lo = mid
                    else:
                        hi = mid
                local_t = pid - tl.load(out_cu_ptr + lo)
                pad_base = tl.load(pad_base_ptr + lo)
                start_mod = tl.load(start_mod_ptr + lo)

            idx = start_mod + local_t
            owner = idx % WORLD
            padded_row = (pad_base + (idx // WORLD)).to(tl.int64)
            is_own = owner == RANK
            peer_slot = tl.maximum(tl.where(owner < RANK, owner, owner - 1), 0).to(
                tl.int64
            )
            own_row = own_wire_ptr + padded_row * KV_DIM
            peer_row = peer_stage_ptr + (peer_slot * wire_rows + padded_row) * KV_DIM
            src_row = tl.where(is_own, own_row, peer_row)
            pid64 = pid.to(tl.int64)

            if SPLIT_OUT:
                nope_dst = out_nope_ptr + pid64 * NOPE_DIM
                for d0 in tl.static_range(0, NOPE_DIM, NOPE_BLOCK):
                    offs = d0 + tl.arange(0, NOPE_BLOCK)
                    mask = offs < NOPE_DIM
                    tl.store(
                        nope_dst + offs,
                        tl.load(src_row + offs, mask=mask),
                        mask=mask,
                    )

                rope_dst = out_rope_ptr + pid64 * ROPE_DIM
                for d0 in tl.static_range(0, ROPE_DIM, ROPE_BLOCK):
                    offs = d0 + tl.arange(0, ROPE_BLOCK)
                    mask = offs < ROPE_DIM
                    tl.store(
                        rope_dst + offs,
                        tl.load(src_row + NOPE_DIM + offs, mask=mask),
                        mask=mask,
                    )
            else:
                out_dst = out_nope_ptr + pid64 * KV_DIM
                for d0 in tl.static_range(0, KV_DIM, NOPE_BLOCK):
                    offs = d0 + tl.arange(0, NOPE_BLOCK)
                    mask = offs < KV_DIM
                    tl.store(
                        out_dst + offs,
                        tl.load(src_row + offs, mask=mask),
                        mask=mask,
                    )


@triton.jit(debug=True, do_not_specialize=["slot"])
def _plan_check_kernel(
    words_ptr,
    deltas_ptr,
    slot,
    WORLD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    ranks = tl.arange(0, BLOCK)
    live = ranks < WORLD
    at = words_ptr + tl.load(deltas_ptr + ranks, mask=live, other=0) + slot * 8
    theirs = tl.load(at.to(tl.pointer_type(tl.int64)), mask=live, other=0)
    mine = tl.load((words_ptr + slot * 8).to(tl.pointer_type(tl.int64)))
    tl.device_assert(
        (theirs == mine) | (ranks >= WORLD),
        "DCP KV prefetch: DCP ranks diverged on step plan",
    )


# ---------------------------------------------------------------------------
# Per-step runtime state and prefetcher.
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _ChunkRuntime:
    layout: DcpChunkLayout
    local_indices: torch.Tensor
    copies_pair: tuple[CopyList, CopyList]
    pack_local_cu: torch.Tensor
    pack_dst_base: torch.Tensor
    reorg_out_cu: torch.Tensor
    reorg_pad_base: torch.Tensor
    reorg_start_mod: torch.Tensor
    # Precomputed output views into single-slot out_pool to avoid per-layer .view() calls.
    reorg_out: tuple[torch.Tensor, torch.Tensor]
    join_view: Any


@dataclass(slots=True)
class _StepState:
    mode: str
    split_out: bool
    dst_dtype: torch.dtype
    same_dtype: bool
    layer_ids: tuple[int, ...]
    kv_buffers_raw: tuple[torch.Tensor, ...]
    kv_buf_strides: tuple[int, ...]
    chunks: tuple[_ChunkRuntime, ...]
    plan_slot: int
    checked: bool = False


@dataclass(slots=True)
class _PendingItem:
    layer_id: int
    layer_pos: int
    chunk_idx: int
    parity: int
    split_out: bool
    join_view: Any


def _raw_wire_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.uint8 if dtype.itemsize == 1 else torch.int16


def _to_int_list(seq: Any) -> list[int]:
    return [int(x) for x in (seq.tolist() if isinstance(seq, torch.Tensor) else seq)]


class DcpKvPrefetcher:
    """Overlapped zero-SM copy-engine DCP KV prefetcher for Dense MLA extend."""

    _PACK_BLOCK_D = 512
    _ROWS_PER_CTA = 4
    _KERNEL_WARPS = 2

    def __init__(
        self,
        dcp_group: Any,
        device: torch.device,
        *,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        kv_cache_dtype: torch.dtype,
        out_dtype: torch.dtype,
        max_chunk_capacity: int,
    ) -> None:
        self.device_group = dcp_group.device_group
        self.cpu_group = dcp_group.cpu_group
        self.world_size: int = dcp_group.world_size
        self.rank: int = dcp_group.rank_in_group
        self.device = torch.device(device)
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.kv_dim = kv_lora_rank + qk_rope_head_dim
        self._reorg_nope_block = (
            min(triton.next_power_of_2(kv_lora_rank), 512) if kv_lora_rank > 0 else 128
        )
        self._reorg_rope_block = (
            min(triton.next_power_of_2(qk_rope_head_dim), 128)
            if qk_rope_head_dim > 0
            else 64
        )
        self.kv_cache_dtype = kv_cache_dtype
        self.out_dtype = out_dtype
        self.wire_dtype = _raw_wire_dtype(kv_cache_dtype)
        self.wire_row_bytes = self.kv_dim * kv_cache_dtype.itemsize
        self.max_out_rows = max_chunk_capacity
        # Cover up to 1024 unaligned requests (+2 padding rows per request).
        self.wire_rows = (
            max_chunk_capacity + self.world_size - 1
        ) // self.world_size + 2048

        self.wire_peers = [p for p in range(self.world_size) if p != self.rank]
        self.pull_order = staggered_pull_order(self.world_size, self.rank)
        self.wire_index = {peer: idx for idx, peer in enumerate(self.wire_peers)}

        self.orders: dict[int, LayerOrder] = {}
        self.order = LayerOrder()
        self.forward_ref: Optional[weakref.ref[Any]] = None
        self.position: Optional[int] = None
        self.step: Optional[_StepState] = None
        self.pending: Optional[_PendingItem] = None
        self.last_parity = 1
        self.plan_count = 0

        self.ready = self._allocate()

    def _agree(self, value: int) -> bool:
        local = torch.tensor([value], dtype=torch.int64)
        gathered = [torch.empty_like(local) for _ in range(self.world_size)]
        dist.all_gather(gathered, local, group=self.cpu_group)
        return all(int(t.item()) == 1 for t in gathered)

    def _clear_buffers(self) -> None:
        self.own_wire = self.plan_words = self.peer_stage = self.out_pool = None  # type: ignore[assignment]

    def _allocate(self) -> bool:
        if (
            self.world_size <= 1
            or self.kv_lora_rank <= 0
            or self.qk_rope_head_dim <= 0
            or self.max_out_rows <= 0
            or self.kv_cache_dtype not in _DTYPE_CODE
            or self.out_dtype not in (torch.bfloat16, torch.float16)
            or symm_mem is None
        ):
            return False

        allocated = 0
        try:
            self.driver = _memcpy_driver()
            with torch.inference_mode(False), torch.no_grad():
                # Double-buffered own_wire[2] allows alternating single-barrier
                # synchronization on channel=parity (no separate HEAD_DONE).
                self.own_wire = symm_mem.empty(
                    (2, self.wire_rows, self.kv_dim),
                    dtype=self.wire_dtype,
                    device=self.device,
                )
                self.plan_words = symm_mem.empty(
                    (PLAN_SLOTS,), dtype=torch.int64, device=self.device
                )
            self.peer_stage = torch.empty(
                (len(self.wire_peers), self.wire_rows, self.kv_dim),
                dtype=self.wire_dtype,
                device=self.device,
            )
            # Single-slot output pool: every caller consumes prefetched KV
            # (kv_b_proj + pack_fn / torch.cat / copy_) before calling
            # start_next(), so strict fence_ev guarantees safe reuse while
            # halving output VRAM (-144 MiB) and keeping <=64K hot in L2.
            self.out_pool = torch.empty(
                (self.max_out_rows * self.kv_dim,),
                dtype=self.out_dtype,
                device=self.device,
            )
            self.dummy_i32 = torch.zeros(1, dtype=torch.int32, device=self.device)
            self.side = torch.cuda.Stream(device=self.device)
            self.copy_stream = torch.cuda.Stream(device=self.device)
            self._side_handle = self.side.cuda_stream
            self._copy_handle = self.copy_stream.cuda_stream
            self._own_wire_slots = (self.own_wire[0], self.own_wire[1])
            self._own_wire_typed = (
                self.own_wire[0].view(self.kv_cache_dtype),
                self.own_wire[1].view(self.kv_cache_dtype),
            )
            self._peer_stage_typed = self.peer_stage.view(self.kv_cache_dtype)
            self._out_pool_raw = self.out_pool.view(torch.uint8)
            self._plan_words_u8 = self.plan_words.view(torch.uint8)
            self._pack_fast: dict[bool, Any] = {}
            self._reorg_fast: dict[tuple[torch.dtype, bool, bool], Any] = {}
            self.fence_ev = torch.cuda.Event()
            self.copy_ready_ev = torch.cuda.Event()
            self.copy_done_ev = torch.cuda.Event()
            self.side_done_ev = torch.cuda.Event()
            allocated = 1
        except (RuntimeError, ValueError, OSError, AttributeError) as err:
            logger.warning("DCP copy-engine KV prefetch unavailable: %s", err)

        if not self._agree(allocated):
            self._clear_buffers()
            return False

        self.own_wire.zero_()
        self.plan_words.zero_()
        torch.cuda.synchronize(self.device)
        connected = 0
        try:
            self.handle = symm_mem.rendezvous(self.own_wire, group=self.device_group)
            self.handle.barrier()
            self.remote = [
                self.handle.get_buffer(
                    p, list(self.own_wire.shape), self.own_wire.dtype, 0
                )
                for p in range(self.world_size)
            ]
            self.remote_ptrs = (
                [buf[0].data_ptr() for buf in self.remote],
                [buf[1].data_ptr() for buf in self.remote],
            )

            self.plan_handle = symm_mem.rendezvous(
                self.plan_words, group=self.device_group
            )
            self.plan_handle.barrier()
            self.plan_remote = [
                self.plan_handle.get_buffer(p, [PLAN_SLOTS], torch.int64, 0)
                for p in range(self.world_size)
            ]
            self.plan_deltas = torch.tensor(
                [b.data_ptr() - self.plan_words.data_ptr() for b in self.plan_remote],
                dtype=torch.int64,
                device=self.device,
            )
            self.stage_ptrs = [
                self.peer_stage[idx].data_ptr() for idx in range(len(self.wire_peers))
            ]
            connected = 1
        except (RuntimeError, ValueError, OSError, AttributeError) as err:
            logger.warning("DCP copy-engine KV prefetch rendezvous failed: %s", err)

        if not self._agree(connected):
            self._clear_buffers()
            return False

        self._warmup_kernels()
        logger.info(
            "Enabled Zero-SM copy-engine DCP KV prefetch "
            "(dcp_size=%d, kv_dtype=%s, out_dtype=%s, max_chunk_tokens=%d, wire_rows=%d)",
            self.world_size,
            self.kv_cache_dtype,
            self.out_dtype,
            self.max_out_rows,
            self.wire_rows,
        )
        return True

    @staticmethod
    def _make_fast_launcher(ck: Any) -> Optional[Any]:
        """Build a zero-overhead C launcher closure for a compiled Triton kernel."""
        if not (
            hasattr(ck, "run")
            and hasattr(ck, "function")
            and hasattr(ck, "packed_metadata")
        ):
            return None
        run, fn, meta = ck.run, ck.function, ck.packed_metadata
        # Direct C-extension fast path when no Triton scratch/gsan is needed.
        if (
            hasattr(run, "launch")
            and getattr(run, "global_scratch_size", 1) == 0
            and getattr(run, "profile_scratch_size", 1) == 0
            and not getattr(run, "gsan_enabled", True)
            and hasattr(run, "arg_annotations")
            and hasattr(run, "kernel_signature")
        ):
            launch = run.launch
            coop = run.launch_cooperative_grid
            pdl = run.launch_pdl
            ann = run.arg_annotations
            ksig = run.kernel_signature
            return lambda gx, stream_h, args: launch(
                gx,
                1,
                1,
                stream_h,
                fn,
                coop,
                pdl,
                meta,
                None,
                None,
                None,
                None,
                None,
                ann,
                ksig,
                args,
            )
        return lambda gx, stream_h, args: run(
            gx, 1, 1, stream_h, fn, meta, None, None, None, *args
        )

    def _launch_pack(
        self,
        parity: int,
        kv_raw: torch.Tensor,
        buf_stride: int,
        local_indices: torch.Tensor,
        local_cu: torch.Tensor,
        dst_base: torch.Tensor,
        n_local: int,
        n_spans: int,
        single_dst_base: int,
        *,
        single_span: bool,
    ) -> None:
        own_slot = self._own_wire_slots[parity]
        grid_x = max(triton.cdiv(n_local, self._ROWS_PER_CTA), 1)
        args = (
            kv_raw,
            local_indices,
            own_slot,
            local_cu,
            dst_base,
            n_local,
            n_spans,
            single_dst_base,
            buf_stride,
        )
        fast = self._pack_fast.get(single_span)
        if fast is not None:
            fast(grid_x, self._side_handle, args)
            return
        ck = _pack_dcp_kv_kernel[(grid_x,)](
            *args,
            KV_DIM=self.kv_dim,
            BLOCK_D=self._PACK_BLOCK_D,
            ROWS=self._ROWS_PER_CTA,
            SINGLE_SPAN=single_span,
            num_warps=self._KERNEL_WARPS,
        )
        cached = self._make_fast_launcher(ck)
        if cached is not None:
            self._pack_fast[single_span] = cached

    def _launch_reorg(
        self,
        parity: int,
        out_nope: torch.Tensor,
        out_rope: torch.Tensor,
        total_out: int,
        dst_dtype: torch.dtype,
        same_dtype: bool,
        out_cu: torch.Tensor,
        pad_base: torch.Tensor,
        start_mod: torch.Tensor,
        n_reqs: int,
        single_start_mod: int,
        *,
        single_req: bool,
        split_out: bool,
    ) -> None:
        in_own = (
            self._own_wire_slots[parity] if same_dtype else self._own_wire_typed[parity]
        )
        in_peer = self.peer_stage if same_dtype else self._peer_stage_typed
        grid_x = max(triton.cdiv(total_out, self._ROWS_PER_CTA), 1)
        args = (
            in_own,
            in_peer,
            out_nope,
            out_rope,
            out_cu,
            pad_base,
            start_mod,
            total_out,
            n_reqs,
            single_start_mod,
            self.wire_rows,
        )
        key = (dst_dtype, single_req, split_out)
        fast = self._reorg_fast.get(key)
        if fast is not None:
            fast(grid_x, self._side_handle, args)
            return
        ck = _reorg_dcp_kv_kernel[(grid_x,)](
            *args,
            RANK=self.rank,
            WORLD=self.world_size,
            NOPE_DIM=self.kv_lora_rank,
            ROPE_DIM=self.qk_rope_head_dim,
            KV_DIM=self.kv_dim,
            NOPE_BLOCK=self._reorg_nope_block,
            ROPE_BLOCK=self._reorg_rope_block,
            ROWS=self._ROWS_PER_CTA,
            SINGLE_REQ=single_req,
            SPLIT_OUT=split_out,
            num_warps=self._KERNEL_WARPS,
        )
        cached = self._make_fast_launcher(ck)
        if cached is not None:
            self._reorg_fast[key] = cached

    def _warmup_kernels(self) -> None:
        """Pre-compile Triton kernels before CUDA graph capture / serving."""
        with torch.cuda.stream(self.side):
            _plan_check_kernel[(1,)](
                self._plan_words_u8,
                self.plan_deltas,
                0,
                WORLD=self.world_size,
                BLOCK=triton.next_power_of_2(self.world_size),
            )
            dummy_kv = torch.zeros(
                (1, 1, self.kv_dim), dtype=self.wire_dtype, device=self.device
            )
            same = self.out_dtype == self.kv_cache_dtype
            reorg_dtype = self.wire_dtype if same else self.out_dtype
            out_a, out_pe = self._split_out_views(1, reorg_dtype)
            out_comb = self._combined_out_view(1, self.wire_dtype)
            for single in (True, False):
                self._launch_pack(
                    0,
                    dummy_kv,
                    self.kv_dim,
                    self.dummy_i32,
                    self.dummy_i32,
                    self.dummy_i32,
                    0,
                    1,
                    0,
                    single_span=single,
                )
                self._launch_reorg(
                    0,
                    out_a,
                    out_pe,
                    0,
                    self.out_dtype,
                    same,
                    self.dummy_i32,
                    self.dummy_i32,
                    self.dummy_i32,
                    1,
                    0,
                    single_req=single,
                    split_out=True,
                )
                self._launch_reorg(
                    0,
                    out_comb,
                    out_comb,
                    0,
                    self.kv_cache_dtype,
                    True,
                    self.dummy_i32,
                    self.dummy_i32,
                    self.dummy_i32,
                    1,
                    0,
                    single_req=single,
                    split_out=False,
                )
        torch.cuda.synchronize(self.device)

    def _buffer_supported(self, kv_buf: torch.Tensor) -> bool:
        return (
            isinstance(kv_buf, torch.Tensor)
            and kv_buf.is_cuda
            and kv_buf.dim() == 3
            and kv_buf.shape[1] == 1
            and kv_buf.shape[2] == self.kv_dim
            and kv_buf.stride(2) == 1
            and kv_buf.stride(0) >= self.kv_dim
            and kv_buf.dtype == self.kv_cache_dtype
        )

    def _split_out_views(
        self, total_out: int, dst_dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self._out_pool_raw
        elem = dst_dtype.itemsize
        nope_bytes = total_out * self.kv_lora_rank * elem
        total_bytes = total_out * self.kv_dim * elem
        return (
            raw[:nope_bytes].view(dst_dtype).view(total_out, self.kv_lora_rank),
            raw[nope_bytes:total_bytes]
            .view(dst_dtype)
            .view(total_out, 1, self.qk_rope_head_dim),
        )

    def _combined_out_view(
        self, total_out: int, dst_dtype: torch.dtype
    ) -> torch.Tensor:
        raw = self._out_pool_raw
        total_bytes = total_out * self.kv_dim * dst_dtype.itemsize
        return raw[:total_bytes].view(dst_dtype).view(total_out, 1, self.kv_dim)

    def _abandon(self) -> None:
        if self.pending is not None:
            self.side_done_ev.record(self.side)
            torch.cuda.current_stream(self.device).wait_event(self.side_done_ev)
            self.pending = None
        self.step = None

    def _advance_order(self, layer_id: int, forward_batch: object) -> Optional[int]:
        seen = self.forward_ref() if self.forward_ref is not None else None
        new_forward = seen is not forward_batch
        if new_forward:
            self._abandon()
            self.forward_ref = weakref.ref(forward_batch)
            order = self.orders.get(layer_id)
            if order is None:
                if len(self.orders) >= 8:
                    self.position = None
                    return None
                order = self.orders[layer_id] = LayerOrder()
            self.order = order
        pos = self.order.advance(layer_id, new_forward)
        self.position = pos
        if pos is None:
            self._abandon()
        return pos

    def observe_layer(self, layer_id: int, forward_batch: object) -> None:
        """Record layer execution order on prefix-free extend steps (chunk 0)."""
        if not self.ready or torch.cuda.is_current_stream_capturing():
            return
        self._advance_order(layer_id, forward_batch)

    def begin_layer(
        self,
        layer_id: int,
        forward_batch: object,
        token_to_kv_pool: Any,
        *,
        mode: str,
        dst_dtype: torch.dtype,
        kv_indices_list: Sequence[torch.Tensor],
        seq_lens_list: Sequence[Any],
        starts_list: Optional[Sequence[Any]] = None,
    ) -> None:
        """Advance layer order and build the batch prefetch plan at layer 0."""
        if not self.ready or torch.cuda.is_current_stream_capturing():
            return
        if self._advance_order(layer_id, forward_batch) == 0:
            self._derive_step_plan(
                token_to_kv_pool=token_to_kv_pool,
                mode=mode,
                dst_dtype=dst_dtype,
                kv_indices_list=kv_indices_list,
                seq_lens_list=seq_lens_list,
                starts_list=starts_list,
            )

    def _i32_for_side(self, values: Sequence[int]) -> torch.Tensor:
        tensor = torch.tensor(values, dtype=torch.int32, device=self.device)
        tensor.record_stream(self.side)
        return tensor

    def _derive_step_plan(
        self,
        *,
        token_to_kv_pool: Any,
        mode: str,
        dst_dtype: torch.dtype,
        kv_indices_list: Sequence[torch.Tensor],
        seq_lens_list: Sequence[Any],
        starts_list: Optional[Sequence[Any]],
    ) -> None:
        if (
            mode not in _MODE_CODE
            or dst_dtype not in _DTYPE_CODE
            or dst_dtype.itemsize > self.out_dtype.itemsize
            or not kv_indices_list
            or len(kv_indices_list) != len(seq_lens_list)
            or getattr(token_to_kv_pool, "layer_transfer_counter", None) is not None
        ):
            return

        router = getattr(token_to_kv_pool, "layer_router", None)
        layer_ids = tuple(self.order.layer_ids)
        if router is None or not layer_ids:
            return

        kv_buffers: list[torch.Tensor] = []
        for lid in layer_ids:
            try:
                kv_layer = router.layer(lid)
            except KeyError:
                return
            view = kv_layer.view
            if (
                getattr(view, "layer_transfer_counter", None) is not None
                or getattr(view, "kv_scale_buffer", None) is not None
                or getattr(view, "use_dsa", False)
            ):
                return
            kv_buf = kv_layer.key()
            if not self._buffer_supported(kv_buf):
                return
            kv_buffers.append(kv_buf)

        chunks: list[_ChunkRuntime] = []
        digest: list[int] = [
            _PLAN_DOMAIN_TAG,
            _MODE_CODE[mode],
            _DTYPE_CODE[self.kv_cache_dtype],
            _DTYPE_CODE[dst_dtype],
            self.world_size,
            len(layer_ids),
            *layer_ids,
            len(kv_indices_list),
        ]

        split_out = mode != "mla_absorb"
        same_dtype = dst_dtype == self.kv_cache_dtype
        reorg_dtype = self.wire_dtype if same_dtype else dst_dtype

        for idx, (local_indices, seq_lens_raw) in enumerate(
            zip(kv_indices_list, seq_lens_list)
        ):
            seq_lens = _to_int_list(seq_lens_raw)
            starts = _to_int_list(starts_list[idx]) if starts_list is not None else None
            layout = plan_dcp_chunk_layout(seq_lens, starts, self.world_size, self.rank)
            if (
                layout.total_out <= 0
                or layout.total_out > self.max_out_rows
                or layout.total_padded <= 0
                or layout.total_padded > self.wire_rows
                or local_indices.numel() != layout.total_local
                or not local_indices.is_cuda
                or not local_indices.is_contiguous()
            ):
                return

            nbytes = layout.total_padded * self.wire_row_bytes
            copies_pair = (
                CopyList(
                    [
                        (
                            self.stage_ptrs[self.wire_index[p]],
                            self.remote_ptrs[0][p],
                            nbytes,
                        )
                        for p in self.pull_order
                    ]
                ),
                CopyList(
                    [
                        (
                            self.stage_ptrs[self.wire_index[p]],
                            self.remote_ptrs[1][p],
                            nbytes,
                        )
                        for p in self.pull_order
                    ]
                ),
            )
            local_indices.record_stream(self.side)
            if split_out:
                reorg_out = self._split_out_views(layout.total_out, reorg_dtype)
                join_view: Any = self._split_out_views(layout.total_out, dst_dtype)
            else:
                r0 = self._combined_out_view(layout.total_out, reorg_dtype)
                reorg_out = (r0, r0)
                join_view = self._combined_out_view(layout.total_out, dst_dtype)
            chunks.append(
                _ChunkRuntime(
                    layout=layout,
                    local_indices=local_indices,
                    copies_pair=copies_pair,
                    pack_local_cu=(
                        self.dummy_i32
                        if layout.pack_single_span
                        else self._i32_for_side(layout.pack_local_cu)
                    ),
                    pack_dst_base=(
                        self.dummy_i32
                        if layout.pack_single_span
                        else self._i32_for_side(layout.pack_dst_base)
                    ),
                    reorg_out_cu=(
                        self.dummy_i32
                        if layout.reorg_single_req
                        else self._i32_for_side(layout.reorg_out_cu)
                    ),
                    reorg_pad_base=(
                        self.dummy_i32
                        if layout.reorg_single_req
                        else self._i32_for_side(layout.reorg_pad_base)
                    ),
                    reorg_start_mod=(
                        self.dummy_i32
                        if layout.reorg_single_req
                        else self._i32_for_side(layout.reorg_start_mod)
                    ),
                    reorg_out=reorg_out,
                    join_view=join_view,
                )
            )
            digest.extend((layout.total_out, layout.total_padded))

        self.plan_count += 1
        plan_slot = self.plan_count % PLAN_SLOTS
        self.plan_words[plan_slot].fill_(plan_hash(digest))
        kv_buffers_raw = tuple(buf.view(self.wire_dtype) for buf in kv_buffers)
        self.step = _StepState(
            mode=mode,
            split_out=split_out,
            dst_dtype=dst_dtype,
            same_dtype=same_dtype,
            layer_ids=layer_ids,
            kv_buffers_raw=kv_buffers_raw,
            kv_buf_strides=tuple(buf.stride(0) for buf in kv_buffers_raw),
            chunks=tuple(chunks),
            plan_slot=plan_slot,
        )

    def _launch_item(self, layer_pos: int, chunk_idx: int) -> None:
        step = self.step
        assert step is not None
        layer_id = step.layer_ids[layer_pos]
        chunk = step.chunks[chunk_idx]
        layout = chunk.layout
        parity = 1 - self.last_parity
        self.last_parity = parity

        # Strict post-consumption fence before pack: makes local_indices,
        # plan_words, and prior KV writes visible on (0, 0), guarantees main
        # finished reading single-slot out_pool before reuse, and keeps
        # pack/barrier/DMA/reorg completely out of persistent attention on
        # inter-layer handoffs (0 us AuxSM-Attn overlap).
        main = torch.cuda.current_stream(self.device)
        self.fence_ev.record(main)
        self.side.wait_event(self.fence_ev)

        torch.cuda.set_stream(self.side)
        try:
            if layout.total_local > 0:
                self._launch_pack(
                    parity,
                    step.kv_buffers_raw[layer_pos],
                    step.kv_buf_strides[layer_pos],
                    chunk.local_indices,
                    chunk.pack_local_cu,
                    chunk.pack_dst_base,
                    layout.total_local,
                    len(layout.pack_dst_base),
                    layout.pack_single_dst_base,
                    single_span=layout.pack_single_span,
                )

            # Single alternating barrier on channel=parity proves both that all
            # peers packed own_wire[parity] AND finished reading own_wire[1-parity]
            # from the previous launch (double-buffered, zero HEAD_DONE needed).
            self.handle.barrier(channel=parity, timeout_ms=BARRIER_TIMEOUT_MS)
            if not step.checked:
                step.checked = True
                _plan_check_kernel[(1,)](
                    self._plan_words_u8,
                    self.plan_deltas,
                    step.plan_slot,
                    WORLD=self.world_size,
                    BLOCK=triton.next_power_of_2(self.world_size),
                )

            # Zero-SM copy-engine remote peer pulls in staggered ring order.
            self.copy_ready_ev.record(self.side)
            self.copy_stream.wait_event(self.copy_ready_ev)
            self.driver.memcpy_list(chunk.copies_pair[parity], self._copy_handle)
            self.copy_done_ev.record(self.copy_stream)
            self.side.wait_event(self.copy_done_ev)

            out_nope, out_rope = chunk.reorg_out
            self._launch_reorg(
                parity,
                out_nope,
                out_rope,
                layout.total_out,
                step.dst_dtype,
                step.same_dtype,
                chunk.reorg_out_cu,
                chunk.reorg_pad_base,
                chunk.reorg_start_mod,
                len(layout.reorg_pad_base),
                layout.reorg_single_start_mod,
                single_req=layout.reorg_single_req,
                split_out=step.split_out,
            )
        finally:
            torch.cuda.set_stream(main)

        self.pending = _PendingItem(
            layer_id=layer_id,
            layer_pos=layer_pos,
            chunk_idx=chunk_idx,
            parity=parity,
            split_out=step.split_out,
            join_view=chunk.join_view,
        )

    def start_first(self) -> None:
        if self.step is not None and self.position == 0 and self.pending is None:
            self._launch_item(0, 0)

    def _join_pending(
        self, layer_id: int, chunk_idx: int, *, split_out: bool
    ) -> Optional[_PendingItem]:
        pending = self.pending
        if (
            pending is None
            or pending.layer_id != layer_id
            or pending.chunk_idx != chunk_idx
            or pending.split_out != split_out
            or torch.cuda.is_current_stream_capturing()
        ):
            return None
        self.pending = None
        self.side_done_ev.record(self.side)
        torch.cuda.current_stream(self.device).wait_event(self.side_done_ev)
        return pending

    def join_split(
        self, layer_id: int, chunk_idx: int
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        pending = self._join_pending(layer_id, chunk_idx, split_out=True)
        return pending.join_view if pending is not None else None

    def join_combined(self, layer_id: int, chunk_idx: int) -> Optional[torch.Tensor]:
        pending = self._join_pending(layer_id, chunk_idx, split_out=False)
        return pending.join_view if pending is not None else None

    def start_next(self, layer_id: int, chunk_idx: int) -> None:
        step, pos = self.step, self.position
        if (
            step is None
            or pos is None
            or self.pending is not None
            or pos >= len(step.layer_ids)
            or step.layer_ids[pos] != layer_id
        ):
            return
        if chunk_idx + 1 < len(step.chunks):
            self._launch_item(pos, chunk_idx + 1)
        elif pos + 1 < len(step.layer_ids):
            self._launch_item(pos + 1, 0)
        else:
            self.step = None


# ---------------------------------------------------------------------------
# Process-wide singleton and thin call-site helpers for runners / MLA mixins.
# ---------------------------------------------------------------------------

_DCP_KV_PREFETCHER: Optional[DcpKvPrefetcher] = None


def init_dcp_kv_prefetch(
    dcp_group: Any,
    device: torch.device,
    *,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    kv_cache_dtype: torch.dtype,
    out_dtype: torch.dtype,
    max_chunk_capacity: int,
) -> Optional[DcpKvPrefetcher]:
    global _DCP_KV_PREFETCHER
    if _DCP_KV_PREFETCHER is not None or dcp_group.world_size <= 1:
        return _DCP_KV_PREFETCHER
    prefetcher = DcpKvPrefetcher(
        dcp_group,
        device,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        kv_cache_dtype=kv_cache_dtype,
        out_dtype=out_dtype,
        max_chunk_capacity=max_chunk_capacity,
    )
    if prefetcher.ready:
        _DCP_KV_PREFETCHER = prefetcher
    return _DCP_KV_PREFETCHER


def init_dcp_kv_prefetch_for_runner(model_runner: Any) -> Optional[DcpKvPrefetcher]:
    """Initialize DCP KV prefetch from ``BaseRunner._warmup()`` if enabled."""
    from sglang.srt.environ import envs
    from sglang.srt.runtime_context import get_parallel

    parallel = get_parallel()
    if (
        not envs.SGLANG_ENABLE_DCP_KV_PREFETCH.get()
        or not parallel.dcp_enabled
        or model_runner.is_draft_worker
        or not model_runner.use_mla_backend
        or getattr(model_runner.token_to_kv_pool, "use_dsa", False)
    ):
        return None
    kv_lora_rank = getattr(model_runner.model_config, "kv_lora_rank", None)
    qk_rope_head_dim = getattr(model_runner.model_config, "qk_rope_head_dim", None)
    if not kv_lora_rank or not qk_rope_head_dim:
        return None
    return init_dcp_kv_prefetch(
        parallel.dcp_group,
        torch.device("cuda", torch.cuda.current_device()),
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        kv_cache_dtype=model_runner.kv_cache_dtype,
        out_dtype=model_runner.dtype,
        max_chunk_capacity=envs.SGLANG_MAX_KV_CHUNK_CAPACITY.get(),
    )


def get_dcp_kv_prefetch() -> Optional[DcpKvPrefetcher]:
    return _DCP_KV_PREFETCHER


def reset_dcp_kv_prefetch() -> None:
    global _DCP_KV_PREFETCHER
    _DCP_KV_PREFETCHER = None


def _reads_are_translated() -> bool:
    from sglang.srt.model_executor.forward_context import get_attn_backend

    runner_ids = getattr(get_attn_backend(), "runner_ids", None)
    return bool(
        runner_ids is not None and getattr(runner_ids, "reads_are_translated", False)
    )


def maybe_start_dcp_mha_prefetch(
    layer_id: int,
    forward_batch: Any,
    token_to_kv_pool: Any,
    *,
    mode: str,
    dst_dtype: torch.dtype,
    has_prefix: bool = True,
) -> Optional[DcpKvPrefetcher]:
    """Start (or observe) DCP KV prefetch for MHA chunked/one-shot extend."""
    prefetcher = _DCP_KV_PREFETCHER
    if prefetcher is None:
        return None
    if mode == "mha_chunked":
        kv_indices_list = forward_batch.prefix_chunk_kv_indices
        seq_lens_list = forward_batch.prefix_chunk_seq_lens_cpu
        starts_list = forward_batch.prefix_chunk_starts_cpu
    else:
        dcp_meta = forward_batch.attn_dcp_metadata
        local_ids = (
            dcp_meta.dcp_local_prefix_kv_indices if dcp_meta is not None else None
        )
        prefix_lens = forward_batch.extend_prefix_lens_cpu
        kv_indices_list = [local_ids] if local_ids is not None else None
        seq_lens_list = [prefix_lens] if prefix_lens is not None else None
        starts_list = None
        has_prefix = has_prefix and prefix_lens is not None and any(prefix_lens)
    if (
        has_prefix
        and not _reads_are_translated()
        and kv_indices_list is not None
        and len(kv_indices_list) > 0
        and seq_lens_list is not None
        and len(seq_lens_list) > 0
    ):
        prefetcher.begin_layer(
            layer_id,
            forward_batch,
            token_to_kv_pool,
            mode=mode,
            dst_dtype=dst_dtype,
            kv_indices_list=kv_indices_list,
            seq_lens_list=seq_lens_list,
            starts_list=starts_list,
        )
        prefetcher.start_first()
        return prefetcher
    prefetcher.observe_layer(layer_id, forward_batch)
    return None


def maybe_consume_dcp_kv_split(
    layer_id: int, chunk_idx: int, *, start_next: bool = True
) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    """Join prefetched ``(kv_a, k_pe)`` and optionally launch the next item."""
    prefetcher = _DCP_KV_PREFETCHER
    if prefetcher is None:
        return None
    res = prefetcher.join_split(layer_id, chunk_idx)
    if res is not None and start_next:
        prefetcher.start_next(layer_id, chunk_idx)
    return res


def maybe_fill_dcp_mla_extend_buffer(
    token_to_kv_pool: Any,
    layer_id: int,
    forward_batch: Any,
    kv_lora_rank: int,
    k_nope: torch.Tensor,
    k_pe: torch.Tensor,
) -> bool:
    """Fill ``attn_dcp_metadata.dcp_kv_buffer`` via prefetch; True if handled."""
    prefetcher = _DCP_KV_PREFETCHER
    if prefetcher is None:
        return False
    metadata = forward_batch.attn_dcp_metadata
    prefix_sum = metadata.dcp_extend_prefix_lens_sum
    if prefix_sum == 0 or _reads_are_translated():
        prefetcher.observe_layer(layer_id, forward_batch)
        return False
    prefetcher.begin_layer(
        layer_id,
        forward_batch,
        token_to_kv_pool,
        mode="mla_absorb",
        dst_dtype=metadata.dcp_kv_buffer.dtype,
        kv_indices_list=[metadata.dcp_local_prefix_kv_indices],
        seq_lens_list=[forward_batch.extend_prefix_lens_cpu],
    )
    prefetcher.start_first()
    prefetched = prefetcher.join_combined(layer_id, 0)
    if prefetched is None:
        return False
    metadata.dcp_kv_buffer[:prefix_sum].copy_(prefetched)
    extend_num_tokens = metadata.dcp_kv_buffer.shape[0] - prefix_sum
    metadata.dcp_kv_buffer[prefix_sum:, ..., :kv_lora_rank] = k_nope[:extend_num_tokens]
    metadata.dcp_kv_buffer[prefix_sum:, ..., kv_lora_rank:] = k_pe[:extend_num_tokens]
    return True
