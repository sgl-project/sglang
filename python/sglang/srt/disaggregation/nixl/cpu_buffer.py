# SPDX-License-Identifier: Apache-2.0
"""Bounded pageable staging for NIXL KV transfers.

Pools belong to a manager, not a process singleton.
Only host addresses are registered with NIXL; CUDA copies stay in PyTorch.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np
import torch

from sglang.srt.disaggregation.base.conn import StateType

CPU_BUFFER_VERSION = b"cpu-buffer-v3"


class CpuBufferPool:
    """Nonblocking first-fit allocator; unsettled writes retain their regions."""

    def __init__(self, size_bytes: int):
        if size_bytes < 256:
            raise ValueError("NIXL CPU buffer pool must be at least 256 bytes")
        self.buffer = torch.empty(
            size_bytes, dtype=torch.uint8, device="cpu", pin_memory=False
        )
        self._ranges = {}
        self._pending = []
        self._lock = threading.Lock()

    def allocate(self, size: int):
        if size <= 0:
            raise ValueError("CPU buffer allocation must be positive")
        aligned = (size + 255) // 256 * 256
        with self._lock:
            start = 0
            for offset, length in sorted(self._ranges.items()):
                if offset - start >= aligned:
                    break
                start = offset + length
            if start + aligned > self.buffer.numel():
                raise BufferError(
                    "NIXL CPU buffer pool exhausted; increase --nixl-cpu-buffer-size-gb"
                )
            self._ranges[start] = aligned
        return start, self.buffer[start : start + size]

    def release(self, offset: int):
        with self._lock:
            del self._ranges[offset]

    def retain(self, handle, offset: int):
        # Called before transfer(): even an exception can follow a partial post.
        with self._lock:
            self._pending.append((handle, offset))

    def reap(self, agent):
        with self._lock:
            pending, self._pending = self._pending, []
        held = []
        for handle, offset in pending:
            try:
                settled = agent.check_xfer_state(handle) in ("DONE", "ERR")
            except Exception:
                settled = False
            if settled:
                self.release(offset)
            else:
                held.append((handle, offset))
        with self._lock:
            self._pending.extend(held)

    def register(self, agent):
        descs = agent.register_memory(
            [(self.buffer.data_ptr(), self.buffer.numel(), 0, "")], "DRAM"
        )
        if not descs:
            raise RuntimeError("NIXL CPU buffer registration failed")
        return descs


@dataclass
class CpuReceiveRegion:
    offset: int | None
    buffer: torch.Tensor
    indices: np.ndarray
    required_acks: int
    state_indices: list[np.ndarray] | None = None
    published: bool = False
    release_requested: bool = False


def page_views(buffers, pointers, item_lens):
    """Match physical KV page views to the existing wire descriptor order."""
    by_ptr = {t.data_ptr(): t for t in buffers}
    views = []
    for ptr, width in zip(pointers, item_lens, strict=True):
        tensor = by_ptr.get(ptr)
        if tensor is None or not tensor.is_contiguous() or width <= 0:
            raise ValueError("CPU-buffer PD requires contiguous MHA/MLA KV buffers")
        raw = tensor.view(torch.uint8).reshape(-1)
        if raw.numel() % width:
            raise ValueError(
                "KV buffer size is not a multiple of its physical page size"
            )
        views.append(raw.view(-1, width))
    return views


def pack_pages(views, indices, output):
    """Gather one layer at a time; a pageable D2H copy completes before return."""
    offset = 0
    min_index = int(np.min(indices)) if len(indices) else 0
    max_index = int(np.max(indices)) if len(indices) else -1
    for view in views:
        if min_index < 0 or max_index >= view.shape[0]:
            raise ValueError("CPU-buffer source page index out of bounds")
        ids = torch.as_tensor(indices, dtype=torch.int64, device=view.device)
        size = len(indices) * view.shape[1]
        output[offset : offset + size].copy_(view.index_select(0, ids).reshape(-1))
        offset += size
    if offset != output.numel():
        raise ValueError("CPU-buffer packed size mismatch")


def scatter_pages(views, indices, packed):
    """Restore complete request pages before decode can observe Success."""
    offset = 0
    min_index = int(np.min(indices)) if len(indices) else 0
    max_index = int(np.max(indices)) if len(indices) else -1
    for view in views:
        if min_index < 0 or max_index >= view.shape[0]:
            raise ValueError("CPU-buffer destination page index out of bounds")
        ids = torch.as_tensor(indices, dtype=torch.int64, device=view.device)
        size = len(indices) * view.shape[1]
        values = packed[offset : offset + size].view(len(indices), view.shape[1])
        view.index_copy_(0, ids, values.to(view.device))
        offset += size
    if offset != packed.numel():
        raise ValueError("CPU-buffer scatter size mismatch")
    # index_copy_ is asynchronous on CUDA, including when values.to() blocked.
    if views and views[0].is_cuda:
        torch.cuda.current_stream(views[0].device).synchronize()


def chunk_descriptors(src_ptr, dst_ptr, widths, num_pages, total_pages, start):
    """Layer-major request storage, filled by independent prefill chunks."""
    if start < 0 or num_pages < 0 or start + num_pages > total_pages:
        raise ValueError("CPU-buffer chunk exceeds advertised receive region")
    src, dst = [], []
    src_offset = dst_offset = 0
    for width in widths:
        size = num_pages * width
        src.append((src_ptr + src_offset, size, 0))
        dst.append((dst_ptr + dst_offset + start * width, size, 0))
        src_offset += size
        dst_offset += total_pages * width
    return src, dst


class CpuBufferMixin:
    """NIXL integration; normal VRAM transfers do not enter these methods."""

    @property
    def cpu_buffer_enabled(self):
        return getattr(self, "_cpu_pool", None) is not None

    def set_cpu_kv_buffers(self, pool, *, restore_stream=None):
        self._cpu_restore_stream = restore_stream
        self._cpu_state_views = []
        for state_type, pointers, widths in zip(
            self.kv_args.state_types or [],
            self.kv_args.state_data_ptrs or [],
            self.kv_args.state_item_lens or [],
            strict=True,
        ):
            if state_type == StateType.MAMBA:
                # Includes PLE short-conv and n-gram slot siblings.
                tensors = [
                    tensor
                    for _, tensor, _, _ in pool.mamba_pool._iter_transfer_state_entries()
                ]
            elif state_type == StateType.QSA_PENDING:
                tensors = [
                    *pool.qsa_key_state_buffer_pool,
                    pool.qsa_rope_position_buffer,
                ]
            elif state_type == StateType.QSA_COMPRESSED:
                tensors = pool.qsa_compressed_k_buffer_pool
            else:
                raise ValueError(f"Unsupported CPU-buffer state type: {state_type}")
            self._cpu_state_views.append(page_views(tensors, pointers, widths))
        pool = getattr(pool, "full_kv_pool", pool)
        buffers = getattr(pool, "kv_buffer", None)
        if buffers is None:
            buffers = list(pool.k_buffer) + list(pool.v_buffer)
        self._cpu_views = page_views(
            buffers, self.kv_args.kv_data_ptrs, self.kv_args.kv_item_lens
        )

    def _validate_cpu_peer(self, peer):
        if not (self.cpu_buffer_enabled or getattr(peer, "cpu_buffer_version", b"")):
            return
        if (
            not self.cpu_buffer_enabled
            or getattr(peer, "cpu_buffer_version", b"") != CPU_BUFFER_VERSION
        ):
            raise ValueError("Both NIXL peers must enable the same CPU-buffer protocol")
        if (
            (getattr(peer, "dst_state_types", []) or [])
            != (getattr(self.kv_args, "state_types", []) or [])
            or peer.decode_tp_size != self.attn_tp_size
            or peer.dst_dcp_size != 1
            or peer.dst_kv_item_lens != self.kv_args.kv_item_lens
            or peer.dst_kv_layer_ids != self.kv_args.kv_layer_ids
            or (getattr(peer, "dst_state_item_lens", []) or [])
            != (getattr(self.kv_args, "state_item_lens", []) or [])
            or (getattr(peer, "dst_state_layer_ids", []) or [])
            != (getattr(self.kv_args, "state_layer_ids", []) or [])
            or (getattr(peer, "dst_state_dim_per_tensor", []) or [])
            != (getattr(self.kv_args, "state_dim_per_tensor", []) or [])
        ):
            raise ValueError("CPU-buffer PD requires matching TP, KV and state layouts")

    def _send_cpu_chunk(self, req, peer, chunk, notif, handles):
        self._validate_cpu_peer(peer)
        if req.cpu_buffer_pages != len(req.dst_kv_indices):
            raise ValueError("Missing or invalid CPU-buffer receive region")
        start = chunk.index_slice.start or 0
        count = len(chunk.prefill_kv_indices)
        self._cpu_pool.reap(self.agent)
        offset, buffer = self._cpu_pool.allocate(count * sum(self.kv_args.kv_item_lens))
        retained = False
        try:
            pack_pages(self._cpu_views, chunk.prefill_kv_indices, buffer)
            src, dst = chunk_descriptors(
                buffer.data_ptr(),
                req.cpu_buffer_ptr,
                self.kv_args.kv_item_lens,
                count,
                req.cpu_buffer_pages,
                start,
            )
            handle = self.agent.initialize_xfer(
                "WRITE",
                self.agent.get_xfer_descs(src, "DRAM"),
                self.agent.get_xfer_descs(dst, "DRAM"),
                req.agent_name,
                notif.encode("ascii"),
            )
            if not handle:
                raise RuntimeError("NIXL failed to initialize CPU-buffer transfer")
            # Record both ownership and the drain barrier BEFORE posting.
            self._cpu_pool.retain(handle, offset)
            retained = True
            handles.append(handle)
            if self.agent.transfer(handle) == "ERR":
                raise RuntimeError("NIXL failed to post CPU-buffer transfer")
        finally:
            if not retained:
                self._cpu_pool.release(offset)

    def _cpu_state_index(self, indices, *, check_bounds=True):
        groups = getattr(self, "_cpu_state_views", [])
        if not groups:
            if indices:
                raise ValueError("Unexpected CPU-buffer state indices")
            return []
        if indices is None or len(indices) != len(groups):
            raise ValueError("Missing CPU-buffer state components")
        result = []
        for state_type, views, ids in zip(
            self.kv_args.state_types, groups, indices, strict=True
        ):
            ids = np.asarray(ids)
            if ids.ndim != 1 or ids.dtype.kind not in "iu":
                raise ValueError("CPU-buffer state indices must be integer vectors")
            if state_type != StateType.QSA_COMPRESSED and len(ids) != 1:
                raise ValueError("CPU-buffer slot state requires one physical slot")
            if len(ids) and (
                np.min(ids) < 0
                or (check_bounds and any(np.max(ids) >= v.shape[0] for v in views))
            ):
                raise ValueError("CPU-buffer state slot out of bounds")
            result.append(ids.copy())
        return result

    def _cpu_state_bytes(self, indices):
        return sum(
            len(ids) * sum(view.shape[1] for view in views)
            for views, ids in zip(self._cpu_state_views, indices, strict=True)
        )

    def _send_cpu_state(self, req, peer, chunk, notif, handles):
        self._validate_cpu_peer(peer)
        indices = self._cpu_state_index(chunk.state_indices)
        if not indices:
            return
        # Remote slots need not fit this worker's pool; their component lengths
        # must match to keep every packed component at the advertised offset.
        destination = self._cpu_state_index(req.dst_state_indices, check_bounds=False)
        if any(
            len(src) != len(dst) for src, dst in zip(indices, destination, strict=True)
        ):
            raise ValueError("CPU-buffer source and destination state counts differ")
        if not req.cpu_buffer_ptr or req.cpu_buffer_pages != len(req.dst_kv_indices):
            raise ValueError("Missing CPU-buffer state receive region")
        size = self._cpu_state_bytes(indices)
        self._cpu_pool.reap(self.agent)
        offset, buffer = self._cpu_pool.allocate(size)
        retained = False
        try:
            position = 0
            for views, ids in zip(self._cpu_state_views, indices, strict=True):
                nbytes = len(ids) * sum(view.shape[1] for view in views)
                pack_pages(views, ids, buffer[position : position + nbytes])
                position += nbytes
            dst = req.cpu_buffer_ptr + req.cpu_buffer_pages * sum(
                self.kv_args.kv_item_lens
            )
            handle = self.agent.initialize_xfer(
                "WRITE",
                self.agent.get_xfer_descs([(buffer.data_ptr(), size, 0)], "DRAM"),
                self.agent.get_xfer_descs([(dst, size, 0)], "DRAM"),
                req.agent_name,
                notif.encode("ascii"),
            )
            if not handle:
                raise RuntimeError(
                    "NIXL failed to initialize CPU-buffer state transfer"
                )
            self._cpu_pool.retain(handle, offset)
            retained = True
            handles.append(handle)
            if self.agent.transfer(handle) == "ERR":
                raise RuntimeError("NIXL failed to post CPU-buffer state transfer")
        finally:
            if not retained:
                self._cpu_pool.release(offset)

    def _allocate_cpu_receive(self, room, indices, required_acks, state_indices=None):
        with self._cpu_receive_lock:
            if room in self._cpu_receives:
                raise ValueError(
                    "CPU-buffer room is still owned by an earlier transfer"
                )
            state_index = self._cpu_state_index(state_indices)
            size = len(indices) * sum(self.kv_args.kv_item_lens)
            size += self._cpu_state_bytes(state_index) if state_index else 0
            if size:
                offset, buffer = self._cpu_pool.allocate(size)
            else:
                offset, buffer = None, self._cpu_pool.buffer[:0]
            region = CpuReceiveRegion(
                offset, buffer, indices.copy(), required_acks, state_index
            )
            self._cpu_receives[room] = region
            return region

    def _finish_cpu_receive(self, room):
        with self._cpu_receive_lock:
            region = self._cpu_receives.get(room)
            if region is None:
                raise RuntimeError("CPU-buffer receive region is missing")
            # A finished decode request may still have a redundant forward in
            # flight on the same physical state slot. Drain that producer before
            # overwriting the slot, not later when constructing a prebuilt batch.
            restore_stream = getattr(self, "_cpu_restore_stream", None)
            if restore_stream is not None:
                torch.cuda.current_stream(restore_stream.device).wait_stream(
                    restore_stream
                )
            kv_bytes = len(region.indices) * sum(self.kv_args.kv_item_lens)
            scatter_pages(self._cpu_views, region.indices, region.buffer[:kv_bytes])
            position = kv_bytes
            for views, ids in zip(
                getattr(self, "_cpu_state_views", []),
                region.state_indices or [],
                strict=True,
            ):
                nbytes = len(ids) * sum(view.shape[1] for view in views)
                scatter_pages(views, ids, region.buffer[position : position + nbytes])
                position += nbytes
            self._release_cpu_receive(room)

    def _release_cpu_receive(self, room):
        with self._cpu_receive_lock:
            region = self._cpu_receives.pop(room, None)
            if region is not None and region.offset is not None:
                self._cpu_pool.release(region.offset)

    def _clear_cpu_receive(self, room):
        with self._cpu_receive_lock:
            region = self._cpu_receives.get(room)
            if region is None:
                return
            region.release_requested = True
            if not region.published or self.is_abort_release_safe(
                room, region.required_acks
            ):
                self._release_cpu_receive(room)
                super().clear_deferred_abort_state(room)

    def note_abort_ack(self, room, prefill_rank, generation):
        if not self.cpu_buffer_enabled:
            return super().note_abort_ack(room, prefill_rank, generation)
        with self._cpu_receive_lock:
            super().note_abort_ack(room, prefill_rank, generation)
            region = self._cpu_receives.get(room)
            if region is not None and region.release_requested:
                self._clear_cpu_receive(room)

    def clear_deferred_abort_state(self, room):
        if not self.cpu_buffer_enabled:
            return super().clear_deferred_abort_state(room)
        with self._cpu_receive_lock:
            # The scheduler's timeout may release GPU slots, but it must not erase
            # the generation needed to reclaim a still-published host destination.
            if room in self._cpu_receives:
                return
            super().clear_deferred_abort_state(room)
