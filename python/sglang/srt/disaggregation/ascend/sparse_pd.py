"""Ascend-only helpers for temporary sparse KV PD disaggregation.

This module keeps sparse-PD transfer bookkeeping inside the Ascend backend so
the generic prefill/decode disaggregation code can continue to pass normal page
indices. The destination indices are rewritten to point at an Ascend HBM staging
buffer, then the staging buffer is copied into sparse KV host SHM after transfer
success.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
import numpy.typing as npt

from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.config import (
    resolve_sparse_kv_offload_mode,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SparsePDTransferMetadata:
    room: int
    slot_id: int
    req_pool_idx: int
    token_count: int
    page_count: int
    decode_prefix_len: int


def get_sparse_pd_manager():
    try:
        from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.manager import (
            get_sparse_kv_manager,
        )
    except ImportError:
        return None
    return get_sparse_kv_manager()


def is_sparse_pd_decode_enabled(sparse_kv_manager=None) -> bool:
    if sparse_kv_manager is None:
        sparse_kv_manager = get_sparse_pd_manager()
    return (
        sparse_kv_manager is not None
        and resolve_sparse_kv_offload_mode().uses_pd_decode_staging
    )


class SparsePDDecodeStagingPool:
    """Single-process D-side staging slot pool for Ascend sparse PD."""

    def __init__(self, manager, page_size: int, slot_count: int = 1):
        self.manager = manager
        self.page_size = int(page_size)
        self.slot_count = int(slot_count)
        if self.page_size <= 0:
            raise ValueError(f"Sparse PD page_size must be > 0: {self.page_size}")
        if self.slot_count <= 0:
            raise ValueError(
                f"Sparse PD staging slot_count must be > 0: {self.slot_count}"
            )

        self.manager.ensure_pd_decode_staging_buffers(
            slot_count=self.slot_count,
            page_size=self.page_size,
        )
        self.pages_per_slot = self.manager.get_pd_decode_pages_per_slot(
            page_size=self.page_size
        )
        self._free_slots = deque(range(self.slot_count))
        self._room_to_slot: dict[int, int] = {}
        self._room_to_metadata: dict[int, SparsePDTransferMetadata] = {}
        self._lock = threading.Lock()

    def has_room(self, room: int) -> bool:
        with self._lock:
            return int(room) in self._room_to_slot

    def available_slots(self) -> int:
        with self._lock:
            return len(self._free_slots)

    def get_slot(self, room: int) -> Optional[int]:
        with self._lock:
            return self._room_to_slot.get(int(room))

    def try_acquire_slot(self, room: int) -> Optional[int]:
        room = int(room)
        with self._lock:
            existing = self._room_to_slot.get(room)
            if existing is not None:
                return existing
            if not self._free_slots:
                return None
            slot_id = int(self._free_slots.popleft())
            self._room_to_slot[room] = slot_id
            return slot_id

    def acquire_slot(self, room: int) -> int:
        slot_id = self.try_acquire_slot(room)
        if slot_id is None:
            raise RuntimeError(
                "No free sparse KV PD decode staging slot is available for "
                f"bootstrap_room={int(room)}."
            )
        return slot_id

    def wait_acquire_slot(
        self,
        room: int,
        polling_interval_s: float = 0.001,
    ) -> int:
        while True:
            slot_id = self.try_acquire_slot(room)
            if slot_id is not None:
                return slot_id
            time.sleep(polling_interval_s)

    def release_room(self, room: int) -> None:
        room = int(room)
        with self._lock:
            slot_id = self._room_to_slot.pop(room, None)
            self._room_to_metadata.pop(room, None)
            if slot_id is not None and slot_id not in self._free_slots:
                self._free_slots.append(slot_id)
        self.manager.clear_pd_request_metadata(bootstrap_room=room)

    def release_all(self) -> None:
        with self._lock:
            self._room_to_slot.clear()
            self._room_to_metadata.clear()
            self._free_slots = deque(range(self.slot_count))
        self.manager.clear_all_pd_request_metadata()

    def rewrite_dst_indices(
        self,
        room: int,
        dst_kv_indices: npt.NDArray[np.int32],
        decode_prefix_len: int = 0,
        wait_for_slot: bool = False,
    ) -> npt.NDArray[np.int32]:
        room = int(room)
        decode_prefix_len = int(decode_prefix_len or 0)
        if decode_prefix_len != 0:
            raise RuntimeError(
                "Ascend sparse KV PD does not support decode-side prefix cache yet; "
                f"got decode_prefix_len={decode_prefix_len} for bootstrap_room={room}."
            )

        req_pool_idx, token_count = self.manager.get_pd_copy_metadata(
            room,
            decode_prefix_len=decode_prefix_len,
        )
        page_count = int(len(dst_kv_indices))
        required_pages = (int(token_count) + self.page_size - 1) // self.page_size
        if page_count < required_pages:
            raise RuntimeError(
                "Sparse KV PD got too few destination pages for staging: "
                f"room={room}, page_count={page_count}, "
                f"required_pages={required_pages}, token_count={token_count}."
            )
        if page_count > self.pages_per_slot:
            raise RuntimeError(
                "Sparse KV PD destination page count exceeds staging slot capacity: "
                f"room={room}, page_count={page_count}, "
                f"pages_per_slot={self.pages_per_slot}."
            )

        slot_id = (
            self.wait_acquire_slot(room) if wait_for_slot else self.acquire_slot(room)
        )
        metadata = SparsePDTransferMetadata(
            room=room,
            slot_id=slot_id,
            req_pool_idx=int(req_pool_idx),
            token_count=int(token_count),
            page_count=page_count,
            decode_prefix_len=decode_prefix_len,
        )
        with self._lock:
            self._room_to_metadata[room] = metadata

        page_start = slot_id * self.pages_per_slot
        return np.arange(
            page_start,
            page_start + page_count,
            dtype=np.int32,
        )

    def get_transfer_metadata(self, room: int) -> SparsePDTransferMetadata:
        room = int(room)
        with self._lock:
            metadata = self._room_to_metadata.get(room)
        if metadata is None:
            raise RuntimeError(
                f"Sparse KV PD transfer metadata for bootstrap_room={room} is missing."
            )
        return metadata

    def offload_room_to_host(
        self,
        room: int,
        release: bool = True,
    ) -> SparsePDTransferMetadata:
        metadata = self.get_transfer_metadata(room)
        try:
            self.manager.offload_pd_decode_staging_to_host(
                slot_id=metadata.slot_id,
                req_pool_idx=metadata.req_pool_idx,
                token_count=metadata.token_count,
            )
        finally:
            if release:
                self.release_room(room)
        return metadata
