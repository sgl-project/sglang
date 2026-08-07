from __future__ import annotations

import logging
import os
from typing import Dict, Optional

import torch

from sglang.kernels.ops.kv_canary.consts import (
    CANARY_FIELD_REAL_KV_HASH,
    RealKvHashMode,
    splitmix64,
)
from sglang.kernels.ops.kv_canary.verify import CANARY_SLOT_BYTES
from sglang.kernels.ops.kv_canary.verify_ref import _splitmix64_fold_bytes_scalar

logger = logging.getLogger(__name__)


class CanaryPoolHost:
    """Host-side sidecar pool for kv-canary sentinels."""

    def __init__(
        self,
        *,
        device_num_slots: int,
        host_num_slots: int,
        device_bufs: Dict[str, torch.Tensor],
        pin_memory: bool = True,
        can_use_write_back_jit: bool = False,
        can_use_jit: bool = False,
        host_kv_k_buffer: Optional[torch.Tensor] = None,
        host_kv_v_buffer: Optional[torch.Tensor] = None,
        host_kv_layout: Optional[str] = None,
        host_kv_page_size: int = 1,
        host_kv_layer_num: int = 0,
        real_kv_hash_mode: RealKvHashMode = RealKvHashMode.NONE,
        real_kv_read_bytes: int = 0,
        pool_name: str = "",
    ):
        if device_num_slots <= 0:
            raise ValueError(
                f"device_num_slots must be positive, got {device_num_slots}"
            )
        if host_num_slots <= 0:
            raise ValueError(f"host_num_slots must be positive, got {host_num_slots}")
        if host_num_slots < device_num_slots:
            logger.warning(
                "CanaryPoolHost: host_num_slots (%d) < device_num_slots (%d)",
                host_num_slots,
                device_num_slots,
            )
        if not device_bufs:
            raise ValueError("device_bufs must not be empty")

        self.device_num_slots = device_num_slots
        self.host_num_slots = host_num_slots
        self.pin_memory = pin_memory
        self.device_bufs: Dict[str, torch.Tensor] = dict(device_bufs)

        for name, buf in self.device_bufs.items():
            if buf.dtype != torch.uint8:
                raise ValueError(
                    f"device buffer {name!r} must be uint8, got {buf.dtype}"
                )
            if tuple(buf.shape) != (device_num_slots, CANARY_SLOT_BYTES):
                raise ValueError(
                    f"device buffer {name!r} must have shape "
                    f"({device_num_slots}, {CANARY_SLOT_BYTES}), got {tuple(buf.shape)}"
                )

        self.host_bufs: Dict[str, torch.Tensor] = {
            name: torch.zeros(
                host_num_slots,
                CANARY_SLOT_BYTES,
                dtype=torch.uint8,
                pin_memory=pin_memory,
            )
            for name in self.device_bufs
        }

        self.can_use_write_back_jit = can_use_write_back_jit
        self.can_use_jit = can_use_jit

        self._host_kv_k_buffer = host_kv_k_buffer
        self._host_kv_v_buffer = host_kv_v_buffer
        self._host_kv_layout = host_kv_layout
        self._host_kv_page_size = host_kv_page_size
        self._host_kv_layer_num = host_kv_layer_num
        self._real_kv_hash_mode = real_kv_hash_mode
        self._real_kv_read_bytes = real_kv_read_bytes
        self._pool_name = pool_name or "CANARY"
        if real_kv_hash_mode == RealKvHashMode.PARTIAL:
            self._effective_read_bytes = 16
        elif real_kv_hash_mode == RealKvHashMode.ALL:
            self._effective_read_bytes = 0
        else:
            self._effective_read_bytes = 0

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ) -> None:
        """D->H: copy sentinels at device_indices into host slots."""
        if host_indices.numel() == 0:
            return
        if host_indices.numel() != device_indices.numel():
            raise ValueError("host_indices and device_indices must have equal length")

        for name, dev in self.device_bufs.items():
            host_buf = self.host_bufs[name]
            h_idx = host_indices.to(host_buf.device)
            d_idx = device_indices.to(dev.device)
            host_buf[h_idx] = dev[d_idx].to(host_buf.device)

        self._verify_backup_real_kv_hash(host_indices=host_indices)

    def load_to_device_per_layer(
        self, device_pool, host_indices, device_indices, layer_id, io_backend
    ) -> None:
        """H->D: copy sentinels from host slots back to device_indices."""
        if host_indices.numel() == 0:
            return
        if host_indices.numel() != device_indices.numel():
            raise ValueError("host_indices and device_indices must have equal length")
        for name, dev in self.device_bufs.items():
            host_buf = self.host_bufs[name]
            h_idx = host_indices.to(host_buf.device)
            d_idx = device_indices.to(dev.device)
            dev[d_idx] = host_buf[h_idx].to(dev.device)

    def _get_slot_kv_bytes(
        self, buffer: torch.Tensor, slot_idx: int, layer: int = 0
    ) -> Optional[list[int]]:
        """Extract raw bytes of a single slot from a host KV buffer."""
        if buffer is None or self._host_kv_layout is None:
            return None

        layout = self._host_kv_layout
        page_size = self._host_kv_page_size

        if layout == "layer_first":
            slot_data = buffer[layer, slot_idx]
        elif layout == "page_first":
            slot_data = buffer[slot_idx, layer]
        elif layout == "page_first_direct":
            page_idx = slot_idx // page_size
            page_off = slot_idx % page_size
            slot_data = buffer[page_idx, layer, page_off]
        elif layout == "page_head":
            page_idx = slot_idx // page_size
            page_off = slot_idx % page_size
            slot_data = buffer[page_idx, :, page_off, layer]
        else:
            return None

        slot_u8 = slot_data.contiguous().view(torch.uint8).reshape(-1)
        return slot_u8.tolist()

    def _compute_slot_real_kv_hash(
        self, buffer: torch.Tensor, slot_idx: int
    ) -> Optional[int]:
        """Re-compute real_kv_hash for one slot, matching the GPU-side formula."""
        raw_bytes = self._get_slot_kv_bytes(buffer, slot_idx, layer=0)
        if raw_bytes is None or len(raw_bytes) == 0:
            return None

        if self._real_kv_hash_mode == RealKvHashMode.PARTIAL:
            effective = min(16, len(raw_bytes))
        elif self._real_kv_hash_mode == RealKvHashMode.ALL:
            effective = len(raw_bytes)
        else:
            return None

        raw_bytes = raw_bytes[:effective]
        folded = _splitmix64_fold_bytes_scalar(raw_bytes=raw_bytes)
        return splitmix64(folded)

    @staticmethod
    def _read_sentinel_field(
        host_buf: torch.Tensor, slot_idx: int, field_idx: int
    ) -> int:
        """Read a uint64 field from a canary sentinel slot (little-endian)."""
        byte_off = field_idx * 8
        field_bytes = host_buf[slot_idx, byte_off : byte_off + 8].tolist()
        return int.from_bytes(bytes(field_bytes), byteorder="little", signed=False)

    def _verify_backup_real_kv_hash(
        self,
        *,
        host_indices: torch.Tensor,
    ) -> None:
        """Verify backed-up sentinel hashes against host KV data."""
        if self._real_kv_hash_mode == RealKvHashMode.NONE:
            return
        if self._host_kv_k_buffer is None:
            return
        if not os.environ.get("CANARY_VERIFY_BACKUP"):
            return

        if self.can_use_write_back_jit:
            torch.cuda.synchronize()
        else:
            torch.cuda.current_stream().synchronize()

        k_tail = self.host_bufs.get("k_tail")
        v_tail = self.host_bufs.get("v_tail")
        if k_tail is None:
            return

        h_indices = host_indices.to(k_tail.device)
        n_slots = int(h_indices.numel())

        mismatches = 0
        for i in range(n_slots):
            h_slot = int(h_indices[i].item())
            if h_slot < 0 or h_slot >= self.host_num_slots:
                logger.warning(
                    "CANARY_VERIFY_BACKUP: slot %d out of range [0, %d), skip",
                    h_slot,
                    self.host_num_slots,
                )
                continue

            stored_k = self._read_sentinel_field(
                k_tail, h_slot, CANARY_FIELD_REAL_KV_HASH
            )
            computed_k = self._compute_slot_real_kv_hash(self._host_kv_k_buffer, h_slot)
            if computed_k is not None and stored_k != computed_k:
                mismatches += 1
                logger.error(
                    "CANARY_VERIFY_BACKUP MISMATCH [%s]: k_tail slot=%d "
                    "stored=0x%016X computed=0x%016X",
                    self._pool_name,
                    h_slot,
                    stored_k,
                    computed_k,
                )
            if v_tail is not None and self._host_kv_v_buffer is not None:
                stored_v = self._read_sentinel_field(
                    v_tail, h_slot, CANARY_FIELD_REAL_KV_HASH
                )
                computed_v = self._compute_slot_real_kv_hash(
                    self._host_kv_v_buffer, h_slot
                )
                if computed_v is not None and stored_v != computed_v:
                    mismatches += 1
                    logger.error(
                        "CANARY_VERIFY_BACKUP MISMATCH [%s]: v_tail slot=%d "
                        "stored=0x%016X computed=0x%016X",
                        self._pool_name,
                        h_slot,
                        stored_v,
                        computed_v,
                    )

        if mismatches > 0:
            msg = (
                f"CANARY_VERIFY_BACKUP: {mismatches} hash mismatch(es) across "
                f"{n_slots} backed-up slots"
            )
            raise RuntimeError(msg)

    def clear(self) -> None:
        for buf in self.host_bufs.values():
            buf.zero_()
