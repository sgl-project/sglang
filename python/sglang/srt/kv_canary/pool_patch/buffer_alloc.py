from __future__ import annotations

import sys
from typing import Tuple

import torch

from sglang.jit_kernel.kv_canary.verify import (
    CANARY_SLOT_BYTES,
    RealKvHashMode,
    RealKvSource,
)
from sglang.srt.kv_canary.config import CanaryConfig

_PARTIAL_REAL_KV_READ_BYTES = 32


def resolve_read_bytes(config: CanaryConfig) -> int:
    if config.real_kv_hash_mode is RealKvHashMode.OFF:
        return 0
    if config.real_kv_hash_mode is RealKvHashMode.ALL:
        return sys.maxsize
    return _PARTIAL_REAL_KV_READ_BYTES


def alloc_canary_buf(
    *,
    num_slots: int,
    device: torch.device,
) -> torch.Tensor:
    """Allocate one canary buffer (head or tail) of shape ``[num_slots, CANARY_SLOT_BYTES]``."""
    return torch.zeros(num_slots, CANARY_SLOT_BYTES, dtype=torch.uint8, device=device)


def make_row_source(
    *,
    layer_buffer: torch.Tensor,
    read_bytes: int,
) -> Tuple[RealKvSource, ...]:
    contiguous = layer_buffer.contiguous()
    num_slots = int(contiguous.shape[0])
    if num_slots == 0 or read_bytes == 0:
        return ()
    flat = contiguous.view(torch.uint8).reshape(num_slots, -1)
    num_bytes_per_token = int(flat.shape[1])
    clipped = max(0, min(int(read_bytes), num_bytes_per_token))
    return (
        RealKvSource(
            tensor=flat,
            page_size=1,
            num_bytes_per_token=num_bytes_per_token,
            read_bytes=clipped,
        ),
    )


def make_packed_source(
    *,
    page_buffer: torch.Tensor,
    page_size: int,
    bytes_per_token: int,
    read_bytes: int,
) -> Tuple[RealKvSource, ...]:
    if read_bytes == 0 or page_buffer.numel() == 0:
        return ()
    flat = page_buffer.contiguous().view(torch.uint8)
    if flat.ndim == 1:
        flat = flat.reshape(1, -1)
    clipped = max(0, min(int(read_bytes), bytes_per_token))
    return (
        RealKvSource(
            tensor=flat,
            page_size=page_size,
            num_bytes_per_token=bytes_per_token,
            read_bytes=clipped,
        ),
    )
