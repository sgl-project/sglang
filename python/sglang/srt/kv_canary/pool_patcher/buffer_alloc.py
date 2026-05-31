from __future__ import annotations

import sys
from typing import Tuple

import torch

from sglang.jit_kernel.kv_canary.consts import RealKvHashMode
from sglang.jit_kernel.kv_canary.verify import (
    CANARY_SLOT_BYTES,
    RealKvSource,
)
from sglang.srt.kv_canary.config import CanaryConfig

_PARTIAL_REAL_KV_READ_BYTES = 16
_REAL_KV_READ_ALIGN = 16


def resolve_real_kv_read_bytes(config: CanaryConfig) -> int:
    if config.real_kv_hash_mode is RealKvHashMode.NONE:
        return 0
    if config.real_kv_hash_mode is RealKvHashMode.ALL:
        return sys.maxsize
    return _PARTIAL_REAL_KV_READ_BYTES


def alloc_canary_buf(
    *,
    num_slots: int,
    device: torch.device,
) -> torch.Tensor:
    return torch.zeros(num_slots, CANARY_SLOT_BYTES, dtype=torch.uint8, device=device)


def _clip_read_bytes_aligned(*, requested: int, num_bytes_per_token: int) -> int:
    """Validate and clip read_bytes for the CUDA fold kernel's 128-bit aligned loads.

    Normalizes sentinels (``sys.maxsize`` -> ``num_bytes_per_token``, ``0`` -> ``0``) and
    rejects negative / unaligned / oversized requests.
    """
    if num_bytes_per_token <= 0 or num_bytes_per_token % _REAL_KV_READ_ALIGN != 0:
        raise ValueError(
            "kv-canary: num_bytes_per_token must be a positive multiple of "
            f"{_REAL_KV_READ_ALIGN}, got {num_bytes_per_token}"
        )
    if requested == 0:
        return 0
    if requested == sys.maxsize:
        return num_bytes_per_token
    if requested < 0:
        raise ValueError(f"kv-canary: read_bytes must be non-negative, got {requested}")
    if requested > num_bytes_per_token:
        raise ValueError(
            "kv-canary: read_bytes must be <= num_bytes_per_token "
            f"({num_bytes_per_token}), got {requested}"
        )
    if requested % _REAL_KV_READ_ALIGN != 0:
        raise ValueError(
            "kv-canary: read_bytes must be a multiple of "
            f"{_REAL_KV_READ_ALIGN}, got {requested}"
        )
    return requested


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
    clipped = _clip_read_bytes_aligned(
        requested=read_bytes, num_bytes_per_token=num_bytes_per_token
    )
    if clipped == 0:
        return ()
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
    clipped = _clip_read_bytes_aligned(
        requested=read_bytes, num_bytes_per_token=bytes_per_token
    )
    if clipped == 0:
        return ()
    return (
        RealKvSource(
            tensor=flat,
            page_size=page_size,
            num_bytes_per_token=bytes_per_token,
            read_bytes=clipped,
        ),
    )
