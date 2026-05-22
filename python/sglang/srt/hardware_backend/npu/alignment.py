# Copyright 2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""2 MB alignment helpers for CANN HCCL IPC RMA registration.

torch_npu's caching allocator can split 2 MB-aligned raw blocks into
misaligned sub-blocks; we over-allocate by one alignment block and return
a view starting at the next 2 MB boundary. Non-NPU devices pass through
to ``torch.zeros``. This module deliberately does NOT import ``torch_npu``
so it stays cheap to import from any platform.
"""

from __future__ import annotations

from numbers import Integral
from typing import Any, List, Tuple, Union

import torch

from sglang.srt.utils import is_npu

ALIGNMENT_BLOCK_2M = 2 * 1024 * 1024


def _device_type(device: Union[str, torch.device, None]) -> str:
    """Bare device type ('npu' / 'cuda' / 'cpu' / ...)."""
    if device is None:
        return ""
    if isinstance(device, torch.device):
        return device.type
    return str(device).split(":", 1)[0]


def _is_npu_device(device: Any) -> bool:
    return is_npu() and _device_type(device) == "npu"


def zeros_2m_aligned(shape, dtype, device) -> torch.Tensor:
    """``torch.zeros`` with 2 MB-aligned ``data_ptr()`` on NPU; passthrough elsewhere."""
    if not _is_npu_device(device):
        return torch.zeros(shape, dtype=dtype, device=device)

    if isinstance(shape, Integral):
        view_shape = (int(shape),)
    else:
        view_shape = tuple(int(s) for s in shape)
    n_elems = 1
    for s in view_shape:
        n_elems *= s

    elem_size = torch.empty(0, dtype=dtype).element_size()
    if ALIGNMENT_BLOCK_2M % elem_size != 0:
        raise RuntimeError(f"ALIGNMENT_BLOCK_2M not divisible by elem_size={elem_size}")

    # Over-allocate 2 MB so a forward shift up to (2 MB - 1) bytes still
    # leaves >= requested_bytes of aligned region.
    requested_bytes = n_elems * elem_size
    mem_size = requested_bytes + ALIGNMENT_BLOCK_2M
    flat = torch.zeros(mem_size // elem_size, dtype=dtype, device=device)

    addr = flat.data_ptr()
    aligned_addr = (
        (addr + ALIGNMENT_BLOCK_2M - 1) // ALIGNMENT_BLOCK_2M * ALIGNMENT_BLOCK_2M
    )
    head_offset_bytes = aligned_addr - addr
    aligned_mem_size = mem_size - head_offset_bytes

    if head_offset_bytes % elem_size != 0:
        raise RuntimeError(f"NPU base 0x{addr:x} not aligned to elem_size={elem_size}")
    if aligned_mem_size < requested_bytes:
        raise RuntimeError(
            f"Aligned region too small: aligned_mem_size={aligned_mem_size}"
            f" < requested_bytes={requested_bytes}"
        )

    head_offset_elems = head_offset_bytes // elem_size
    out = flat[head_offset_elems : head_offset_elems + n_elems].view(view_shape)

    if out.data_ptr() != aligned_addr:
        raise RuntimeError(
            f"Self-align view data_ptr mismatch: aligned=0x{aligned_addr:x}"
            f" out=0x{out.data_ptr():x}"
        )
    if out.data_ptr() % ALIGNMENT_BLOCK_2M != 0:
        raise RuntimeError(f"Self-align failed: out=0x{out.data_ptr():x}")
    # `out` keeps `flat`'s storage alive via tensor refcount.
    return out


def zeros_2m_aligned_segments(
    num_segments: int,
    segment_shape,
    dtype: torch.dtype,
    device,
) -> Tuple[None, List[torch.Tensor]]:
    """N independent 2 MB-aligned tensors (one per layer for KV registration).

    Each segment owns its own underlying allocation. The returned ``owner``
    is ``None`` and exists only for API symmetry with earlier shared-owner
    variants.
    """
    if not _is_npu_device(device):
        return None, [
            torch.zeros(segment_shape, dtype=dtype, device=device)
            for _ in range(num_segments)
        ]

    if isinstance(segment_shape, Integral):
        segment_shape = (int(segment_shape),)
    else:
        segment_shape = tuple(int(s) for s in segment_shape)

    segments: List[torch.Tensor] = []
    for i in range(num_segments):
        seg = zeros_2m_aligned(segment_shape, dtype, device)
        if seg.data_ptr() % ALIGNMENT_BLOCK_2M != 0:
            raise RuntimeError(f"segment {i} not aligned: ptr=0x{seg.data_ptr():x}")
        if not seg.is_contiguous():
            raise RuntimeError(f"segment {i} unexpectedly non-contiguous")
        segments.append(seg)
    return None, segments


def assert_2m_aligned_kv_args(kv_args) -> None:
    """Fail fast on misaligned kv/aux/state ptrs before HCCL IPC RMA registration."""
    groups = (
        ("kv_data_ptrs", getattr(kv_args, "kv_data_ptrs", None)),
        ("aux_data_ptrs", getattr(kv_args, "aux_data_ptrs", None)),
        ("state_data_ptrs", getattr(kv_args, "state_data_ptrs", None)),
    )
    misaligned = [
        (name, i, ptr)
        for name, ptrs in groups
        if ptrs
        for i, ptr in enumerate(ptrs)
        if ptr % ALIGNMENT_BLOCK_2M != 0
    ]
    if misaligned:
        details = ", ".join(f"{name}[{i}]=0x{ptr:x}" for name, i, ptr in misaligned[:8])
        raise RuntimeError(
            f"NPU PD: {len(misaligned)} buffer(s) not 2 MB-aligned. First: {details}."
        )
