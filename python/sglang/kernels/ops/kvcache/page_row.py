# SPDX-License-Identifier: Apache-2.0
"""Opaque page-row transfers between pinned host memory and paged device pools."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

from sglang.kernels.jit.utils import (
    cache_once,
    is_hip_runtime,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# Sixteen 16 B loads in flight per lane measured fastest on B300 PCIe for MSA
# rows: ~51 GB/s H2D and ~52 GB/s D2H, vs 46 / 35 GB/s with 8 / 32 per lane.
_VECS_PER_LANE = 16
_BLOCK_SIZE = 512
# Two CTAs saturate B300 PCIe; MI350X needs 16 or more (measured on gfx950).
_BLOCK_QUOTA = 32 if is_hip_runtime() else 2


@cache_once
def _jit_page_row_module(load: bool, block_quota: int) -> Module:
    args = make_cpp_args(load, _VECS_PER_LANE, _BLOCK_SIZE, block_quota)
    return load_jit(
        "page_row",
        *args,
        cuda_files=["kvcacheio/page_row.cuh"],
        cuda_wrappers=[("run", f"&PageRowKernel<{args}>::run")],
    )


def warmup_page_row_transfer(block_quota: int = _BLOCK_QUOTA) -> None:
    """Compile both directions before requests can submit transfers, so compile
    errors surface at pool construction rather than mid-serving."""
    _jit_page_row_module(True, block_quota)
    _jit_page_row_module(False, block_quota)


def transfer_page_rows(
    host_ptrs: torch.Tensor,
    device_ptrs: torch.Tensor,
    host_pages: torch.Tensor,
    device_pages: torch.Tensor,
    *,
    host_stride: int,
    device_stride: int,
    row_bytes: int,
    load: bool,
    block_quota: int = _BLOCK_QUOTA,
) -> None:
    """Copy ``row_bytes`` per (layer, page) on the current stream.

    ``host_ptrs`` / ``device_ptrs`` are CUDA uint64 ``[layers]`` row-0 addresses
    (host addresses are mapped pinned memory); page ``p`` of a layer starts at
    ``ptr + p * stride``. Pages are validated, same-device CUDA int32/int64 ids.
    Bases, strides and ``row_bytes`` must be 128-byte aligned. The caller owns
    buffers and indices until its stream completion event fires.
    """
    if host_pages.numel() == 0 or host_ptrs.numel() == 0:
        return
    # Positional: kwargs make this per-launch cache lookup measurably slower.
    _jit_page_row_module(load, block_quota).run(
        host_ptrs,
        device_ptrs,
        host_pages,
        device_pages,
        host_stride,
        device_stride,
        row_bytes,
    )


@triton.jit
def _validate_pages(
    HI, DI, HP, DP, OK, N, HC, DC, HS: tl.constexpr, DS: tl.constexpr, PS: tl.constexpr
):
    offset = tl.program_id(0) * 1024 + tl.arange(0, 1024)
    mask = offset < N
    start = offset // PS * PS
    h = tl.load(HI + offset * HS, mask, 0).to(tl.int64)
    d = tl.load(DI + offset * DS, mask, 0).to(tl.int64)
    hs = tl.load(HI + start * HS, mask, 0).to(tl.int64)
    ds = tl.load(DI + start * DS, mask, 0).to(tl.int64)
    valid = (
        (hs % PS == 0)
        & (ds % PS == 0)
        & (h == hs + offset % PS)
        & (d == ds + offset % PS)
        & (h >= 0)
        & (h < HC)
        & (d >= 0)
        & (d < DC)
    )
    tl.store(OK + tl.program_id(0), tl.sum((valid | ~mask).to(tl.int32), 0) == 1024)
    first = mask & (offset % PS == 0)
    tl.store(HP + offset // PS, hs // PS, first)
    tl.store(DP + offset // PS, ds // PS, first)


def prepare_page_ids(
    host_indices, device_indices, host_capacity, device_capacity, *, page_size
):
    """Validate full pages of token slots on one CUDA device, then return their IDs."""
    if (
        host_indices.ndim != 1
        or device_indices.ndim != 1
        or host_indices.numel() != device_indices.numel()
        or host_indices.numel() % page_size
    ):
        raise ValueError("Page-row transfers require equal, full-page index lists.")
    if any(
        x.dtype not in (torch.int32, torch.int64)
        for x in (host_indices, device_indices)
    ):
        raise ValueError("Page-row transfers require integer indices.")
    if not host_indices.is_cuda or host_indices.device != device_indices.device:
        raise ValueError("Page-id preparation requires same-device CUDA indices.")
    count = host_indices.numel()
    pages = count // page_size
    h = torch.empty(pages, device=host_indices.device, dtype=host_indices.dtype)
    d = torch.empty(pages, device=device_indices.device, dtype=device_indices.dtype)
    if count:
        flags = torch.empty(
            triton.cdiv(count, 1024), device=host_indices.device, dtype=torch.bool
        )
        _validate_pages[(flags.numel(),)](
            host_indices,
            device_indices,
            h,
            d,
            flags,
            count,
            host_capacity,
            device_capacity,
            host_indices.stride(0),
            device_indices.stride(0),
            page_size,
            num_warps=4,
        )
        torch._assert_async(
            flags.all(),
            "Page-row transfers require aligned, ordered, in-range full pages.",
        )
    return h, d
