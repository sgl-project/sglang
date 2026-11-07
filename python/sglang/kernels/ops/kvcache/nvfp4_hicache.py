# SPDX-License-Identifier: Apache-2.0
"""Page-id preparation for paged MiniMax NVFP4 HiCache transfers."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _validate_pages(
    HI, DI, HP, DP, VALID, N, HC, DC, HS: tl.constexpr, DS: tl.constexpr
):
    offset = tl.program_id(0) * 1024 + tl.arange(0, 1024)
    mask = offset < N
    start = offset // 128 * 128
    h = tl.load(HI + offset * HS, mask, 0).to(tl.int64)
    d = tl.load(DI + offset * DS, mask, 0).to(tl.int64)
    hs = tl.load(HI + start * HS, mask, 0).to(tl.int64)
    ds = tl.load(DI + start * DS, mask, 0).to(tl.int64)
    valid = (
        (hs % 128 == 0)
        & (ds % 128 == 0)
        & (h == hs + offset % 128)
        & (d == ds + offset % 128)
        & (h >= 0)
        & (h < HC)
        & (d >= 0)
        & (d < DC)
    )
    tl.store(VALID + tl.program_id(0), tl.sum((valid | ~mask).to(tl.int32), 0) == 1024)
    first = mask & (offset % 128 == 0)
    tl.store(HP + offset // 128, hs // 128, first)
    tl.store(DP + offset // 128, ds // 128, first)


def prepare_nvfp4_pages(host_indices, device_indices, host_capacity, device_capacity):
    """Validate full 128-token pages on one CUDA device, then return their IDs."""
    if (
        host_indices.ndim != 1
        or device_indices.ndim != 1
        or host_indices.numel() != device_indices.numel()
        or host_indices.numel() % 128
    ):
        raise ValueError(
            "MiniMax NVFP4 HiCache transfers require equal, full-page index lists."
        )
    if any(
        x.dtype not in (torch.int32, torch.int64)
        for x in (host_indices, device_indices)
    ):
        raise ValueError("MiniMax NVFP4 HiCache requires integer indices.")
    if not host_indices.is_cuda or host_indices.device != device_indices.device:
        raise ValueError("NVFP4 page preparation requires same-device CUDA indices.")
    count = host_indices.numel()
    h = torch.empty(count // 128, device=host_indices.device, dtype=host_indices.dtype)
    d = torch.empty(
        count // 128, device=device_indices.device, dtype=device_indices.dtype
    )
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
            num_warps=4,
        )
        torch._assert_async(
            flags.all(),
            "MiniMax NVFP4 HiCache requires aligned, ordered, in-range full pages.",
        )
    return h, d
