# SPDX-License-Identifier: Apache-2.0
"""Metadata entry for a fresh, complete Step5 CSA region; CUDA arithmetic only."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def _storage_overlaps(left: torch.Tensor, right: torch.Tensor) -> bool:
    if left.device != right.device or not left.numel() or not right.numel():
        return False
    a, b = left.untyped_storage(), right.untyped_storage()
    return max(a.data_ptr(), b.data_ptr()) < min(
        a.data_ptr() + a.nbytes(), b.data_ptr() + b.nbytes()
    )


def _validate_buffers(
    index_key: torch.Tensor, compression_logits: torch.Tensor, out: torch.Tensor
) -> int:
    """Validate host-visible metadata on CPU/CUDA without reading tensor values.

    CPU acceptance here only permits metadata tests. The public computation
    entry separately requires NVIDIA CUDA and has no CPU arithmetic fallback.
    """
    import torch

    for name, value in (
        ("index_key", index_key),
        ("compression_logits", compression_logits),
    ):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        if (
            value.layout != torch.strided
            or value.device.type not in ("cpu", "cuda")
            or value.dtype != torch.bfloat16
            or value.ndim != 3
            or value.shape[1:] != (8, 256)
            or not value.is_contiguous()
            or value.requires_grad
        ):
            raise ValueError(
                f"{name} must be inference-only contiguous BF16 [regions, 8, 256]"
            )
    if (
        index_key.shape != compression_logits.shape
        or index_key.device != compression_logits.device
    ):
        raise ValueError(
            "index_key and compression_logits must have the same shape and device"
        )
    if not isinstance(out, torch.Tensor):
        raise TypeError("out must be a tensor")
    if (
        out.layout != torch.strided
        or out.device != index_key.device
        or out.dtype != torch.uint8
        or out.shape != (index_key.shape[0], 256)
        or not out.is_contiguous()
        or out.requires_grad
    ):
        raise ValueError(
            "out must be contiguous uint8 [regions, 256] on the input device"
        )
    if _storage_overlaps(out, index_key) or _storage_overlaps(out, compression_logits):
        raise ValueError("out must not share storage memory with either input")
    return index_key.shape[0]


def step5_csa_region(
    index_key: torch.Tensor, compression_logits: torch.Tensor, *, out: torch.Tensor
) -> torch.Tensor:
    """Write logical E4M3 bytes for complete fresh R8/H1/D256 regions.

    Inputs are BF16 [N, 8, 256]; out is caller-owned uint8 [N, 256]. All are
    contiguous; nonzero storage offsets are supported. Input buffers may alias
    each other, but output storage must be disjoint from both. No page swizzle,
    old accumulator, partial mask, request state or allocation is performed.

    One scalar maximum spans all eight tokens and all 256 channels. Each
    channel accumulates weighted K and weights in offset order, uses a 1e-20
    denominator floor, then converts FP32 -> BF16 -> E4M3 RN satfinite.
    Resident K/Z must be finite; values are not inspected on the host. Finite
    inputs can still overflow FP32 intermediates. No broad overflow-domain or
    source-equivalence qualification is implied by this entry's metadata checks.

    The first numerator term uses separate non-FTZ RN multiply and add(+0);
    the other seven updates use ordered FP32 FMA. Weights use exp2, followed
    by ordered FP32 denominator additions and RN division. This fresh-state
    contract does not apply to an existing accumulator or incomplete region.

    Empty N returns out without importing/launching the Triton leaf. Warm up
    before CUDA Graph capture; subsequent calls use the supplied writer buffer
    and perform no data-dependent host reads or scratch allocation.
    """
    import torch

    regions = _validate_buffers(index_key, compression_logits, out)
    if index_key.device.type != "cuda" or torch.version.hip is not None:
        raise ValueError("Step5 CSA region arithmetic requires NVIDIA CUDA")
    if regions == 0:
        return out
    from sglang.kernels.ops.attention.step5.csa_region_triton import (
        _launch_step5_csa_region,
    )

    _launch_step5_csa_region(index_key, compression_logits, out)
    return out
