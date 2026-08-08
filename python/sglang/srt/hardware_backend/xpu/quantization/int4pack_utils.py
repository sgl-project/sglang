# SPDX-License-Identifier: Apache-2.0
"""Helpers for lowering GPTQ/AWQ int4 weights to the torch XPU int4pack layout."""

from __future__ import annotations

import torch

# AutoAWQ packs 8 nibbles per int32 in this interleaved order; reversing it
# recovers natural column order. Matches reverse_awq_pack_order used in
# moe_wna16.convert_awq_tensor and awq_triton.
AWQ_REVERSE_PACK_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]

# group sizes accepted by _weight_int4pack_mm_xpu (verified empirically).
SUPPORTED_GROUP_SIZES = (16, 32, 64, 128, 256)


def pack_int4_to_uint8(q: torch.Tensor) -> torch.Tensor:
    """Pack an ``[N, K]`` tensor of codes ``q in [0, 15]`` into ``[N, K // 2]``.

    Low nibble holds even ``k``, high nibble holds odd ``k`` (torch int4pack B).
    """
    assert q.shape[-1] % 2 == 0, "K must be even to pack into int4 bytes"
    q = q.to(torch.uint8)
    low = q[..., 0::2]
    high = q[..., 1::2]
    return (low | (high << 4)).contiguous()


def unpack_awq_to_codes(packed: torch.Tensor, rows: int) -> torch.Tensor:
    """Deinterleave AWQ-packed int32 ``[rows, cols]`` into codes ``[rows, cols*8]``.

    Codes are in ``[0, 15]`` and restored to natural (non-interleaved) order.
    """
    t = packed.contiguous().view(torch.uint8)  # [rows, cols * 4]
    shifter = torch.tensor([0, 4], dtype=torch.uint8, device=t.device)
    t = (t[:, :, None] >> shifter) & 0xF  # [rows, cols * 4, 2]
    t = t.view(-1, 8)[:, AWQ_REVERSE_PACK_ORDER]  # undo interleave
    return t.reshape(rows, -1)  # [rows, cols * 8]


def _nibble_shifts(device: torch.device) -> torch.Tensor:
    return torch.arange(0, 32, 4, device=device, dtype=torch.int32)


def unpack_gptq_qweight(qweight: torch.Tensor) -> torch.Tensor:
    """``[K // 8, N]`` int32 packed along K -> ``[K, N]`` codes in ``[0, 15]``."""
    n = qweight.shape[1]
    shifts = _nibble_shifts(qweight.device)  # [8]
    # [K // 8, 8, N]; sub-index i selects k = row * 8 + i
    codes = (qweight.unsqueeze(1) >> shifts.view(1, 8, 1)) & 0xF
    return codes.reshape(-1, n)  # [K, N]


def unpack_gptq_qzeros(qzeros: torch.Tensor) -> torch.Tensor:
    """``[K // gs, N // 8]`` int32 packed along N -> ``[K // gs, N]`` codes."""
    rows = qzeros.shape[0]
    shifts = _nibble_shifts(qzeros.device)  # [8]
    # [K // gs, N // 8, 8]; sub-index j selects n = col * 8 + j
    codes = (qzeros.unsqueeze(-1) >> shifts.view(1, 1, 8)) & 0xF
    return codes.reshape(rows, -1)  # [K // gs, N]


def build_qscale_and_zeros(scales: torch.Tensor, zp: torch.Tensor) -> torch.Tensor:
    """Build the ``[K // gs, N, 2]`` qScaleAndZeros tensor.

    ``scales`` and ``zp`` are both ``[K // gs, N]`` (zp in code space ``[0, 15]``).
    ``zero = scale * (8 - zp)`` folds the integer zero-point into a float bias.
    """
    scales = scales.contiguous()
    zeros = scales * (8.0 - zp.to(scales.dtype))
    return torch.stack([scales, zeros], dim=-1).contiguous()


def xpu_int4pack_mm(
    x: torch.Tensor,
    qweight_packed: torch.Tensor,
    group_size: int,
    qscale_and_zeros: torch.Tensor,
    out_features: int,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run ``_weight_int4pack_mm`` with leading-dim flatten / restore + bias."""
    out_shape = x.shape[:-1] + (out_features,)
    reshaped_x = x.reshape(-1, x.shape[-1]).contiguous()
    out = torch.ops.aten._weight_int4pack_mm(
        reshaped_x, qweight_packed, group_size, qscale_and_zeros
    )
    if bias is not None:
        out = out + bias
    return out.reshape(out_shape)
