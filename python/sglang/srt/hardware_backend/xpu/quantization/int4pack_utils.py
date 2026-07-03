# SPDX-License-Identifier: Apache-2.0
"""Helpers for lowering GPTQ/AWQ int4 weights to the torch XPU int4pack layout.

The native op is ``torch.ops.aten._weight_int4pack_mm(A, B, group_size, qsz)``:

* ``A``    : ``[M, K]`` fp16/bf16 activation (2D, contiguous).
* ``B``    : ``[N, K // 2]`` uint8 packed weight. Within each byte the low
             nibble is the *even* ``k`` and the high nibble is the *odd* ``k``;
             nibble values are the raw quant codes ``q in [0, 15]``.
* ``qsz``  : ``[K // group_size, N, 2]`` tensor (activation dtype) whose last
             dim is ``[scale, zero]``.

The kernel dequantizes as ``W = (q - 8) * scale + zero``. Both GPTQ and AWQ use
``W = (q - zp) * scale``, so we fold the integer zero-point into a float
``zero = scale * (8 - zp)``.

These conventions were verified on torch 2.12.0+xpu; see
``kernel-plan/scratch/verify_int4pack.py``.
"""

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


def build_qscale_and_zeros(
    scales: torch.Tensor, zp: torch.Tensor
) -> torch.Tensor:
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
    reshaped_x = x.reshape(-1, x.shape[-1])
    out = torch.ops.aten._weight_int4pack_mm(
        reshaped_x, qweight_packed, group_size, qscale_and_zeros
    )
    if bias is not None:
        out = out + bias
    return out.reshape(out_shape)
