"""CPU/PyTorch helpers for a Q4_K-to-W4A16 conversion.

This module deliberately contains no XPU kernel import or loader integration.
It defines the compact logical input expected by a future unsigned-INT4 W4A16
consumer while keeping the conversion easy to validate on CPU.

GGUF 0.19 Q4_K stores 256 values in a 144-byte super-block.  Its eight
32-value groups have independent six-bit scale and minimum multipliers.  The
result below has an adjacent-element nibble layout and one scale/zero pair per
32 values, so its dequantization is ``(q - zero) * scale``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

Q4_K_SUPER_BLOCK_SIZE = 256
Q4_K_BLOCK_BYTES = 144
W4A16_GROUP_SIZE = 32


@dataclass(frozen=True)
class Q4KW4A16:
    """Unsigned INT4 W4A16 representation derived from GGUF Q4_K.

    ``qweight`` packs adjacent logical elements: element ``2*i`` is the low
    nibble and element ``2*i + 1`` is the high nibble of byte ``i``.  ``scales``
    and ``zeros`` are FP32 intentionally: Q4_K's scale and minimum use two
    independently rounded FP16 bases, and retaining their quotient in FP32
    makes the affine W4A16 form agree with ``gguf.dequantize`` without adding a
    lossy metadata-cast policy to this standalone helper.
    """

    qweight: torch.Tensor
    scales: torch.Tensor
    zeros: torch.Tensor


def _validate_q4_k_bytes(qweight: torch.Tensor) -> None:
    if qweight.dtype != torch.uint8:
        raise TypeError(f"Q4_K bytes must be uint8, got {qweight.dtype}")
    if qweight.ndim != 2:
        raise ValueError(f"Q4_K bytes must be rank 2, got {qweight.ndim}")
    if qweight.shape[1] == 0 or qweight.shape[1] % Q4_K_BLOCK_BYTES:
        raise ValueError(
            "Q4_K byte width must be a non-zero multiple of "
            f"{Q4_K_BLOCK_BYTES}, got {qweight.shape[1]}"
        )


def _unpack_scale_and_min(
    scale_bytes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode the eight 6-bit Q4_K scale/min multipliers per super-block."""
    meta = scale_bytes.to(torch.int32)
    # gguf.quants.Q4_K.get_scale_min() labels bytes 0..3 as A..D, bytes
    # 4..7 as a..d, and bytes 8..11 as the high-bit fields.  Keeping the
    # construction vectorized also makes the byte layout explicit.
    scale6 = torch.cat(
        (
            meta[..., :4] & 0x3F,
            (meta[..., 8:12] & 0x0F) | ((meta[..., :4] >> 2) & 0x30),
        ),
        dim=-1,
    )
    min6 = torch.cat(
        (
            meta[..., 4:8] & 0x3F,
            (meta[..., 8:12] >> 4) | ((meta[..., 4:8] >> 2) & 0x30),
        ),
        dim=-1,
    )
    return scale6, min6


def repack_q4_k_to_w4a16(qweight: torch.Tensor) -> Q4KW4A16:
    """Convert raw GGUF 0.19 Q4_K bytes to unsigned INT4 W4A16 tensors.

    The input is a ``[N, blocks * 144]`` byte matrix.  The result contains
    ``qweight[N, K/2]`` and ``scales/zeros[N, K/32]`` where ``K=blocks*256``.
    A Q4_K group with zero scale is canonicalized to codes=0, scale=1 and
    zero=minimum. This represents its constant ``-minimum`` value exactly.
    """
    _validate_q4_k_bytes(qweight)
    rows = qweight.shape[0]
    super_blocks = qweight.shape[1] // Q4_K_BLOCK_BYTES
    block = qweight.reshape(rows, super_blocks, Q4_K_BLOCK_BYTES)

    bases = (
        block[..., :4].contiguous().view(torch.float16).reshape(rows, super_blocks, 2)
    )
    scale_base = bases[..., 0].to(torch.float32)
    min_base = bases[..., 1].to(torch.float32)
    scale6, min6 = _unpack_scale_and_min(block[..., 4:16])
    scales = scale_base.unsqueeze(-1) * scale6.to(torch.float32)
    minimums = min_base.unsqueeze(-1) * min6.to(torch.float32)

    # Q4_K has four 32-byte regions.  In each, its low nibbles come first in
    # element order, then its high nibbles; this is the layout used by gguf
    # 0.19's Q4_K.dequantize_blocks().
    qs = block[..., 16:].reshape(rows, super_blocks, 4, 32)
    scale_groups = scales.reshape(rows, -1).contiguous()
    minimum_groups = minimums.reshape(rows, -1)
    zero_scale = scale_groups == 0
    values = torch.cat((qs & 0x0F, qs >> 4), dim=-1).reshape(rows, -1, W4A16_GROUP_SIZE)
    values = torch.where(zero_scale.unsqueeze(-1), 0, values).reshape(rows, -1)
    packed = (values[:, 0::2] | (values[:, 1::2] << 4)).contiguous()

    zeros = torch.where(
        zero_scale, minimum_groups, minimum_groups / scale_groups
    ).contiguous()
    scale_groups = torch.where(
        zero_scale, torch.ones_like(scale_groups), scale_groups
    ).contiguous()
    return Q4KW4A16(packed, scale_groups, zeros)


def dequantize_w4a16(
    repacked: Q4KW4A16, out_dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Dequantize ``Q4KW4A16`` using its defining ``(q-zero)*scale`` rule."""
    qweight, scales, zeros = repacked.qweight, repacked.scales, repacked.zeros
    if qweight.dtype != torch.uint8 or qweight.ndim != 2:
        raise ValueError("W4A16 qweight must be a rank-2 uint8 tensor")
    rows, packed_width = qweight.shape
    logical_width = packed_width * 2
    expected_groups = logical_width // W4A16_GROUP_SIZE
    if logical_width % W4A16_GROUP_SIZE or scales.shape != (rows, expected_groups):
        raise ValueError("W4A16 scale shape does not match packed qweight")
    if zeros.shape != scales.shape:
        raise ValueError("W4A16 zero shape does not match scales")

    q = torch.stack((qweight & 0x0F, qweight >> 4), dim=-1).reshape(rows, -1)
    return (
        (
            (
                q.to(out_dtype).reshape(rows, expected_groups, W4A16_GROUP_SIZE)
                - zeros.to(out_dtype).unsqueeze(-1)
            )
            * scales.to(out_dtype).unsqueeze(-1)
        )
        .reshape(rows, logical_width)
        .contiguous()
    )


__all__ = [
    "Q4KW4A16",
    "Q4_K_BLOCK_BYTES",
    "Q4_K_SUPER_BLOCK_SIZE",
    "W4A16_GROUP_SIZE",
    "dequantize_w4a16",
    "repack_q4_k_to_w4a16",
]
