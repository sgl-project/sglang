# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""UltraQuant 4-bit KV cache format: FP4 E2M1 codes + UE8M0 group scales.

For each group of ``GROUP_SIZE`` elements along the head dimension::

    s    = 2 ** round(log2(CONSTANT_C * absmax(group)))   # UE8M0, one byte
    code = nearest_fp4_e2m1(group / s)                     # one nibble

Dequantization is ``code * s``; ``CONSTANT_C`` is folded into the stored
exponent and never appears on the read side.

Keys are Hadamard-rotated before quantization, values are not. The rotation is
orthonormal, so rotating queries and keys alike leaves attention scores
unchanged while spreading key outliers across the head dimension, which is what
makes 4 bits viable for keys.

Storage is structure-of-arrays: codes and scales occupy separate buffers, which
keeps a code row exactly ``head_dim // 2`` bytes and naturally aligned. The
functions here are the numerical reference; the serving path uses the Triton
kernels in ``sglang.kernels.ops.kvcache.ultraquant``, which are tested against
this module.
"""

import torch

from sglang.srt.layers.quantization.kvfp4_tensor import E2M1_VALUES

GROUP_SIZE = 32
"""Elements sharing one UE8M0 scale; matches the scaled MFMA's scale granularity."""

CONSTANT_C = 0.156
"""MSE-optimal scale constant: ``s_raw = CONSTANT_C * absmax``."""

UE8M0_BIAS = 127
"""E8M0 byte ``e`` encodes ``2 ** (e - UE8M0_BIAS)``; ``e == 0`` means zero."""

UE8M0_MIN_EXP = -126
UE8M0_MAX_EXP = 127

# The 15 distinct E2M1 levels in ascending order; quantization buckets against
# the midpoints between consecutive levels.
E2M1_LEVELS_SORTED = tuple(sorted(set(E2M1_VALUES)))

E2M1_MIDPOINTS = tuple(
    (E2M1_LEVELS_SORTED[i] + E2M1_LEVELS_SORTED[i + 1]) / 2.0
    for i in range(len(E2M1_LEVELS_SORTED) - 1)
)

E2M1_ZERO_LEVEL_INDEX = E2M1_LEVELS_SORTED.index(0.0)
"""Index of the exact-zero level, which is also the sign split point."""


def n_groups(head_dim: int) -> int:
    """Number of UE8M0 scales stored per (token, head)."""
    if head_dim % GROUP_SIZE != 0:
        raise ValueError(
            f"head_dim={head_dim} must be a multiple of GROUP_SIZE={GROUP_SIZE}"
        )
    return head_dim // GROUP_SIZE


def code_bytes(head_dim: int) -> int:
    """Bytes of packed FP4 codes stored per (token, head)."""
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even, got {head_dim}")
    return head_dim // 2


def sorted_index_to_e2m1_bits(sorted_index: torch.Tensor) -> torch.Tensor:
    """Map an ascending-level index in ``[0, 14]`` to its E2M1 bit pattern.

    Closed form rather than a lookup table: ascending levels run from the most
    negative (``0b1111``) down to ``0b1001`` for the negative half, then
    ``0b0000`` upward for the non-negative half.
    """
    return torch.where(
        sorted_index < E2M1_ZERO_LEVEL_INDEX,
        (len(E2M1_VALUES) - 1) - sorted_index,
        sorted_index - E2M1_ZERO_LEVEL_INDEX,
    )


def hadamard_matrix(
    dim: int, device: torch.device, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Orthonormal Sylvester Hadamard matrix, so ``H @ H.T == I``."""
    if dim <= 0 or (dim & (dim - 1)) != 0:
        raise ValueError(f"Hadamard rotation requires power-of-two dim, got {dim}")
    h = torch.ones((1, 1), dtype=torch.float64)
    while h.shape[0] < dim:
        h = torch.cat(
            [torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)],
            dim=0,
        )
    return (h / dim**0.5).to(device=device, dtype=dtype).contiguous()


def ue8m0_encode(scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Snap positive scales to a power of two and encode as UE8M0 bytes.

    Returns ``(snapped_scale_fp32, byte_uint8)``. Non-positive and non-finite
    inputs encode as byte 0, the zero sentinel, and snap to 0.0. Rounding is
    half-up via ``floor(log2(s) + 0.5)`` to match the Triton kernels.
    """
    is_zero = (scale <= 0) | ~torch.isfinite(scale)
    safe = torch.where(is_zero, torch.ones_like(scale), scale)
    exponent = torch.floor(torch.log2(safe) + 0.5).to(torch.int32)
    exponent = torch.clamp(exponent, min=UE8M0_MIN_EXP, max=UE8M0_MAX_EXP)

    byte = torch.where(
        is_zero,
        torch.zeros_like(exponent),
        exponent + UE8M0_BIAS,
    ).to(torch.uint8)
    snapped = torch.where(
        is_zero,
        torch.zeros_like(scale, dtype=torch.float32),
        torch.exp2(exponent.to(torch.float32)),
    )
    return snapped, byte


def ue8m0_decode(byte: torch.Tensor) -> torch.Tensor:
    """Decode UE8M0 bytes to fp32; byte 0 decodes to exactly 0.0."""
    exponent = byte.to(torch.int32) - UE8M0_BIAS
    return torch.where(
        byte == 0,
        torch.zeros((), dtype=torch.float32, device=byte.device),
        torch.exp2(exponent.to(torch.float32)),
    )


def pack_nibbles(codes: torch.Tensor) -> torch.Tensor:
    """Pack 4-bit codes along the last axis, two per byte, low nibble first."""
    if codes.shape[-1] % 2 != 0:
        raise ValueError(f"cannot pack an odd last dim: {codes.shape[-1]}")
    low = codes[..., 0::2] & 0xF
    high = codes[..., 1::2] & 0xF
    return (low | (high << 4)).to(torch.uint8)


def unpack_nibbles(packed: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`pack_nibbles`, widening the last axis by two."""
    wide = packed.to(torch.int32)
    out = torch.empty(
        packed.shape[:-1] + (2 * packed.shape[-1],),
        dtype=torch.uint8,
        device=packed.device,
    )
    out[..., 0::2] = (wide & 0xF).to(torch.uint8)
    out[..., 1::2] = ((wide >> 4) & 0xF).to(torch.uint8)
    return out


class UltraQuantKVQuantizeUtil:
    """Reference quantize/dequantize for the UltraQuant KV format.

    Operates on the trailing head dimension, so it accepts any leading shape.
    ``batched_quantize`` is the correctness oracle for the Triton store kernel.
    """

    @staticmethod
    def batched_quantize(
        tensor: torch.Tensor, *, rotate: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize ``[..., head_dim]`` to packed codes and UE8M0 scale bytes.

        ``rotate`` applies the Hadamard rotation and must be true for keys and
        false for values. Returns ``([..., head_dim // 2], [..., n_groups])``,
        both uint8.
        """
        head_dim = tensor.shape[-1]
        groups = n_groups(head_dim)

        values = tensor.to(torch.float32)
        if rotate:
            rotation = hadamard_matrix(head_dim, tensor.device)
            values = values @ rotation.T

        grouped = values.reshape(*values.shape[:-1], groups, GROUP_SIZE)
        absmax = grouped.abs().amax(dim=-1)
        snapped, scale_bytes = ue8m0_encode(absmax * CONSTANT_C)

        # Zero groups carry scale byte 0; divide by one to keep the normalized
        # values finite, then force their codes to the exact-zero level.
        is_zero = (snapped == 0).unsqueeze(-1)
        normalized = grouped / torch.where(is_zero, 1.0, snapped.unsqueeze(-1))

        midpoints = tensor.new_tensor(E2M1_MIDPOINTS, dtype=torch.float32)
        sorted_index = torch.bucketize(normalized, midpoints)
        sorted_index = torch.where(is_zero, E2M1_ZERO_LEVEL_INDEX, sorted_index)

        bits = sorted_index_to_e2m1_bits(sorted_index).to(torch.uint8)
        codes = pack_nibbles(bits.reshape(*values.shape[:-1], head_dim))
        return codes, scale_bytes

    @staticmethod
    def batched_dequantize(
        codes: torch.Tensor,
        scale_bytes: torch.Tensor,
        *,
        dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        """Dequantize packed codes and UE8M0 bytes back to ``dtype``.

        Keys come back in the rotated basis, because that is how they are
        stored; no inverse rotation is applied.
        """
        head_dim = 2 * codes.shape[-1]
        groups = n_groups(head_dim)

        lookup = codes.new_tensor(E2M1_VALUES, dtype=torch.float32)
        levels = lookup[unpack_nibbles(codes).to(torch.long)]

        grouped = levels.reshape(*levels.shape[:-1], groups, GROUP_SIZE)
        scales = ue8m0_decode(scale_bytes).unsqueeze(-1)
        return (grouped * scales).reshape(*levels.shape[:-1], head_dim).to(dtype)
