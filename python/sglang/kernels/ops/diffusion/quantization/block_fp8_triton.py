# SPDX-License-Identifier: Apache-2.0
"""Block-FP8 producers: e4m3 payload plus one UE8M0 scale per 128-wide group
along K, the pre-quantized input DeepGEMM takes on SM100 and SM120.

Each producer is bit-exact against the unfused bf16 kernel followed by
``sglang_per_token_group_quant_fp8`` with ``scale_ue8m0``: an exact bf16
absmax floored at 1e-10, ``amax * (1 / 448)`` rounded up to a power of two by
its bits -- the biased exponent, plus one unless the mantissa is zero -- one
fp32 multiply by the exact inverse power of two, a clamp at +448 and a
saturating RNE cast. The exponent bytes go four groups to an int32, the first
group in the low byte, bytes past the last group zero, ``[rows, ceil(groups /
4)]`` row-major rather than DeepGEMM's MN-major, so gathering rows gathers
scales.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.diffusion.common.numerics import round_bf16_to_fp32

# The constants sglang's per-token-group FP8 quantizer bakes in; these kernels
# have to agree with it bit for bit, so they are its, not ours.
_FP8_MAX = tl.constexpr(448.0)
_FP8_MAX_INV = tl.constexpr(1.0 / 448.0)
_QUANT_EPS = tl.constexpr(1e-10)


@triton.jit
def _indexed_scale_shift_block_fp8_kernel(
    q_ptr,
    q_scale_ptr,
    x_ptr,
    shift_ptr,
    scale_ptr,
    indices_ptr,
    num_groups,
    stride_x_row,
    stride_shift_row,
    stride_scale_row,
    stride_indices,
    stride_q_row,
    stride_q_scale_row,
    GROUP_SIZE: tl.constexpr,
    BLOCK_GROUPS: tl.constexpr,
):
    row = tl.program_id(0)
    groups = tl.arange(0, BLOCK_GROUPS)
    lanes = tl.arange(0, GROUP_SIZE)
    columns = groups[:, None] * GROUP_SIZE + lanes[None, :]
    mask = (groups < num_groups)[:, None]
    index = tl.load(indices_ptr + row * stride_indices)

    x = tl.load(x_ptr + row * stride_x_row + columns, mask=mask, other=0.0).to(
        tl.float32
    )
    shift = tl.load(
        shift_ptr + index * stride_shift_row + columns, mask=mask, other=0.0
    ).to(tl.float32)
    scale = tl.load(
        scale_ptr + index * stride_scale_row + columns, mask=mask, other=0.0
    ).to(tl.float32)
    # indexed_scale_shift_bf16_'s arithmetic, cast where it stores bf16.
    one_plus_scale = round_bf16_to_fp32(1.0 + scale)
    scaled = round_bf16_to_fp32(x * one_plus_scale)
    modulated = (scaled + shift).to(tl.bfloat16).to(tl.float32)

    amax = tl.maximum(tl.max(tl.abs(modulated), axis=1), _QUANT_EPS)
    bits = (amax * _FP8_MAX_INV).to(tl.int32, bitcast=True)
    exponent = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    inverse = ((254 - exponent) << 23).to(tl.float32, bitcast=True)
    q = tl.minimum(modulated * inverse[:, None], _FP8_MAX)
    exponent = tl.reshape(
        tl.where(groups < num_groups, exponent, 0), [BLOCK_GROUPS // 4, 4]
    )
    packs = tl.arange(0, BLOCK_GROUPS // 4)
    tl.store(
        q_scale_ptr + row * stride_q_scale_row + packs,
        tl.sum(exponent << (tl.arange(0, 4) * 8)[None, :], axis=1),
        mask=packs < tl.cdiv(num_groups, 4),
    )
    tl.store(
        q_ptr + row * stride_q_row + columns,
        q.to(q_ptr.dtype.element_ty),
        mask=mask,
    )


def can_use_indexed_scale_shift_block_fp8(
    x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor, *, group_size: int
) -> bool:
    """Whether the unfused path would take the fused modulation kernel.

    ``indexed_scale_shift_bf16_`` is only used for contiguous CUDA bf16 rows
    and bf16 parameters; anything else falls back to torch arithmetic, which
    this kernel does not reproduce.
    """
    return (
        x.is_cuda
        and x.dim() == 2
        and x.dtype == shift.dtype == scale.dtype == torch.bfloat16
        and x.is_contiguous()
        and shift.stride(-1) == 1
        and scale.stride(-1) == 1
        and x.shape[1] % group_size == 0
    )


def indexed_scale_shift_block_fp8(
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
    *,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``indexed_scale_shift_bf16_`` then ``sglang_per_token_group_quant_fp8``.

    Returns the same fp8 rows those two calls produce, bit for bit, without
    writing the modulated rows back, and the UE8M0 exponents that quantizer
    packs for DeepGEMM, four groups to an int32, ``[rows, ceil(groups / 4)]``
    but row-major. ``x`` is left untouched.
    """
    if not can_use_indexed_scale_shift_block_fp8(
        x, shift, scale, group_size=group_size
    ):
        raise ValueError(
            "indexed_scale_shift_block_fp8 needs contiguous CUDA bf16 rows whose "
            f"width divides by {group_size}, and bf16 shift/scale with a unit "
            "last stride"
        )
    rows, hidden_size = x.shape
    num_groups = hidden_size // group_size
    q = torch.empty((rows, hidden_size), device=x.device, dtype=torch.float8_e4m3fn)
    q_scale = torch.empty(
        (rows, -(-num_groups // 4)), device=x.device, dtype=torch.int32
    )
    if rows == 0:
        return q, q_scale
    _indexed_scale_shift_block_fp8_kernel[(rows,)](
        q,
        q_scale,
        x,
        shift,
        scale,
        indices,
        num_groups,
        x.stride(0),
        shift.stride(0),
        scale.stride(0),
        indices.stride(0),
        q.stride(0),
        q_scale.stride(0),
        GROUP_SIZE=group_size,
        BLOCK_GROUPS=max(triton.next_power_of_2(num_groups), 4),
        num_warps=8,
    )
    return q, q_scale
