"""NVFP4 KV rows for the Triton sparse-attention kernels (MSA-v1 cache layout).

A row is one head of one token: ``D // 2`` payload bytes (two e2m1 codes per
byte, the even element in the low nibble) and ``D // 16`` E4M3 block scales,
with a unit outer scale -- the layout `MiniMaxM3Nvfp4KVPool` stores.
"""

import msgspec
import torch
import triton
import triton.language as tl


class Nvfp4Cache(msgspec.Struct, frozen=True):
    """One layer of a packed NVFP4 cache, as `MiniMaxM3Nvfp4KVPool` stores it.

    ``payload`` is uint8 ``[pages, (heads,) page_size, D // 2]`` and ``scales``
    uint8 ``[pages, (heads,) page_size, D // 16]`` (E4M3 bits); the index-K
    cache has no head dimension.
    """

    payload: torch.Tensor
    scales: torch.Tensor

    @property
    def page_size(self) -> int:
        return self.payload.shape[-2]

    @property
    def device(self) -> torch.device:
        return self.payload.device

    @property
    def num_heads(self) -> int:
        return self.payload.shape[1] if self.payload.dim() == 4 else 1

    def payload_strides(self) -> tuple[int, int, int]:
        """(page, head, row) strides in bytes."""
        return _page_head_row_strides(self.payload)

    def scale_strides(self) -> tuple[int, int, int]:
        return _page_head_row_strides(self.scales)


def _page_head_row_strides(buf: torch.Tensor) -> tuple[int, int, int]:
    if buf.dim() == 4:
        return buf.stride(0), buf.stride(1), buf.stride(2)
    return buf.stride(0), 0, buf.stride(1)


@triton.jit
def nvfp4_f32_to_e2m1(x):
    """Round fp32 to e2m1 codes (round half to even, saturating at 6)."""
    a = tl.abs(x)
    mag = (
        (a > 0.25).to(tl.int32)
        + (a >= 0.75).to(tl.int32)
        + (a > 1.25).to(tl.int32)
        + (a >= 1.75).to(tl.int32)
        + (a > 2.5).to(tl.int32)
        + (a >= 3.5).to(tl.int32)
        + (a > 5.0).to(tl.int32)
    )
    return (mag | tl.where(x < 0, 8, 0)).to(tl.uint8)


@triton.jit
def nvfp4_slot_offsets(
    slots,
    stride_page,
    stride_head,
    stride_row,
    head,
    PAGE_SIZE: tl.constexpr,
):
    """Byte offset of each slot's row (payload or scales) in a paged cache."""
    return (
        (slots // PAGE_SIZE) * stride_page
        + head * stride_head
        + (slots % PAGE_SIZE) * stride_row
    )


@triton.jit
def _fp4x2_to_f16(byte_i32):
    """gfx950: one packed e2m1 byte -> (low, high) fp16 in one conversion.

    The instruction's f32 scale only contributes its exponent (E8M0), so the
    E4M3 block scale is applied by the caller; 1.0 leaves the codes unscaled.
    """
    return tl.inline_asm_elementwise(
        "v_cvt_scalef32_pk_f16_fp4 $0, $2, 1.0\n\tv_lshrrev_b32 $1, 16, $0",
        "=&v,=&v,v",
        [byte_i32],
        dtype=(tl.float16, tl.float16),
        is_pure=True,
        pack=1,
    )


@triton.jit
def nvfp4_load_even_odd(
    payload_ptr,
    scale_ptr,
    payload_row_off,
    scale_row_off,
    row_mask,
    D: tl.constexpr,
):
    """Load rows as their even and odd columns, two fp16 ``[N, D//2]`` tiles.

    Byte ``j`` of a row holds elements ``2j`` (low nibble) and ``2j+1``, so the
    split needs no interleave; a dot over D is the sum of the even and odd dots.
    e2m1 times an E4M3 scale has at most 6 significant bits and stays in the fp16
    normal range, so the dequantized values are exact. gfx950 only.
    """
    HALF: tl.constexpr = D // 2
    cols = tl.arange(0, HALF)
    packed = tl.load(
        payload_ptr + payload_row_off[:, None] + cols[None, :],
        mask=row_mask[:, None],
        other=0,
    )
    scale_bytes = tl.load(
        scale_ptr + scale_row_off[:, None] + (cols // 8)[None, :],
        mask=row_mask[:, None],
        other=0,
    )
    even, odd = _fp4x2_to_f16(packed.to(tl.int32))
    scale = scale_bytes.to(tl.float8e4nv, bitcast=True).to(tl.float16)
    return even * scale, odd * scale


@triton.jit
def nvfp4_load_rows(
    payload_ptr,
    scale_ptr,
    payload_row_off,
    scale_row_off,
    row_mask,
    D: tl.constexpr,
):
    """Load rows as one fp16 ``[N, D]`` tile in element order, with the values of
    `nvfp4_load_even_odd` and each block scale read once. gfx950 only."""
    N: tl.constexpr = payload_row_off.shape[0]
    HALF: tl.constexpr = D // 2
    NBLK: tl.constexpr = D // 16
    packed = tl.load(
        payload_ptr + payload_row_off[:, None] + tl.arange(0, HALF)[None, :],
        mask=row_mask[:, None],
        other=0,
    )
    scale_bytes = tl.load(
        scale_ptr + scale_row_off[:, None] + tl.arange(0, NBLK)[None, :],
        mask=row_mask[:, None],
        other=0,
    )
    scale = scale_bytes.to(tl.float8e4nv, bitcast=True).to(tl.float16)
    scale = tl.reshape(tl.broadcast_to(scale[:, :, None], [N, NBLK, 8]), [N, HALF])
    even, odd = _fp4x2_to_f16(packed.to(tl.int32))
    return tl.reshape(tl.join(even * scale, odd * scale), [N, D])
