"""Triton kernel for writing MXFP8 scale factors in interleaved layout.

When page_size=128 and sf_vec_size=32, FA4 expects scale factors in the
BlockScaledBasicChunk atom layout: [num_pages, nheads, 32, 4, 4].

The interleave mapping for a token at page offset `t` (0-127), head `h`,
scale index `s` (0-3) is:

    output[page, h, t % 32, t // 32, s]

Linear offset: (page_offset % 32) * 16 + (page_offset // 32) * 4 + scale_idx

The 4 scales per (token, head) are contiguous in both input and output,
so we vectorize as u32 loads/stores (4 bytes at a time).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _store_sf_interleaved_kernel(
    sf_in_ptr,  # [num_tokens * nheads] u32 (4 scales packed per u32)
    sf_out_ptr,  # [num_pages * nheads * 128] u32 (interleaved, 4 scales per u32)
    loc_ptr,  # [num_tokens] i64
    num_tokens,  # type: ignore
    nheads: tl.constexpr,
    page_size: tl.constexpr,  # 128
    BLOCK_T: tl.constexpr,
):
    """Scatter-write per-token scale factors into interleaved layout.

    Input is viewed as [num_tokens, nheads] of u32 (4 packed e8m0 scales).
    Output is [num_pages, nheads, 128] of u32 where the 128 positions
    follow the interleave pattern: index = (page_offset % 32) * 4 + (page_offset // 32).
    """
    pid = tl.program_id(0)
    tok_start = pid * BLOCK_T
    tok_offsets = tok_start + tl.arange(0, BLOCK_T)
    mask = tok_offsets < num_tokens

    # Load slot indices
    slots = tl.load(loc_ptr + tok_offsets, mask=mask, other=0)
    page_offsets = slots % page_size
    page_idxs = slots // page_size

    # Interleave: (page_offset % 32) * 4 + (page_offset // 32)
    # This is the u32 offset within a (nheads, 128) page block
    interleaved_pos = (page_offsets % 32) * 4 + (page_offsets // 32)

    # Per-page stride in u32: nheads * 128 (= nheads * 32 * 4)
    page_stride: tl.constexpr = nheads * 128

    for h in tl.static_range(nheads):
        # Load 4 packed scales as u32: sf_in[tok, h]
        in_offsets = tok_offsets * nheads + h
        vals = tl.load(sf_in_ptr + in_offsets, mask=mask, other=0)

        # Store to interleaved position: sf_out[page, h * 128 + interleaved_pos]
        out_offsets = page_idxs * page_stride + h * 128 + interleaved_pos
        tl.store(sf_out_ptr + out_offsets, vals, mask=mask)


def store_sf_interleaved(
    sf_in: torch.Tensor,  # [num_tokens, nheads, sf_dim] e8m0
    sf_out: torch.Tensor,  # [num_pages, nheads, 32, 4, 4] e8m0
    loc: torch.Tensor,  # [num_tokens] int64
    page_size: int = 128,
):
    """Scatter-write per-token scale factors into interleaved page layout."""
    assert (
        page_size == 128
    ), f"Interleaved SF layout requires page_size=128, got {page_size}"
    num_tokens, nheads, sf_dim = sf_in.shape
    assert sf_dim == 4, f"Expected sf_dim=4 (hdim=128, sf_vec_size=32), got {sf_dim}"

    # View as u32: 4 contiguous e8m0 bytes → 1 u32
    sf_in_u8 = sf_in.view(torch.uint8) if sf_in.dtype == torch.float8_e8m0fnu else sf_in
    sf_out_u8 = (
        sf_out.view(torch.uint8) if sf_out.dtype == torch.float8_e8m0fnu else sf_out
    )

    sf_in_u32 = (
        sf_in_u8.reshape(num_tokens, nheads, 4)
        .contiguous()
        .view(torch.int32)
        .reshape(num_tokens, nheads)
    )
    sf_out_u32 = sf_out_u8.reshape(-1, 4).view(torch.int32).reshape(-1)

    BLOCK_T = 128
    grid = ((num_tokens + BLOCK_T - 1) // BLOCK_T,)

    _store_sf_interleaved_kernel[grid](
        sf_in_u32,
        sf_out_u32,
        loc,
        num_tokens,
        nheads=nheads,
        page_size=page_size,
        BLOCK_T=BLOCK_T,
    )
