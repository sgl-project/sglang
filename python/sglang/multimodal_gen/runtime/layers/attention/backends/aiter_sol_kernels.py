# SPDX-License-Identifier: Apache-2.0
#
# Fused Sol-Attn routing for AITER's mha_v4 mode-2 rows (ROCm, gfx950).
#
# aiter builds the block selection from traceable tensor operations so that a
# caller can trace it. Run eagerly, that is about thirty small kernels per
# attention call, and it makes a (num_work_items, num_kv_blocks) 64-bit image to
# pack the bitmap from. This module does the same work after the proxy product
# in one Triton kernel, which writes the four forms the kernel reads: the
# selection mask, the per-row exact-block count, the packed bitmap and the
# ragged index list. Giving each row its own segment of the index list keeps
# the starts a function of the shape, so no running sum over the counts and no
# second pass over the mask are needed.

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _sol_route_pack_kernel(
    proxy_ptr,
    mask_ptr,
    lut_count_ptr,
    bitmap_ptr,
    kv_indices_ptr,
    stride_proxy_b,
    stride_proxy_h,
    stride_proxy_i,
    stride_proxy_j,
    stride_mask_b,
    stride_mask_h,
    stride_mask_i,
    stride_mask_j,
    beta,
    num_kv_blocks,
    num_q_tiles,
    num_heads,
    bitmap_ds,
    BETA_IS_TENSOR: tl.constexpr,
    PARTIAL_TAIL: tl.constexpr,
    NUM_WORDS: tl.constexpr,
):
    """Threshold one proxy row and write its mask, count, bitmap and indices.

    One program holds one (batch, head, q_tile) row. The row is read as
    (NUM_WORDS, 32) so that the bitmap words are a reduction along the second
    axis and need no 64-bit image.

    The index list is written in the row's own segment of kv_indices_ptr, which
    holds num_kv_blocks entries per row. A row therefore starts at a position
    this kernel already knows, and the selected blocks compact into it by a
    running sum along the row.

    tau = mean_j(proxy) + beta * population_std_j(proxy), and a block is exact
    where its proxy is above tau. A partial tail block is always exact, because
    the pooled branch is defined for whole blocks only. A row that selects
    nothing keeps its highest-proxy block, which gives the softmax a real
    maximum to work from.
    """
    idx = tl.program_id(0)
    heads_tiles = num_heads * num_q_tiles
    b = idx // heads_tiles
    rem = idx % heads_tiles
    h = rem // num_q_tiles
    i = rem % num_q_tiles

    word = tl.arange(0, NUM_WORDS)[:, None]
    bit = tl.arange(0, 32)[None, :]
    j = word * 32 + bit
    in_row = j < num_kv_blocks

    row = proxy_ptr + b * stride_proxy_b + h * stride_proxy_h + i * stride_proxy_i
    proxy = tl.load(row + j * stride_proxy_j, mask=in_row, other=0.0).to(tl.float32)

    count_f = num_kv_blocks.to(tl.float32)
    mean = tl.sum(tl.where(in_row, proxy, 0.0)) / count_f
    centered = tl.where(in_row, proxy - mean, 0.0)
    variance = tl.sum(centered * centered) / count_f
    if BETA_IS_TENSOR:
        beta_value = tl.load(beta).to(tl.float32)
    else:
        beta_value = beta
    tau = mean + beta_value * tl.sqrt(variance)

    selected = (proxy > tau) & in_row
    if PARTIAL_TAIL:
        selected = selected | (j == num_kv_blocks - 1)

    # A row that cleared every block keeps its highest-proxy one.
    ranked = tl.where(in_row, proxy, float("-inf"))
    best = tl.argmax(tl.reshape(ranked, (NUM_WORDS * 32,)), axis=0)
    empty = tl.sum(selected.to(tl.int32)) == 0
    selected = selected | (empty & (j == best) & in_row)

    out = mask_ptr + b * stride_mask_b + h * stride_mask_h + i * stride_mask_i
    tl.store(out + j * stride_mask_j, selected.to(tl.int8), mask=in_row)
    tl.store(lut_count_ptr + idx, tl.sum(selected.to(tl.int32)))

    # Bits past the row are set: a set bit means "already exact", which is what
    # masks the overhang out of the pooled pass.
    bits = tl.where(in_row, selected, True).to(tl.int32)
    words = tl.sum(bits << bit, axis=1)
    word_idx = tl.arange(0, NUM_WORDS)
    tl.store(bitmap_ptr + idx * bitmap_ds + word_idx, words, mask=word_idx < bitmap_ds)

    # The blocks keep their order, so an exclusive running sum of the selection
    # gives each one its position in the row's segment.
    taken = tl.reshape(selected.to(tl.int32), (NUM_WORDS * 32,))
    position = tl.cumsum(taken, axis=0) - taken
    block = tl.arange(0, NUM_WORDS * 32)
    tl.store(kv_indices_ptr + idx * num_kv_blocks + position, block, mask=taken == 1)


_LUT_STARTS: dict[tuple, torch.Tensor] = {}


def row_lut_start(
    num_work_items: int, num_kv_blocks: int, device: torch.device
) -> torch.Tensor:
    """Where each row's index list starts.

    Every row owns num_kv_blocks entries, so the starts follow from the shape
    and are held for as long as the shape repeats.
    """
    key = (num_work_items, num_kv_blocks, device)
    start = _LUT_STARTS.get(key)
    if start is None:
        start = (
            torch.arange(num_work_items, dtype=torch.int32, device=device)
            * num_kv_blocks
        )
        _LUT_STARTS[key] = start
    return start


def route_and_pack(
    proxy: torch.Tensor,
    beta: float | torch.Tensor,
    num_kv_blocks: int,
    bitmap_ds: int,
    partial_tail: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Selection mask, exact-block count, packed bitmap and index list.

    proxy is (batch, nheads_q, num_q_tiles, num_kv_blocks) float32. The index
    list holds num_kv_blocks entries per row, which row_lut_start addresses.
    """
    if isinstance(beta, torch.Tensor):
        if beta.numel() != 1 or beta.device != proxy.device:
            raise ValueError("beta must be a scalar tensor on the proxy device.")
    batch, num_heads, num_q_tiles, blocks = proxy.shape
    if blocks != num_kv_blocks:
        raise ValueError(
            f"proxy's last axis is {blocks}, expected num_kv_blocks={num_kv_blocks}."
        )
    num_work_items = batch * num_heads * num_q_tiles
    num_words = triton.next_power_of_2(bitmap_ds * 32) // 32
    # Keep about 32 lanes per work item as the row grows.
    num_warps = max(4, min(16, num_words * 32 // 256))

    mask = torch.empty(
        (batch, num_heads, num_q_tiles, num_kv_blocks),
        dtype=torch.bool,
        device=proxy.device,
    )
    lut_count = torch.empty(num_work_items, dtype=torch.int32, device=proxy.device)
    bitmap = torch.empty(
        (num_work_items, bitmap_ds), dtype=torch.int32, device=proxy.device
    )
    kv_block_indices = torch.empty(
        num_work_items * num_kv_blocks, dtype=torch.int32, device=proxy.device
    )

    _sol_route_pack_kernel[(num_work_items,)](
        proxy,
        mask,
        lut_count,
        bitmap,
        kv_block_indices,
        proxy.stride(0),
        proxy.stride(1),
        proxy.stride(2),
        proxy.stride(3),
        mask.stride(0),
        mask.stride(1),
        mask.stride(2),
        mask.stride(3),
        beta,
        num_kv_blocks,
        num_q_tiles,
        num_heads,
        bitmap_ds,
        BETA_IS_TENSOR=isinstance(beta, torch.Tensor),
        PARTIAL_TAIL=partial_tail,
        NUM_WORDS=num_words,
        num_warps=num_warps,
    )
    return mask, lut_count, bitmap.view(torch.uint32), kv_block_indices
