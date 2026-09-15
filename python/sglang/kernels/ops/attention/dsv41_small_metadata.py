"""Small-batch V4.1 page and compression metadata."""

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.dsv4_attn_metadata_kernels import (
    PageTablePositionsResult,
)


@triton.jit
def _small_page_table(
    REQ_TO_TOKEN,
    REQS,
    LENS,
    OUT_LENS,
    POS,
    PAGES,
    SWA,
    STRIDE: tl.constexpr,
    NUM_PAGES: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    WINDOW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row, tile = tl.program_id(0), tl.program_id(1)
    if tile == 0:
        length = tl.load(LENS + row).to(tl.int32)
        tl.store(OUT_LENS + row, length)
        tl.store(POS + row, length - 1)
        tl.store(SWA + row, tl.minimum(length, WINDOW))
    req = tl.load(REQS + row).to(tl.int64)
    p = tile * BLOCK + tl.arange(0, BLOCK)
    slot = tl.load(
        REQ_TO_TOKEN + req * STRIDE + p.to(tl.int64) * PAGE_SIZE,
        mask=p < NUM_PAGES,
        other=0,
    ).to(tl.int32)
    tl.store(PAGES + row * NUM_PAGES + p, slot // PAGE_SIZE, mask=p < NUM_PAGES)


def page_table_positions_small(
    *,
    req_to_token,
    req_pool_indices_repeated,
    seq_lens_casual,
    max_seq_len,
    page_size,
    swa_window,
):
    assert page_size > 0 and page_size & (page_size - 1) == 0
    rows = seq_lens_casual.numel()
    pages = triton.cdiv(max_seq_len, page_size)
    kw = dict(device=seq_lens_casual.device, dtype=torch.int32)
    lengths, positions, swa = [torch.empty(rows, **kw) for _ in range(3)]
    table = torch.empty((rows, pages), **kw)
    _small_page_table[(rows, triton.cdiv(pages, 256))](
        req_to_token,
        req_pool_indices_repeated,
        seq_lens_casual,
        lengths,
        positions,
        table,
        swa,
        req_to_token.stride(0),
        pages,
        page_size,
        swa_window,
        256,
    )
    return PageTablePositionsResult(
        seq_lens_casual=lengths,
        positions_casual=positions,
        page_table=table,
        swa_topk_lengths=swa,
    )


@triton.jit
def _low_ratio_metadata(
    LENS,
    LOC,
    OUT1,
    LEN1,
    SPARSE1,
    PAGE1,
    OUT2,
    LEN2,
    SPARSE2,
    PAGE2,
    TOPK: tl.constexpr,
    PADDED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    length = tl.load(LENS + row).to(tl.int32)
    loc = tl.load(LOC + row).to(tl.int64)
    len1, len2 = tl.maximum(length, 1), tl.maximum(length >> 1, 1)
    tl.store(OUT1 + row, loc)
    tl.store(OUT2 + row, tl.where((length & 1) == 0, loc >> 1, -1))
    tl.store(LEN1 + row, len1)
    tl.store(LEN2 + row, len2)
    tl.store(SPARSE1 + row, tl.minimum(len1, TOPK))
    tl.store(SPARSE2 + row, tl.minimum(len2, TOPK))
    cols = tl.arange(0, BLOCK)
    tl.store(PAGE1 + row * PADDED + cols, -1, cols < PADDED)
    tl.store(PAGE2 + row * PADDED + cols, -1, cols < PADDED)


def low_ratio_metadata(seq_lens, out_loc, topk):
    assert seq_lens.numel() == out_loc.numel()
    rows = seq_lens.numel()
    kw = dict(device=seq_lens.device, dtype=torch.int32)
    padded = triton.cdiv(topk, 64) * 64
    outputs = []
    for _ in range(2):
        outputs.extend(
            [
                torch.empty(rows, device=out_loc.device, dtype=torch.int64),
                torch.empty(rows, **kw),
                torch.empty(rows, **kw),
                torch.empty((rows, padded), **kw),
            ]
        )
    _low_ratio_metadata[(rows,)](
        seq_lens, out_loc, *outputs, topk, padded, triton.next_power_of_2(padded)
    )
    return outputs
