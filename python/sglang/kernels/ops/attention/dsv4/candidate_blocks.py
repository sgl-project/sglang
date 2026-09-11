"""Candidate-block scores and visibility masking for paged indexer logits."""

import torch
import triton
import triton.language as tl

from sglang.kernels.jit.utils import is_arch_support_pdl


@triton.jit
def _maximum_with_nan(a, b):
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@triton.jit
def _candidate_scores_kernel(
    X,
    LENS,
    OUT,
    SCORES,
    WIDTH: tl.constexpr,
    STRIDE: tl.constexpr,
    BLOCKS: tl.constexpr,
    GROUP: tl.constexpr,
    GROUP_PAD: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    blocks = tl.program_id(1) * TILE + tl.arange(0, TILE)
    offsets = tl.arange(0, GROUP_PAD)
    cols = blocks[:, None] * GROUP + offsets[None, :]
    length = tl.load(LENS + row)
    in_bounds = (cols < WIDTH) & (offsets[None, :] < GROUP)
    values = tl.load(
        X + row * STRIDE + cols, in_bounds & (cols < length), other=-float("inf")
    ).to(tl.float32)
    tl.store(OUT + row * WIDTH + cols, values, in_bounds)
    scores = tl.reduce(values, axis=1, combine_fn=_maximum_with_nan)
    scores = tl.where(
        (length > 0) & (blocks == (length - 1) // GROUP), float("inf"), scores
    )
    tl.store(SCORES + row * BLOCKS + blocks, scores, blocks < BLOCKS)


@triton.jit
def _candidate_mask_kernel(
    X,
    LENS,
    KEEP,
    OUT,
    WIDTH: tl.constexpr,
    STRIDE: tl.constexpr,
    KEEP_STRIDE: tl.constexpr,
    KEEP_COL_STRIDE: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * TILE + tl.arange(0, TILE)
    visible = (cols < WIDTH) & (cols < tl.load(LENS + row))
    keep = tl.load(KEEP + row * KEEP_STRIDE + cols * KEEP_COL_STRIDE, visible, other=0)
    values = tl.load(X + row * STRIDE + cols, visible & keep, other=-float("inf")).to(
        tl.float32
    )
    tl.store(OUT + row * WIDTH + cols, values, cols < WIDTH)


@triton.jit
def _publish_candidate_mask_kernel(
    INDICES,
    VALUES,
    KEEP,
    WIDTH: tl.constexpr,
    GROUP: tl.constexpr,
    TOPK: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    i = tl.program_id(1) * TILE + tl.arange(0, TILE)
    selected = tl.load(INDICES + row * TOPK + i // GROUP, i < TOPK * GROUP, 0)
    score = tl.load(VALUES + row * TOPK + i // GROUP, i < TOPK * GROUP, -float("inf"))
    cols = selected * GROUP + i % GROUP
    # torch.topk returns unique block indices: each output position has one writer.
    tl.store(
        KEEP + row * WIDTH + cols,
        score > -float("inf"),
        (i < TOPK * GROUP) & (cols < WIDTH),
    )


def candidate_block_logits(
    logits: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    topk_blocks: int,
    block_size: int,
    published: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Keep torch.topk's block selection, including its tie behavior.

    A source masks the unread tail while reducing each block. A consumer masks
    visibility and the published candidates in one pass, without copying the
    capacity-sized logits before each masked_fill.
    """
    rows, width = logits.shape
    output = torch.empty((rows, width), dtype=torch.float32, device=logits.device)
    if published is not None:
        _candidate_mask_kernel[(rows, triton.cdiv(width, 4096))](
            logits,
            seq_lens,
            published,
            output,
            width,
            logits.stride(0),
            published.stride(0),
            published.stride(1),
            4096,
        )
        return output, None

    blocks = triton.cdiv(width, block_size)
    scores = torch.empty((rows, blocks), dtype=torch.float32, device=logits.device)
    group_pad = triton.next_power_of_2(block_size)
    tile = max(1, 1024 // group_pad)
    _candidate_scores_kernel[(rows, triton.cdiv(blocks, tile))](
        logits,
        seq_lens,
        output,
        scores,
        width,
        logits.stride(0),
        blocks,
        block_size,
        group_pad,
        tile,
    )
    # Publication only needs membership; sorting the selected pairs is unused.
    top = scores.topk(min(topk_blocks, blocks), dim=-1, sorted=False)
    keep = torch.zeros((rows, width), dtype=torch.bool, device=logits.device)
    _publish_candidate_mask_kernel[
        (rows, triton.cdiv(top.indices.shape[1] * block_size, 256))
    ](
        top.indices,
        top.values,
        keep,
        width,
        block_size,
        top.indices.shape[1],
        256,
        num_warps=4,
    )
    return output, keep


@triton.jit
def _candidate_row_lens_kernel(
    LENS,
    NBLOCKS,
    VALID,
    ROWS,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
    TILE: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    rows = tl.program_id(0) * TILE + tl.arange(0, TILE)
    mask = rows < ROWS
    if USE_PDL:
        tl.extra.cuda.gdc_wait()  # LENS is the previous kernel's output
    length = tl.load(LENS + rows, mask, 0).to(tl.int32)
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()
    nblocks = (length + (BLOCK - 1)) // BLOCK
    kept = tl.minimum(nblocks, TOPK)
    # the kept blocks laid out back to back, the newest one possibly partial
    valid = BLOCK * (kept - 1) + (length - 1) % BLOCK + 1
    valid = tl.where(length > 0, valid, 0)
    tl.store(NBLOCKS + rows, nblocks, mask)
    tl.store(VALID + rows, valid, mask)


def candidate_row_lens(
    seq_lens: torch.Tensor, topk_blocks: int, block_size: int = 8
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per row: its number of blocks ``ceil(seq_len / block_size)`` and the
    length of its sparse logits row once the ``min(topk_blocks, blocks)`` kept
    blocks are laid out back to back (the newest block possibly partial):
    ``block_size * (kept - 1) + (seq_len - 1) % block_size + 1``. Both int32
    ``[rows]``; a zero-length row gets 0 for both."""
    assert seq_lens.dim() == 1 and seq_lens.is_contiguous()
    rows = seq_lens.numel()
    nblocks = torch.empty(rows, dtype=torch.int32, device=seq_lens.device)
    valid = torch.empty_like(nblocks)
    tile = 256
    use_pdl = is_arch_support_pdl()
    pdl_kwargs = {"launch_pdl": True} if use_pdl else {}
    _candidate_row_lens_kernel[(triton.cdiv(rows, tile),)](
        seq_lens,
        nblocks,
        valid,
        rows,
        topk_blocks,
        block_size,
        tile,
        use_pdl,
        num_warps=4,
        **pdl_kwargs,
    )
    return nblocks, valid
