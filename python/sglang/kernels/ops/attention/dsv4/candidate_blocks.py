"""Candidate blocks of the two-level indexer: a candidate source keeps whole
blocks of ``block_size`` compressed positions, its newest block always, and a
consumer selects among the kept blocks only.

The per-row block counts and sparse-row lengths, the block selection (the JIT
chain on SM100, torch elsewhere), and the torch top-k among chosen blocks."""

from typing import List, Optional, Union

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from sglang.kernels.jit.utils import is_arch_support_pdl


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


def amax_topk_blocks(
    logits: torch.Tensor,
    seq_lens: torch.Tensor,
    nblocks: torch.Tensor,
    topk_blocks: int,
    max_seq_len: Optional[int] = None,
) -> torch.Tensor:
    """Per row the ``topk_blocks`` blocks of 8 positions with the largest block
    maximum among its first ``seq_lens[b]`` positions, the newest block always
    included: block ids in no particular order, ``-1`` past the row's count.
    ``nblocks`` is ``ceil(seq_lens / 8)`` as int32."""
    from .candidate_table import amax8_varlen
    from .topk import plan_topk_v2, topk_transform_paged_v2

    rows = logits.shape[0]
    block = 8
    if max_seq_len is None:
        max_seq_len = logits.shape[1]
    # NOTE: plan cannot be the previous kernel of topk_transform_paged_v2
    plan = plan_topk_v2(nblocks)
    # block maxima, the newest block +inf; the top-k reads each row up to nblocks
    # only, so nothing past a row's keys is initialised (v2 needs stride % 4 == 0)
    keys = logits.new_empty(rows, -(-max_seq_len // (4 * block)) * 4)
    amax8_varlen(logits, seq_lens, out=keys)
    blocks = torch.empty(rows, topk_blocks, dtype=torch.int32, device=logits.device)
    topk_transform_paged_v2(keys, nblocks, None, blocks, 1, plan)
    return blocks


def select_candidate_block_ids(
    logits: torch.Tensor,
    compress_lens: Union[torch.Tensor, int],
    topk_blocks: int,
    block_size: int,
) -> torch.Tensor:
    """The torch block selection: per row the ids of the ``topk_blocks`` blocks
    with the largest maximum score, the block holding position
    ``compress_lens - 1`` always kept; int32 ``[rows, min(topk_blocks, blocks)]``,
    unordered, ``-1`` where a row has fewer finite blocks. ``logits`` must already
    be ``-inf`` past each row's causal length."""
    width = logits.size(-1)
    padding = -width % block_size
    scores = F.pad(logits, (0, padding), value=-torch.inf) if padding else logits
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    num_blocks = scores.size(-1)
    last = (compress_lens - 1) // block_size
    scores = scores.masked_fill(
        torch.arange(num_blocks, device=logits.device) == last, torch.inf
    )
    top = scores.topk(min(topk_blocks, num_blocks), dim=-1)
    return top.indices.to(torch.int32).masked_fill_(~(top.values > -torch.inf), -1)


def topk_among_blocks(
    scores: torch.Tensor,
    lens: torch.Tensor,
    blocks: torch.Tensor,
    k: int,
    block_size: int,
) -> torch.Tensor:
    """The torch top-``k`` of each row among its chosen blocks: ``scores``
    ``[rows, width]`` indexed by position, ``lens`` ``[rows]`` the causal length,
    ``blocks`` ``[rows, n]`` block ids with ``-1`` padding. Returns int64
    ``[rows, k]`` positions, unordered, ``-1`` where a row has fewer than ``k``
    finite candidates; positions at or past ``lens`` never count."""
    rows, width = scores.shape
    if width == 0:
        return torch.full((rows, k), -1, dtype=torch.int64, device=scores.device)
    offsets = torch.arange(block_size, device=scores.device)
    columns = (blocks.to(torch.int64)[:, :, None] * block_size + offsets).flatten(1)
    valid = (
        (blocks >= 0).repeat_interleave(block_size, dim=1)
        & (columns < lens.to(torch.int64)[:, None])
        & (columns < width)
    )
    candidates = scores.gather(1, columns.clamp(0, width - 1))
    candidates = candidates.masked_fill(~valid, -torch.inf)
    top = candidates.topk(min(k, candidates.shape[1]), dim=-1, sorted=False)
    picked = columns.gather(1, top.indices).masked_fill(~(top.values > -torch.inf), -1)
    if picked.shape[1] < k:
        picked = F.pad(picked, (0, k - picked.shape[1]), value=-1)
    return picked


def get_tail_row_indices(
    full_rows_per_request: List[int],
    tail_rows_per_request: List[int],
    device: torch.device,
) -> torch.Tensor:
    """int64 indices of each request's last ``tail_rows_per_request[b]`` rows, in
    row order, copied without a host sync (pinned staging, non-blocking)."""
    rows, start = [], 0
    for n, t in zip(full_rows_per_request, tail_rows_per_request):
        rows.extend(range(start + n - t, start + n))
        start += n
    device = torch.device(device)
    if device.type != "cuda":
        return torch.tensor(rows, dtype=torch.int64, device=device)
    staged = torch.tensor(rows, dtype=torch.int64, pin_memory=True)
    return staged.to(device, non_blocking=True)
