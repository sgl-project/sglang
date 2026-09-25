"""Candidate blocks of the two-level indexer: the per-row block counts and
sparse-row lengths, the JIT block top-k of the sparse table, and the torch
block ids and masks."""

from typing import Optional, Union

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from sglang.kernels.jit.utils import is_arch_support_pdl

from .candidate_table import CANDIDATE_BLOCK_SIZE, amax8_varlen
from .topk import plan_topk_v2, topk_transform_paged_v2


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
    rows = logits.shape[0]
    block = CANDIDATE_BLOCK_SIZE
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


def mask_topk_scores(
    scores: torch.Tensor,
    indices: torch.Tensor,
    offsets: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Keep masked indexer scores out of attention even when top-k underfills."""
    columns = indices.to(torch.int64)
    if offsets is not None:
        columns = columns - offsets[:, None]
    selected_scores = scores.gather(1, columns.clamp(0, scores.shape[1] - 1))
    valid = (
        (columns >= 0) & (columns < scores.shape[1]) & (selected_scores > -torch.inf)
    )
    return indices.masked_fill(~valid, -1)


def _candidate_block_topk(
    logits: torch.Tensor,
    compress_lens: Union[torch.Tensor, int],
    topk_blocks: int,
    block_size: int,
) -> torch.return_types.topk:
    width = logits.size(-1)
    padding = -width % block_size
    scores = F.pad(logits, (0, padding), value=-torch.inf) if padding else logits
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    num_blocks = scores.size(-1)

    last = (compress_lens - 1) // block_size
    scores = scores.masked_fill(
        torch.arange(num_blocks, device=logits.device) == last, torch.inf
    )

    return scores.topk(min(topk_blocks, num_blocks), dim=-1)


def select_candidate_block_ids(
    logits: torch.Tensor,
    compress_lens: Union[torch.Tensor, int],
    topk_blocks: int,
    block_size: int,
) -> torch.Tensor:
    top = _candidate_block_topk(
        logits=logits,
        compress_lens=compress_lens,
        topk_blocks=topk_blocks,
        block_size=block_size,
    )
    return top.indices.to(torch.int32).masked_fill_(~(top.values > -torch.inf), -1)


def candidate_block_mask(
    blocks: torch.Tensor, width: int, block_size: int
) -> torch.Tensor:
    num_blocks = (width + block_size - 1) // block_size
    keep = torch.zeros(
        (*blocks.shape[:-1], num_blocks + 1), dtype=torch.bool, device=blocks.device
    )
    keep.scatter_(-1, blocks.to(torch.int64).masked_fill(blocks < 0, num_blocks), True)
    return keep[..., :num_blocks].repeat_interleave(block_size, dim=-1)[..., :width]


def select_candidate_blocks(
    logits: torch.Tensor,
    compress_lens: Union[torch.Tensor, int],
    topk_blocks: int,
    block_size: int,
) -> torch.Tensor:
    top = _candidate_block_topk(
        logits=logits,
        compress_lens=compress_lens,
        topk_blocks=topk_blocks,
        block_size=block_size,
    )
    width = logits.shape[-1]
    num_blocks = (width + block_size - 1) // block_size
    keep = torch.zeros(
        (*logits.shape[:-1], num_blocks), dtype=torch.bool, device=logits.device
    ).scatter_(-1, top.indices, top.values > -torch.inf)
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]
