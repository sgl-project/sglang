"""Candidate blocks of the two-level indexer: a source keeps whole blocks of
``block_size`` compressed positions, its newest block always, and a consumer
selects among them. Block counts, block keys, the block top-k (JIT / torch), and
the torch top-k among chosen blocks."""

from typing import Optional, Union

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

from .candidate_table import CANDIDATE_BLOCK_SIZE
from .topk import plan_topk_v2, topk_transform_paged_v2
from .utils import make_name


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


@cache_once
def _jit_block_amax_module():
    args = make_cpp_args(is_arch_support_pdl())
    return load_jit(
        make_name("block_amax"),
        *args,
        cuda_files=["deepseek_v4/block_amax.cuh"],
        cuda_wrappers=[("amax8_varlen", f"BlockAmaxKernel<{args}>::amax8_varlen")],
    )


def amax8_varlen(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    topk: int = 0,
    *,
    max_seqlen: int = 0,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Level-one keys of the two-level indexer: ``out[b, i]`` is the max of
    ``scores[b, 8 i : 8 i + 8]`` for ``i < ceil(seq_lens[b] / 8)``, the last of
    them ``+inf`` (the newest block is always selected), nothing written past
    that count. Rows with at most ``topk`` blocks are skipped (every block is
    selected anyway); ``topk=0`` never skips. ``out`` is allocated as
    ``[rows, ceil(max_seqlen / 8)]`` when not given, ``max_seqlen`` defaulting to
    the width of ``scores``; every ``seq_lens[b]`` must fit in ``8 * out.shape[1]``.
    fp32 only for now; ``scores`` rows must be 32-byte aligned (stride a multiple
    of 8). Returns ``out``.
    """
    if out is None:
        num_tokens, max_len = scores.shape
        if max_seqlen == 0:
            max_seqlen = max_len
        out = scores.new_empty(num_tokens, (max_seqlen + 7) // 8)
    _jit_block_amax_module().amax8_varlen(scores, seq_lens, out, topk)
    return out


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


def select_candidate_block_ids(
    logits: torch.Tensor,
    compress_lens: Union[torch.Tensor, int],
    topk_blocks: int,
    block_size: int,
) -> torch.Tensor:
    """Per row the ids of the ``topk_blocks`` blocks with the largest score, the
    block of position ``compress_lens - 1`` always kept: int32
    ``[rows, min(topk_blocks, blocks)]``, unordered, ``-1`` past the finite blocks.
    ``logits`` must be ``-inf`` past each row's causal length."""
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
    """Top-``k`` of each row of ``scores`` ``[rows, width]`` within its ``blocks``
    ``[rows, n]`` (``-1`` padded) and its causal ``lens``: int64 ``[rows, k]``
    positions, unordered, ``-1`` where fewer than ``k`` candidates are finite."""
    rows, width = scores.shape
    n = blocks.shape[1]
    if width == 0 or n == 0:
        return torch.full((rows, k), -1, dtype=torch.int64, device=scores.device)
    if width % block_size:
        # A block-aligned width lets the gather below index blocks, not columns;
        # callers on the DeepGEMM path slice their tile to one so this is a no-op.
        scores = F.pad(scores, (0, -width % block_size), value=-torch.inf)
    num_blocks = scores.shape[1] // block_size
    by_block = scores.unflatten(1, (num_blocks, block_size))
    ids = blocks.to(torch.int64).clamp_(0, num_blocks - 1)
    # [rows, n, block_size], the only scratch of the row count's size
    candidates = by_block.gather(1, ids[:, :, None].expand(rows, n, block_size))
    # column j of block b is position b * block_size + j, kept if b is a real
    # block and the position is causal: j < lens - b * block_size
    offsets = torch.arange(block_size, device=scores.device, dtype=torch.int32)
    room = lens.to(torch.int32)[:, None] - blocks.to(torch.int32) * block_size
    valid = (blocks >= 0)[:, :, None] & (offsets[None, None, :] < room[:, :, None])
    candidates.masked_fill_(~valid, -torch.inf)
    flat = candidates.flatten(1)
    top = flat.topk(min(k, flat.shape[1]), dim=-1, sorted=False)
    picked = blocks.to(torch.int64).gather(1, top.indices // block_size) * block_size
    picked = (picked + top.indices % block_size).masked_fill(
        ~(top.values > -torch.inf), -1
    )
    if picked.shape[1] < k:
        picked = F.pad(picked, (0, k - picked.shape[1]), value=-1)
    return picked
