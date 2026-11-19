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
def _jit_block_amax_module(in_dtype: torch.dtype, out_dtype: torch.dtype):
    args = make_cpp_args(is_arch_support_pdl(), in_dtype, out_dtype)
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
    ``scores`` and ``out`` are each fp32 or bf16 (``out`` defaults to fp32), the
    rows of ``scores`` aligned to a block of 8 (stride a multiple of 8). Returns
    ``out``.
    """
    if out is None:
        num_tokens, max_len = scores.shape
        if max_seqlen == 0:
            max_seqlen = max_len
        out = scores.new_empty(num_tokens, (max_seqlen + 7) // 8, dtype=torch.float32)
    module = _jit_block_amax_module(scores.dtype, out.dtype)
    module.amax8_varlen(scores, seq_lens, out, topk)
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
    keys = logits.new_empty(
        rows, -(-max_seq_len // (4 * block)) * 4, dtype=torch.float32
    )
    amax8_varlen(logits, seq_lens, out=keys)
    blocks = torch.empty(rows, topk_blocks, dtype=torch.int32, device=logits.device)
    topk_transform_paged_v2(keys, nblocks, None, blocks, 1, plan)
    return blocks


def candidate_block_mask(
    blocks: torch.Tensor, width: int, block_size: int
) -> torch.Tensor:
    """Materialize a source's block IDs once for Hopper paged-score consumers."""
    num_blocks = (width + block_size - 1) // block_size
    keep = torch.zeros(
        (*blocks.shape[:-1], num_blocks + 1), dtype=torch.bool, device=blocks.device
    )
    keep.scatter_(-1, blocks.to(torch.int64).masked_fill(blocks < 0, num_blocks), True)
    return keep[..., :num_blocks].repeat_interleave(block_size, dim=-1)[..., :width]


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


@triton.jit
def _radix_select_candidate_ids(
    ordered, blocks, OUTPUT, row, K: tl.constexpr, KEY_BITS: tl.constexpr
):
    # Narrow the Kth key to one radix bucket per pass.
    bins = tl.arange(0, 256)
    prefix = tl.full((), 0, tl.uint32)
    rank = K
    for shift in tl.static_range(KEY_BITS - 8, -1, -8):
        digit = ((ordered >> shift) & 255).to(tl.int32)
        active = (ordered >> shift >> 8) == (prefix >> shift >> 8)
        hist = tl.histogram(digit, 256, mask=active)
        above = tl.cumsum(hist, axis=0, reverse=True) - hist
        bucket = tl.max(tl.where(above + hist >= rank, bins, -1), axis=0)
        rank -= tl.sum(tl.where(bins == bucket, above, 0), axis=0)
        prefix |= bucket.to(tl.uint32) << shift
    equal = ordered == prefix
    equal_rank = tl.cumsum(equal.to(tl.int32), axis=0)
    keep = (ordered > prefix) | (equal & (equal_rank <= rank))
    # Zero and the all-ones key represent padding and NaN, respectively.
    picked = keep & (ordered != 0) & (ordered != ((1 << KEY_BITS) - 1))
    count = tl.cumsum(picked.to(tl.int32), axis=0)
    total = tl.sum(picked.to(tl.int32), axis=0)
    tl.store(OUTPUT + row * K + count - 1, blocks, picked)
    tl.store(OUTPUT + row * K + blocks, -1, (blocks >= total) & (blocks < K))


@triton.jit
def _ordered_candidate_keys(maxima, valid, has_nan, KEY_BITS: tl.constexpr):
    bits = maxima.to(tl.uint32, bitcast=True)
    ordered = tl.where(bits & 0x80000000 != 0, ~bits, bits | 0x80000000)
    ordered = tl.where(has_nan, 0xFFFFFFFF, tl.where(valid, ordered, 0))
    return ordered >> (32 - KEY_BITS)


@triton.jit(do_not_specialize=["width", "row_stride"])
def _reduce_candidate_keys_kernel(
    LOGITS,
    LENGTHS,
    KEYS,
    width,
    row_stride,
    length,
    NUM_BLOCKS,
    LENGTH_STRIDE: tl.constexpr,
    HAS_LENGTHS: tl.constexpr,
    KEY_BITS: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    block = tl.program_id(1) * TILE + tl.arange(0, TILE)
    cols = block[:, None] * 8 + tl.arange(0, 8)[None, :]
    scores = tl.load(LOGITS + row * row_stride + cols, cols < width, -float("inf")).to(
        tl.float32
    )
    maxima = tl.max(scores, axis=1)
    has_nan = tl.max((scores != scores).to(tl.int32), axis=1) != 0
    if HAS_LENGTHS:
        length = tl.load(LENGTHS + row * LENGTH_STRIDE)
    is_newest = (length > 0) & (block == (length - 1) // 8)
    maxima = tl.where(is_newest, float("inf"), maxima)
    has_nan &= ~is_newest
    keys = _ordered_candidate_keys(
        maxima, (maxima > -float("inf")) & ~has_nan, has_nan, KEY_BITS
    )
    tl.store(KEYS + row * NUM_BLOCKS + block, keys, block < NUM_BLOCKS)


@triton.jit(do_not_specialize=["num_blocks"])
def _select_reduced_candidate_keys_kernel(
    KEYS,
    OUTPUT,
    num_blocks,
    K: tl.constexpr,
    TILE: tl.constexpr,
    KEY_BITS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    block = tl.arange(0, TILE)
    keys = tl.load(KEYS + row * num_blocks + block, block < num_blocks, 0)
    _radix_select_candidate_ids(keys.to(tl.uint32), block, OUTPUT, row, K, KEY_BITS)


@triton.jit(do_not_specialize=["width", "row_stride"])
def _select_candidate_block_ids_kernel(
    LOGITS,
    LENGTHS,
    OUTPUT,
    width,
    row_stride,
    length,
    LENGTH_STRIDE: tl.constexpr,
    HAS_LENGTHS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    K: tl.constexpr,
    TILE: tl.constexpr,
    SELECT_ALL: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    blocks = tl.arange(0, TILE)
    cols = blocks[:, None] * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[None, :]
    scores = tl.load(LOGITS + row * row_stride + cols, cols < width, -float("inf")).to(
        tl.float32
    )
    maxima = tl.max(scores, axis=1)
    has_nan = tl.max((scores != scores).to(tl.int32), axis=1) != 0
    if HAS_LENGTHS:
        length = tl.load(LENGTHS + row * LENGTH_STRIDE)
    newest = (length - 1) // BLOCK_SIZE
    is_newest = (length > 0) & (blocks == newest)
    maxima = tl.where(is_newest, float("inf"), maxima)
    has_nan &= ~is_newest
    valid = (blocks * BLOCK_SIZE < width) & (maxima > -float("inf")) & ~has_nan

    if SELECT_ALL:
        # Keep valid IDs at the front without sorting the block scores.
        count = tl.cumsum(valid.to(tl.int32), axis=0)
        total = tl.sum(valid.to(tl.int32), axis=0)
        dest = tl.where(valid, count - 1, total + blocks - count)
        selected = tl.where(valid, blocks, -1)
        tl.store(OUTPUT + row * K + dest, selected, blocks < K)
    else:
        KEY_BITS: tl.constexpr = 16 if LOGITS.dtype.element_ty == tl.bfloat16 else 32
        keys = _ordered_candidate_keys(maxima, valid, has_nan, KEY_BITS)
        _radix_select_candidate_ids(keys, blocks, OUTPUT, row, K, KEY_BITS)


def select_candidate_block_ids_triton(
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
    if (
        logits.is_cuda
        and torch.version.hip is None
        and logits.ndim == 2
        and logits.stride(1) == 1
        and logits.dtype in (torch.float32, torch.bfloat16)
        and block_size == 8
        and 0 < width <= 131072
        and (width <= 32768 or topk_blocks == 2048)
        and (
            isinstance(compress_lens, int)
            or (
                isinstance(compress_lens, torch.Tensor)
                and compress_lens.device == logits.device
                and compress_lens.dtype in (torch.int32, torch.int64)
                and compress_lens.shape == (logits.shape[0], 1)
            )
        )
    ):
        rows = logits.shape[0]
        num_blocks = triton.cdiv(width, block_size)
        k = min(topk_blocks, num_blocks)
        out = torch.empty((rows, k), dtype=torch.int32, device=logits.device)
        if rows and k:
            has_lengths = isinstance(compress_lens, torch.Tensor)
            if k == 2048 and num_blocks > k:
                # Parallelize the full-width read and keep only one key per block
                # live during selection.
                key_bits = 16 if logits.dtype == torch.bfloat16 else 32
                keys = torch.empty(
                    (rows, num_blocks),
                    dtype=torch.uint16 if key_bits == 16 else torch.uint32,
                    device=logits.device,
                )
                _reduce_candidate_keys_kernel[(rows, triton.cdiv(num_blocks, 256))](
                    logits,
                    compress_lens if has_lengths else None,
                    keys,
                    width,
                    logits.stride(0),
                    0 if has_lengths else compress_lens,
                    num_blocks,
                    compress_lens.stride(0) if has_lengths else 0,
                    has_lengths,
                    key_bits,
                    256,
                    num_warps=4,
                )
                _select_reduced_candidate_keys_kernel[(rows,)](
                    keys,
                    out,
                    num_blocks,
                    k,
                    triton.next_power_of_2(num_blocks),
                    key_bits,
                    num_warps=16,
                )
                return out
            _select_candidate_block_ids_kernel[(rows,)](
                logits,
                compress_lens if has_lengths else None,
                out,
                width,
                logits.stride(0),
                0 if has_lengths else compress_lens,
                compress_lens.stride(0) if has_lengths else 0,
                has_lengths,
                block_size,
                k,
                triton.next_power_of_2(num_blocks),
                k == num_blocks,
                num_warps=8,
            )
        return out
    return select_candidate_block_ids(logits, compress_lens, topk_blocks, block_size)


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
        # the DeepGEMM callers slice their tile block-aligned, so this is a no-op there
        scores = F.pad(scores, (0, -width % block_size), value=-torch.inf)
    num_blocks = scores.shape[1] // block_size
    by_block = scores.unflatten(1, (num_blocks, block_size))
    blocks64 = blocks.to(torch.int64)
    ids = blocks64.clamp(0, num_blocks - 1)
    # [rows, n, block_size], the only scratch of the row count's size
    candidates = by_block.gather(1, ids[:, :, None].expand(rows, n, block_size))
    # column j of block b is position b * block_size + j, kept if j < lens - b * block_size
    offsets = torch.arange(block_size, device=scores.device, dtype=lens.dtype)
    room = lens[:, None] - blocks64 * block_size
    drop = (offsets[None, None, :] >= room[:, :, None]) | (blocks64 < 0)[:, :, None]
    candidates.masked_fill_(drop, -torch.inf)
    return candidate_topk(candidates.flatten(1), blocks64, k, block_size)


def candidate_topk(
    scores: torch.Tensor,
    blocks: torch.Tensor,
    k: int,
    block_size: int = CANDIDATE_BLOCK_SIZE,
) -> torch.Tensor:
    """Top-k masked candidate-column scores as request-local positions, -1 padded."""
    if scores.shape[1] == 0:
        return torch.full(
            (scores.shape[0], k), -1, dtype=torch.int64, device=scores.device
        )
    top = scores.topk(min(k, scores.shape[1]), dim=-1, sorted=False)
    block_of_pick = top.indices // block_size
    picked = blocks.to(torch.int64).gather(1, block_of_pick) * block_size + (
        top.indices - block_of_pick * block_size
    )
    picked = picked.masked_fill(top.values == -torch.inf, -1)
    if picked.shape[1] < k:
        picked = F.pad(picked, (0, k - picked.shape[1]), value=-1)
    return picked
