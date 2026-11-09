"""MTP draft-extend sparse attention on the target verify's FP8 pool-union path
(SGLANG_ENABLE_QSA_DRAFT_EXTEND_UNION, default off).

The spec-v2 draft extend lays its rows out as target verify does: a window of four rows per request, in position
order, row j seeing positions up to its own (graph metadata mode 2: length = position + 1, compressed length =
length // 4). Its stock attention stores the new K/V, counts each row's valid tokens, packs every row's up to 2051
selected tokens (_compact_kv) and runs trtllm decode over the packed rows. This runs the verify's chain instead
(pool_union_ws.qsa_pool_union_decode_ws): write the new K/V, each request's token union once (kernel U), the
warp-specialized union attention and the merge, reading the FP8 pool in place.

Same attended set per row: the indexer's expanded indices hold block c's first token at column ratio * c (the
Triton expansion keeps column positions), so the block indices are that column floor-divided by ratio, and with
length = position + 1 and compressed length = length // ratio every selected block is whole and before the row's
uncompressed tail, which both paths take as positions (position + 1) // ratio * ratio .. position. The numerics are
the union path's (FP32 accumulation order and tiling differ from trtllm decode), so the draft's hidden states can
differ at the BF16-ulp level: not bitwise for the draft; target-verified tokens are unaffected for greedy rows.
"""

from __future__ import annotations

import torch

from sglang.srt.layers.attention.qsa import pool_union_decode as union
from sglang.srt.layers.attention.qsa import pool_union_ws

WIDTH = 4


def block_indices_of(token_indices: torch.Tensor, *, ratio: int, block_topk: int) -> torch.Tensor:
    """[rows, block_topk] int32 block ids from expand_qsa_block_indices' [rows, block_topk * ratio + ratio - 1]
    output: column ratio * c holds block c's first token (or -1), so floor division by ratio gives the block (-1
    stays -1). Columns past a row's whole blocks hold tail positions; the union kernel never reads them (it takes
    c < min(length // ratio, block_topk))."""
    return torch.div(token_indices[:, : block_topk * ratio : ratio], ratio, rounding_mode="floor")


def attend(
    *,
    q: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    loc: torch.Tensor,
    req_to_token: torch.Tensor,
    row_requests: torch.Tensor,
    token_indices: torch.Tensor,
    sequence_lens: torch.Tensor,
    query_positions: torch.Tensor,
    scale: float,
    ratio: int,
    block_topk: int,
) -> torch.Tensor:
    """Writes k/v_new at loc and returns the rows' sparse attention [rows, QH * D], as the stock store + packed
    trtllm decode over token_indices would (up to rounding)."""
    blocks = block_indices_of(token_indices, ratio=ratio, block_topk=block_topk)
    return pool_union_ws.qsa_pool_union_decode_ws(
        q,
        k_cache,
        v_cache,
        k_new,
        v_new,
        loc,
        req_to_token,
        row_requests,
        blocks,
        sequence_lens,
        query_positions,
        WIDTH,
        scale,
        ratio=ratio,
        write_new_kv=True,
        **union.LAUNCH,
    )


def admits(rows: int, kv_dtype: torch.dtype, q_dtype: torch.dtype) -> bool:
    """The union path's row range and dtypes (FP8 pool, BF16 q), without the verify's own switch."""
    return (
        union.MIN_ROWS <= rows <= union.MAX_ROWS
        and rows % WIDTH == 0
        and kv_dtype == torch.float8_e4m3fn
        and q_dtype == torch.bfloat16
    )
