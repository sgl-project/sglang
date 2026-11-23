"""Vocabulary-tiled TOP normalization, retaining full logprobs for sampling.

Per-tile FP32 statistics are merged in a fixed pairwise tree over 256-column
tiles. The operation writes every logprob and keeps the input/output dtype;
both engines call it, so it defines the shared numerical baseline.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _pair_sum(values, WIDTH: tl.constexpr):
    tl.static_assert(WIDTH <= 4096 and WIDTH & (WIDTH - 1) == 0)
    for _ in tl.static_range(12):
        if values.shape[0] > 1:
            left, right = tl.split(tl.reshape(values, (values.shape[0] // 2, 2)))
            values = left + right
    return tl.sum(values)


@triton.jit
def _pair_sum_rows(values):
    for _ in tl.static_range(8):
        if values.shape[1] > 1:
            left, right = tl.split(
                tl.reshape(values, (values.shape[0], values.shape[1] // 2, 2))
            )
            values = left + right
    return tl.sum(values, axis=1)


# The row count and row stride are runtime arguments, so a new batch size reuses the compiled kernel;
# rows only bound the masks, never a row's arithmetic.
@triton.jit(do_not_specialize=["M"])
def _tile_stats(
    X,
    MAX,
    SUM,
    V: tl.constexpr,
    TILES: tl.constexpr,
    STRIDE,
    TILE: tl.constexpr,
    M,
    ROWS: tl.constexpr,
):
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    tile = tl.program_id(1)
    cols = tile * TILE + tl.arange(0, TILE)
    mask = (row[:, None] < M) & (cols[None, :] < V)
    x = tl.load(
        X + row[:, None].to(tl.int64) * STRIDE + cols[None, :],
        mask,
        other=-float("inf"),
    ).to(tl.float32)
    maximum = tl.max(x, axis=1)
    # A tile may be entirely masked while another tile contains valid logits.
    terms = tl.where(
        mask & (maximum[:, None] != -float("inf")), tl.exp(x - maximum[:, None]), 0.0
    )
    total = _pair_sum_rows(terms)
    tl.store(MAX + row * TILES + tile, maximum, row < M)
    tl.store(SUM + row * TILES + tile, total, row < M)


@triton.jit
def _merge_stats(MAX, SUM, LSE, TILES: tl.constexpr, WIDTH: tl.constexpr):
    row = tl.program_id(0)
    tiles = tl.arange(0, WIDTH)
    maximum = tl.load(MAX + row * TILES + tiles, tiles < TILES, other=-float("inf"))
    total = tl.load(SUM + row * TILES + tiles, tiles < TILES, other=0.0)
    global_max = tl.max(maximum)
    terms = tl.where(
        (tiles < TILES) & (maximum != -float("inf")),
        total * tl.exp(maximum - global_max),
        0.0,
    )
    lse = global_max + tl.log(_pair_sum(terms, WIDTH))
    tl.store(LSE + row, lse)


@triton.jit(do_not_specialize=["M"])
def _write_logprobs(
    X,
    LSE,
    Y,
    V: tl.constexpr,
    STRIDE,
    BLOCK: tl.constexpr,
    M,
    ROWS: tl.constexpr,
):
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    block = tl.program_id(1)
    cols = block * BLOCK + tl.arange(0, BLOCK)
    mask = (row[:, None] < M) & (cols[None, :] < V)
    x = tl.load(
        X + row[:, None].to(tl.int64) * STRIDE + cols[None, :], mask, other=0.0
    ).to(tl.float32)
    lse = tl.load(LSE + row, row < M, other=0.0)
    tl.store(Y + row[:, None].to(tl.int64) * V + cols[None, :], x - lse[:, None], mask)


def tiled_log_softmax(x: torch.Tensor) -> torch.Tensor:
    """Stands in for the stock batch-invariant _log_softmax_kernel (batch_invariant_ops.py), which accumulates the
    row's exp-sum in the logits' dtype (BF16 for BF16 logits).

    Normalize CUDA rows with a fixed vocabulary tree, independent of row count."""
    if not x.is_cuda or x.ndim != 2 or x.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("TOP tiled log-softmax requires 2D CUDA BF16/FP32 logits")
    rows, vocab = x.shape
    if x.stride(1) != 1 or not 0 < vocab <= 256 * 4096:
        raise ValueError(
            "TOP tiled log-softmax requires unit column stride and vocabulary <= 1048576"
        )
    out = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    if rows == 0:
        return out
    tiles = triton.cdiv(vocab, 256)
    maxima = torch.empty((rows, tiles), device=x.device, dtype=torch.float32)
    sums = torch.empty_like(maxima)
    lse = torch.empty((rows,), device=x.device, dtype=torch.float32)
    # Grouping rows changes launch scheduling only, never a row's arithmetic.
    row_tile = 8 if rows >= 32 else 1
    _tile_stats[(triton.cdiv(rows, row_tile), tiles)](
        x, maxima, sums, vocab, tiles, x.stride(0), 256, rows, row_tile
    )
    _merge_stats[(rows,)](
        maxima, sums, lse, tiles, max(2, triton.next_power_of_2(tiles))
    )
    _write_logprobs[(triton.cdiv(rows, row_tile), triton.cdiv(vocab, 1024))](
        x, lse, out, vocab, x.stride(0), 1024, rows, row_tile
    )
    return out
