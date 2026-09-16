"""Paged QSA scoring without materializing gathered keys or per-head logits."""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _paged_mqa(
    Q,
    K,
    Table,
    Lengths,
    Output,
    Q_ROW: tl.constexpr,
    Q_HEAD: tl.constexpr,
    K_PAGE: tl.constexpr,
    K_TOKEN: tl.constexpr,
    TABLE_ROW: tl.constexpr,
    TABLE_COL: tl.constexpr,
    LENGTH_STRIDE: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    CAPACITY: tl.constexpr,
    WIDTH: tl.constexpr,
    DIVISOR: tl.constexpr,
    BLOCK: tl.constexpr,
    PROGRAMS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    length = tl.load(Lengths + row * LENGTH_STRIDE)
    end = tl.minimum(tl.minimum(length, CAPACITY), WIDTH)
    for tile in range(tl.program_id(1), tl.cdiv(end, BLOCK), PROGRAMS):
        columns = tile * BLOCK + tl.arange(0, BLOCK)
        valid = (columns < length) & (columns < CAPACITY) & (columns < WIDTH)
        pages = tl.load(
            Table + row * TABLE_ROW + (columns // PAGE_SIZE) * TABLE_COL,
            valid,
            other=0,
        )
        # Match the reference's negative-page clamp; padding is masked before K loads.
        pages = tl.maximum(pages, 0).to(tl.int64)
        dims = tl.arange(0, DIM)
        keys = tl.load(
            K
            + pages[:, None] * K_PAGE
            + (columns % PAGE_SIZE)[:, None] * K_TOKEN
            + dims[None, :],
            valid[:, None],
            other=0,
        ).to(tl.float32)
        scores = tl.full((BLOCK,), 0, tl.float32)
        for head in range(HEADS):
            query = tl.load(Q + row * Q_ROW + head * Q_HEAD + dims).to(tl.float32)
            dot = tl.sum(keys * query[None, :], axis=1)
            scores += tl.maximum(dot, 0)
        logits = tl.where(valid, scores / DIVISOR, -float("inf"))
        tl.store(Output + row * WIDTH + columns, logits, columns < WIDTH)


def can_run_mqa_decode(q, k, table, lengths, width) -> bool:
    """Support small decode/verify batches; live page ids must index the cache.

    Limit this initial path to 128 rows so its row-by-program grid stays bounded.
    """
    return (
        q.device.type == "npu"
        and q.ndim == 3
        and q.shape[0] <= 128
        and k.ndim == 4
        and table.ndim == 2
        and lengths.ndim == 1
        and q.dtype in (torch.float32, torch.float16, torch.bfloat16)
        and k.dtype == q.dtype
        and q.device == k.device == table.device == lengths.device
        and 0 < q.shape[1] <= 8
        and q.shape[2] in (64, 128, 256)
        and k.shape[2:] == (1, q.shape[2])
        and k.shape[0] > 0
        and k.shape[1] > 0
        and table.shape[0] == lengths.shape[0] == q.shape[0]
        and table.dtype in (torch.int32, torch.int64)
        and lengths.dtype in (torch.int32, torch.int64)
        and q.stride(-1) == k.stride(-1) == 1
        and isinstance(width, int)
        and width >= 0
    )


def mqa_decode(q, k, table, lengths, width, score_scale=None):
    """Return FP32 sum-head ReLU(QK) scores with -inf outside visible columns."""
    if not can_run_mqa_decode(q, k, table, lengths, width):
        raise ValueError("Unsupported NPU paged MQA tensor configuration")
    # Top-k reads all columns, including those outside the live page range.
    logits = torch.full(
        (q.shape[0], width), -float("inf"), device=q.device, dtype=torch.float32
    )
    if not q.shape[0] or not width:
        return logits
    # Bound on-chip storage for FP32 keys and reduction intermediates at dim=256.
    block = 32
    programs = min(32, triton.cdiv(width, block))
    _paged_mqa[(q.shape[0], programs)](
        q,
        k,
        table,
        lengths,
        logits,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        table.stride(0),
        table.stride(1),
        lengths.stride(0),
        q.shape[1],
        q.shape[2],
        k.shape[1],
        table.shape[1] * k.shape[1],
        width,
        score_scale or math.sqrt(q.shape[2]),
        block,
        programs,
        enable_fp_fusion=False,
    )
    return logits
