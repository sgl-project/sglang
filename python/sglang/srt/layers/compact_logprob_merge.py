"""Bounded CUDA merge for the compact TP input-logprob exchange."""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _pack_state(
    logits,
    maxima,
    logs,
    top_values,
    top_ids,
    lookup_rows,
    lookup_ids,
    packed,
    ROWS: tl.constexpr,
    SCORES: tl.constexpr,
    K: tl.constexpr,
    LOCAL_K: tl.constexpr,
    START: tl.constexpr,
    END: tl.constexpr,
    VOCAB: tl.constexpr,
    LOGIT_ROW: tl.constexpr,
    LOGIT_COL: tl.constexpr,
    LOOKUP_ROW_STRIDE: tl.constexpr,
    LOOKUP_ID_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    role = tl.program_id(1)
    if role == 0:
        m = tl.load(maxima + x, x < ROWS, other=0.0)
        s = tl.load(logs + x, x < ROWS, other=0.0)
        tl.store(packed + x, m, x < ROWS)
        tl.store(packed + ROWS + x, s, x < ROWS)
    elif role == 1:
        score_row = tl.load(lookup_rows + x * LOOKUP_ROW_STRIDE, x < SCORES, other=0)
        score_token = tl.load(lookup_ids + x * LOOKUP_ID_STRIDE, x < SCORES, other=0)
        owned = (score_token >= START) & (score_token < END)
        score_value = tl.load(
            logits + score_row * LOGIT_ROW + (score_token - START) * LOGIT_COL,
            (x < SCORES) & owned,
            other=0.0,
        ).to(tl.float32)
        tl.store(packed + 2 * ROWS + x, score_value, x < SCORES)
    else:
        row, slot = x // K, x % K
        if LOCAL_K > 0:
            present = (x < ROWS * K) & (slot < LOCAL_K)
            value = tl.load(
                top_values + row * LOCAL_K + slot, present, other=-float("inf")
            ).to(tl.float32)
            token = tl.load(top_ids + row * LOCAL_K + slot, present, other=0)
            token = tl.where(present, token + START, VOCAB).to(tl.float32)
        else:
            value = tl.full((BLOCK,), -float("inf"), tl.float32)
            token = tl.full((BLOCK,), VOCAB, tl.float32)
        base = 2 * ROWS + SCORES
        tl.store(packed + base + x, value, x < ROWS * K)
        tl.store(packed + base + ROWS * K + x, token, x < ROWS * K)


def pack_compact_input_state(
    logits,
    maxima,
    logs,
    top_values,
    top_ids,
    lookup_rows,
    lookup_ids,
    start,
    end,
    vocab,
    k,
):
    """Write the exact FP32 wire layout without intermediate casts/index tensors."""
    rows, scores = logits.shape[0], lookup_ids.numel()
    local_k = min(k, end - start)
    assert rows > 0 and k > 0 and vocab <= 2**24
    assert 2 * rows + scores + 2 * rows * k <= 2**31 - 1
    if local_k:
        assert top_values.is_contiguous() and top_ids.is_contiguous()
    packed = torch.empty(
        2 * rows + scores + 2 * rows * k, device=logits.device, dtype=torch.float32
    )
    _pack_state[(triton.cdiv(max(rows, scores, rows * k), 256), 3)](
        logits,
        maxima,
        logs,
        top_values if local_k else maxima,
        top_ids if local_k else maxima,
        lookup_rows,
        lookup_ids,
        packed,
        rows,
        scores,
        k,
        local_k,
        start,
        end,
        vocab,
        logits.stride(0),
        logits.stride(1),
        lookup_rows.stride(0),
        lookup_ids.stride(0),
        256,
    )
    return packed


@triton.jit
def _merge_rows(
    packed,
    out_max,
    out_log_sum,
    out_values,
    out_ids,
    ROWS: tl.constexpr,
    SCORES: tl.constexpr,
    K: tl.constexpr,
    TP: tl.constexpr,
    STRIDE: tl.constexpr,
    TP_BLOCK: tl.constexpr,
    CANDIDATES: tl.constexpr,
    NORMALIZE: tl.constexpr,
):
    row = tl.program_id(0)
    rank = tl.arange(0, TP_BLOCK)
    maxima = tl.load(packed + rank * STRIDE + row, rank < TP, other=-float("inf"))
    logs = tl.load(packed + rank * STRIDE + ROWS + row, rank < TP, other=0.0)
    maximum = tl.max(maxima, 0)
    maximum = tl.where(
        tl.sum((maxima != maxima).to(tl.int32), 0) > 0, float("nan"), maximum
    )
    mass = tl.where(rank < TP, libdevice.exp((maxima - maximum) + logs), 0.0)
    log_sum = libdevice.log(tl.sum(mass, 0))
    tl.store(out_max + row, maximum)
    tl.store(out_log_sum + row, log_sum)

    lane = tl.arange(0, CANDIDATES)
    owner = lane // K
    slot = lane % K
    base = owner * STRIDE + 2 * ROWS + SCORES + row * K + slot
    values = tl.load(packed + base, lane < TP * K, other=-float("inf"))
    ids = tl.load(packed + base + ROWS * K, lane < TP * K, other=0).to(tl.uint32)
    ids = tl.where(lane < TP * K, ids, 0xFFFFFFFF).to(tl.uint32)
    # Ordered IEEE bits plus reversed IDs give descending score, ascending ID.
    # Canonicalize signed zero (equal scores) and NaNs (Torch sorts them first).
    bits = values.to(tl.uint32, bitcast=True)
    bits = tl.where(values == 0.0, 0, bits).to(tl.uint32)
    bits = tl.where(values != values, 0x7FC00000, bits).to(tl.uint32)
    ordered = tl.where((bits & 0x80000000) != 0, ~bits, bits ^ 0x80000000).to(tl.uint32)
    keys = (ordered.to(tl.uint64) << 32) | (0xFFFFFFFF - ids).to(tl.uint64)
    keys = tl.sort(keys, descending=True)
    sorted_ordered = (keys >> 32).to(tl.uint32)
    sorted_bits = tl.where(
        (sorted_ordered & 0x80000000) != 0,
        sorted_ordered ^ 0x80000000,
        ~sorted_ordered,
    ).to(tl.uint32)
    sorted_values = sorted_bits.to(tl.float32, bitcast=True)
    if NORMALIZE:
        sorted_values = (sorted_values - maximum) - log_sum
    sorted_ids = (0xFFFFFFFF - (keys & 0xFFFFFFFF).to(tl.uint32)).to(tl.int64)
    tl.store(out_values + row * K + lane, sorted_values, lane < K)
    tl.store(out_ids + row * K + lane, sorted_ids, lane < K)


@triton.jit
def _merge_scores(
    packed,
    output,
    lookup_rows,
    maxima,
    log_sums,
    ROWS: tl.constexpr,
    SCORES: tl.constexpr,
    STRIDE: tl.constexpr,
    TP: tl.constexpr,
    TP_BLOCK: tl.constexpr,
    BLOCK: tl.constexpr,
    NORMALIZE: tl.constexpr,
    LOOKUP_STRIDE: tl.constexpr,
):
    score = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    rank = tl.arange(0, TP_BLOCK)
    values = tl.load(
        packed + rank[:, None] * STRIDE + 2 * ROWS + score[None, :],
        (rank[:, None] < TP) & (score[None, :] < SCORES),
        other=0.0,
    )
    selected = tl.sum(values, axis=0)
    if NORMALIZE:
        row = tl.load(lookup_rows + score * LOOKUP_STRIDE, score < SCORES, other=0)
        maximum = tl.load(maxima + row, score < SCORES, other=0.0)
        log_sum = tl.load(log_sums + row, score < SCORES, other=0.0)
        selected = (selected - maximum) - log_sum
    tl.store(output + score, selected, score < SCORES)


def merge_compact_input_state(gathered, rows, scores, k, lookup_rows=None):
    """Merge the FP32 exchange; optional lookup rows request normalized scores/Top-N."""
    tp, stride = gathered.shape
    assert gathered.is_cuda and gathered.dtype == torch.float32
    assert gathered.is_contiguous() and rows > 0 and 0 < tp <= 32 and 0 < tp * k <= 1024
    assert gathered.numel() <= 2**31 - 1
    assert stride == 2 * rows + scores + 2 * rows * k
    if lookup_rows is not None:
        assert lookup_rows.ndim == 1 and lookup_rows.numel() == scores
        assert lookup_rows.dtype in (torch.int32, torch.int64)
    maximum = torch.empty(rows, device=gathered.device, dtype=torch.float32)
    log_sum = torch.empty_like(maximum)
    values = torch.empty((rows, k), device=gathered.device, dtype=torch.float32)
    ids = torch.empty((rows, k), device=gathered.device, dtype=torch.int64)
    selected = torch.empty(scores, device=gathered.device, dtype=torch.float32)
    _merge_rows[(rows,)](
        gathered,
        maximum,
        log_sum,
        values,
        ids,
        rows,
        scores,
        k,
        tp,
        stride,
        triton.next_power_of_2(tp),
        triton.next_power_of_2(tp * k),
        lookup_rows is not None,
        enable_fp_fusion=False,
    )
    if scores:
        _merge_scores[(triton.cdiv(scores, 128),)](
            gathered,
            selected,
            lookup_rows if lookup_rows is not None else selected,
            maximum,
            log_sum,
            rows,
            scores,
            stride,
            tp,
            triton.next_power_of_2(tp),
            128,
            lookup_rows is not None,
            lookup_rows.stride(0) if lookup_rows is not None else 0,
            enable_fp_fusion=False,
        )
    return maximum, log_sum, selected, values, ids
