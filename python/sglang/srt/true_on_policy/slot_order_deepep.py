"""Stands in for the stock DeepEP normal combine, where ep_gather sums each rank's local expert rows into one BF16 row
before DeepEP sums the ranks, so the grouping and its roundings follow expert placement.

Deliver expert slots with transport annotations, preserving their owner-side identity.
"""

from __future__ import annotations

from collections import deque
from typing import NamedTuple

import torch
import torch.distributed as dist
import triton
import triton.language as tl

from sglang.srt.true_on_policy.sampling import register_pre_emit_check

# Source fingerprint: the first 16 16-bit words of the dispatched row (16 BF16 values, or 32 FP8 bytes), bitcast.
FINGERPRINT_COLS = 16
ANNOTATION_COLS = 16
_BF16_BYTES = 2
_INTERNODE_ALIGN_BYTES = 512
_ALIGN_COLS = _INTERNODE_ALIGN_BYTES // _BF16_BYTES


class SlotOrderDeepEPError(RuntimeError):
    pass


# Handle-only slot rows: DeepEP's intranode TMA path needs an even number of 16-byte units per row
# (intranode.cu: hidden_int4 % 2 == 0), so 32 bytes is the smallest row.
HANDLE_ONLY_COLS = 32 // _BF16_BYTES

# EP size -> NVL (send, recv) chunk tokens of the slot dispatch and the slot combine.
# Measured on 2x/8xH200 with 4,096-token prefill chunks;
# EP sizes not listed, and EP across nodes, keep DeepEP's configs.
_SLOT_NVL_CHUNKS = {
    2: ((24, 256), (10, 256)),
    8: ((16, 128), (4, 128)),
}
_NUM_MAX_NVL_PEERS = 8


def slot_transport_configs(group_size: int):
    """DeepEP Configs of the slot dispatch and combine; (None, None) keeps the defaults.

    The slot path moves one row per (token, k) and is bound by per-row work on the
    receiving rank, so it scales with SMs while the primary dispatch does not.
    """
    chunks = _SLOT_NVL_CHUNKS.get(group_size)
    if chunks is None or group_size > _NUM_MAX_NVL_PEERS:
        return None, None
    from deep_ep import Config

    device = torch.cuda.get_device_properties(torch.cuda.current_device())
    # EP2's short slot dispatch and wide slot return run between expert kernels,
    # without compute overlap. On Hopper, use the SMs left idle by DeepEP's
    # generic 20-SM setting. Keep its EP2 chunk sizes and other devices' defaults.
    if group_size == 2 and device.major != 9:
        return None, None
    # EP2 uses 128 SMs on H200; retain EP8's measured half-SM setting (64).
    # Both calls share the slot handle, so they share num_sms.
    num_sms = device.multi_processor_count // 16 * (16 if group_size == 2 else 8)
    (dispatch_send, dispatch_recv), (combine_send, combine_recv) = chunks
    return (
        Config(num_sms, dispatch_send, dispatch_recv, 6, 128),
        Config(num_sms, combine_send, combine_recv, 6, 128),
    )


def padded_width(cols: int) -> int:
    return (cols + _ALIGN_COLS - 1) // _ALIGN_COLS * _ALIGN_COLS


def _rows_contiguous(t: torch.Tensor) -> torch.Tensor:
    """The kernels below address rows by stride(0) and columns densely."""
    return t if t.stride(-1) == 1 else t.contiguous()


def _row_words(t: torch.Tensor) -> torch.Tensor:
    """Rows of a BF16 or FP8 dispatch payload as 16-bit words, so fingerprints compare bytes, never values."""
    if t.element_size() not in (1, 2):
        raise SlotOrderDeepEPError(
            f"[top] slot fingerprints read BF16 or FP8 rows, got {t.dtype}"
        )
    return _rows_contiguous(t).view(torch.int16)


@triton.jit
def _pack_slot_payload_kernel(
    words_ptr,
    words_stride,
    ids_ptr,
    weights_ptr,
    out_ptr,
    num_slots,
    source_rank,
    TOPK: tl.constexpr,
    FP_COLS: tl.constexpr,
    ANN_COLS: tl.constexpr,
    WIDTH: tl.constexpr,
    ROWS: tl.constexpr,
):
    slots = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    live = slots < num_slots
    tokens = slots // TOPK
    experts = tl.maximum(tl.load(ids_ptr + slots, mask=live, other=0), 0)
    wbits = tl.load(weights_ptr + slots, mask=live, other=0.0).to(
        tl.int32, bitcast=True
    )
    cols = tl.arange(0, WIDTH)
    row = slots[:, None]
    c = cols[None, :]
    fp = tl.load(
        words_ptr + tokens[:, None].to(tl.int64) * words_stride + c,
        mask=live[:, None] & (c < FP_COLS),
        other=0,
    )
    a = c - FP_COLS
    tok = tokens[:, None]
    ann = tl.where(a == 0, experts[:, None] // 256, 0)
    ann = tl.where(a == 1, experts[:, None] % 256, ann)
    ann = tl.where(a == 2, tok // 65536, ann)
    ann = tl.where(a == 3, tok // 256 % 256, ann)
    ann = tl.where(a == 4, tok % 256, ann)
    ann = tl.where(a == 5, row % TOPK, ann)
    ann = tl.where(a == 6, source_rank // 256, ann)
    ann = tl.where(a == 7, source_rank % 256, ann)
    for b in tl.static_range(4):
        ann = tl.where(a == 8 + b, (wbits[:, None] >> (8 * b)) & 255, ann)
    ann_value = tl.where(c >= FP_COLS + ANN_COLS, float("nan"), ann.to(tl.float32))
    ann_bits = ann_value.to(tl.bfloat16).to(tl.int16, bitcast=True)
    bits = tl.where(c < FP_COLS, fp, ann_bits)
    tl.store(out_ptr + row.to(tl.int64) * WIDTH + c, bits, mask=live[:, None])


@triton.jit
def _pack_return_kernel(
    src_ptr,
    src_stride,
    output_index_ptr,
    output_index_stride,
    primary_row_ptr,
    primary_k_ptr,
    ann_ptr,
    ann_stride,
    out_ptr,
    HIDDEN: tl.constexpr,
    ANN_COLS: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0).to(tl.int64)
    block = tl.program_id(1)
    r = tl.load(primary_row_ptr + i)
    k = tl.load(primary_k_ptr + i)
    gemm_row = tl.load(output_index_ptr + r * output_index_stride + k).to(tl.int64)
    cols = block * BLOCK + tl.arange(0, BLOCK)
    in_hidden = cols < HIDDEN
    a = cols - HIDDEN
    in_ann = (a >= 0) & (a < ANN_COLS)
    x = tl.load(src_ptr + gemm_row * src_stride + cols, mask=in_hidden, other=0.0)
    y = tl.load(ann_ptr + i * ann_stride + a, mask=in_ann, other=0.0)
    v = tl.where(in_hidden, x, tl.where(in_ann, y, float("nan")))
    tl.store(out_ptr + i * WIDTH + cols, v, mask=cols < WIDTH)


# Receipt failure bits, one per contract clause; messages in _RECEIPT_MESSAGES.
_BAD_SLOT_ID, _ORDER, _WEIGHT, _FINGERPRINT, _PAD_DISPATCH, _ANN_EXPERT, _ANN_WEIGHT = (
    1 << i for i in range(7)
)
_COUNT, _PAD_RETURN, _RET_ORDER, _RET_EXPERT, _RET_WEIGHT = (
    1 << i for i in range(7, 12)
)
_RECEIPT_MESSAGES = {
    _BAD_SLOT_ID: "[top] slot dispatch delivered an invalid expert id",
    _ORDER: "[top] expert-id order differs between the two receives",
    _WEIGHT: "[top] fp32 routing-weight bytes differ between the receives",
    _FINGERPRINT: "[top] source fingerprints differ between the receives",
    _PAD_DISPATCH: "[top] slot dispatch: alignment padding arrived non-NaN: a writer touched bytes the contract declares dead",
    _ANN_EXPERT: "[top] transported global expert ids disagree with the local receive layout",
    _ANN_WEIGHT: "[top] annotation weight bytes disagree with the dispatched fp32 weights",
    _COUNT: "[top] route count mismatch between the primary and slot-level receives",
    _PAD_RETURN: "[top] return combine: alignment padding arrived non-NaN: a writer touched bytes the contract declares dead",
    _RET_ORDER: "[top] returned annotations are not in the slot dispatch's source order",
    _RET_EXPERT: "[top] transported expert ids differ from the router's",
    _RET_WEIGHT: "[top] transported routing-weight bytes differ from the router's",
}


@triton.jit
def _digit(row_ptr, col):
    return tl.floor(tl.load(row_ptr + col).to(tl.float32) + 0.5).to(tl.int32)


@triton.jit
def _ann_weight_bits(row_ptr, base):
    return (
        _digit(row_ptr, base)
        | (_digit(row_ptr, base + 1) << 8)
        | (_digit(row_ptr, base + 2) << 16)
        | (_digit(row_ptr, base + 3) << 24)
    )


@triton.jit
def _or_combine(a, b):
    return a | b


@triton.jit
def _masked_digit(ptrs, col, mask):
    return tl.floor(tl.load(ptrs + col, mask=mask, other=0.0).to(tl.float32) + 0.5).to(
        tl.int32
    )


@triton.jit
def _route_offsets_kernel(
    idx_ptr,
    rows,
    num_slots,
    offsets_ptr,
    flag_ptr,
    TOPK: tl.constexpr,
    TOPK_P2: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Exclusive row offsets of the primary receive's local routes, in row-major order."""
    ks = tl.arange(0, TOPK_P2)
    running = 0
    for start in range(0, rows, BLOCK):
        r = start + tl.arange(0, BLOCK)
        live = r < rows
        ids = tl.load(
            idx_ptr + r[:, None].to(tl.int64) * TOPK + ks[None, :],
            mask=live[:, None] & (ks[None, :] < TOPK),
            other=-1,
        )
        counts = tl.sum((ids >= 0).to(tl.int32), axis=1)
        tl.store(offsets_ptr + r, running + tl.cumsum(counts, 0) - counts, mask=live)
        running += tl.sum(counts, 0)
    if running != num_slots:
        tl.atomic_or(flag_ptr, 1 << 7)


@triton.jit
def _validate_receive_kernel(
    recv_topk_idx_ptr,
    recv_topk_w_ptr,
    offsets_ptr,
    recv_words_ptr,
    recv_words_stride,
    payload_ptr,
    slot_idx_ptr,
    slot_w_ptr,
    expert_offset,
    num_slots,
    primary_row_ptr,
    primary_k_ptr,
    flag_ptr,
    TOPK: tl.constexpr,
    TOPK_P2: tl.constexpr,
    FP_COLS: tl.constexpr,
    ANN_COLS: tl.constexpr,
    WIDTH: tl.constexpr,
):
    """One primary row: place each local route at its slot and check that slot's receipts."""
    r = tl.program_id(0).to(tl.int64)
    ks = tl.arange(0, TOPK_P2)
    in_k = ks < TOPK
    ids = tl.load(recv_topk_idx_ptr + r * TOPK + ks, mask=in_k, other=-1).to(tl.int32)
    valid = in_k & (ids >= 0)
    slot = tl.load(offsets_ptr + r) + tl.cumsum(valid.to(tl.int32), 0) - 1
    ok = valid & (slot < num_slots)
    slot64 = slot.to(tl.int64)
    tl.store(primary_row_ptr + slot64, r, mask=ok)
    tl.store(primary_k_ptr + slot64, ks.to(tl.int64), mask=ok)

    slot_id = tl.load(slot_idx_ptr + slot64, mask=ok, other=0).to(tl.int32)
    w_primary = (
        tl.load(recv_topk_w_ptr + r * TOPK + ks, mask=ok, other=0.0)
        .to(tl.float32)
        .to(tl.int32, bitcast=True)
    )
    w_slot = (
        tl.load(slot_w_ptr + slot64, mask=ok, other=0.0)
        .to(tl.float32)
        .to(tl.int32, bitcast=True)
    )

    cols = tl.arange(0, WIDTH)
    rows = payload_ptr + slot64[:, None] * WIDTH + cols[None, :]
    fp_mask = ok[:, None] & (cols[None, :] < FP_COLS)
    got = tl.load(rows, mask=fp_mask, other=0.0).to(tl.int16, bitcast=True)
    want = tl.load(
        recv_words_ptr + r * recv_words_stride + cols, mask=cols < FP_COLS, other=0
    )
    fp_bad = tl.sum(((got != want[None, :]) & fp_mask).to(tl.int32), axis=1) > 0
    pad = tl.load(
        rows,
        mask=ok[:, None] & (cols[None, :] >= FP_COLS + ANN_COLS),
        other=float("nan"),
    ).to(tl.float32)
    pad_bad = tl.sum((pad == pad).to(tl.int32), axis=1) > 0

    ann = payload_ptr + slot64 * WIDTH + FP_COLS
    ann_expert = _masked_digit(ann, 0, ok) * 256 + _masked_digit(ann, 1, ok)
    ann_w = (
        _masked_digit(ann, 8, ok)
        | (_masked_digit(ann, 9, ok) << 8)
        | (_masked_digit(ann, 10, ok) << 16)
        | (_masked_digit(ann, 11, ok) << 24)
    )

    bad = tl.where(slot_id < 0, 1 << 0, 0)
    bad |= tl.where(ids != slot_id, 1 << 1, 0)
    bad |= tl.where(w_primary != w_slot, 1 << 2, 0)
    bad |= tl.where(fp_bad, 1 << 3, 0)
    bad |= tl.where(pad_bad, 1 << 4, 0)
    bad |= tl.where(ann_expert != slot_id + expert_offset, 1 << 5, 0)
    bad |= tl.where(ann_w != w_slot, 1 << 6, 0)
    bad = tl.reduce(tl.where(ok, bad, 0), 0, _or_combine)
    if bad != 0:
        tl.atomic_or(flag_ptr, bad)


@triton.jit
def _validate_return_kernel(
    returned_ptr,
    router_ids_ptr,
    router_w_ptr,
    flag_ptr,
    topk,
    HIDDEN: tl.constexpr,
    ANN_COLS: tl.constexpr,
    WIDTH: tl.constexpr,
    PAD_BLOCK: tl.constexpr,
):
    s = tl.program_id(0).to(tl.int64)
    expert = tl.load(router_ids_ptr + s).to(tl.int32)
    if expert >= 0:
        row = returned_ptr + s * WIDTH
        ann = row + HIDDEN
        token = _digit(ann, 2) * 65536 + _digit(ann, 3) * 256 + _digit(ann, 4)
        cols = HIDDEN + ANN_COLS + tl.arange(0, PAD_BLOCK)
        pad = tl.load(row + cols, mask=cols < WIDTH, other=float("nan")).to(tl.float32)
        w = tl.load(router_w_ptr + s).to(tl.float32).to(tl.int32, bitcast=True)
        bad = tl.where(tl.sum((pad == pad).to(tl.int32)) > 0, 1 << 8, 0)
        bad |= tl.where((token != s // topk) | (_digit(ann, 5) != s % topk), 1 << 9, 0)
        bad |= tl.where(_digit(ann, 0) * 256 + _digit(ann, 1) != expert, 1 << 10, 0)
        bad |= tl.where(_ann_weight_bits(ann, 8) != w, 1 << 11, 0)
        if bad != 0:
            tl.atomic_or(flag_ptr, bad)


# Receipt bitmasks copied to pinned host memory, read after their GPU work is done.
_PENDING_RECEIPTS: deque = deque()


def defer_receipt_flag(flag: torch.Tensor) -> None:
    host = torch.empty(1, dtype=torch.int32, pin_memory=True)
    host.copy_(flag, non_blocking=True)
    done = torch.cuda.Event()
    done.record()
    _PENDING_RECEIPTS.append((host, done))


def read_pending_receipts(*, block: bool) -> None:
    """Raise on any deferred receipt failure; ``block`` waits for the newest one."""
    while _PENDING_RECEIPTS:
        host, done = _PENDING_RECEIPTS[0]
        if not block and not done.query():
            return
        done.synchronize()
        _PENDING_RECEIPTS.popleft()
        if int(host[0]):
            _PENDING_RECEIPTS.clear()
            _raise_on_receipt_flag(host)


def _no_failed_receipts() -> None:
    """No token leaves a forward whose deferred transport receipts failed."""
    read_pending_receipts(block=True)


register_pre_emit_check(_no_failed_receipts)


def _raise_on_receipt_flag(flag: torch.Tensor) -> None:
    bits = int(flag.item())
    for bit, message in _RECEIPT_MESSAGES.items():
        if bits & bit:
            raise SlotOrderDeepEPError(message)


class SlotDispatchPlan(NamedTuple):
    num_tokens: int
    topk: int
    handle: object
    num_slots: int


class ExpertSlotView(NamedTuple):
    primary_row: torch.Tensor
    primary_k: torch.Tensor
    annotations: torch.Tensor
    flag: torch.Tensor


def raw_dispatch(buffer, payload, topk_idx, topk_weights, num_experts, config=None):
    (
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        num_tokens_per_expert,
        is_token_in_rank,
        event,
    ) = buffer.get_dispatch_layout(topk_idx, num_experts)
    recv_x, recv_topk_idx, recv_topk_weights, recv_counts, handle, _ = buffer.dispatch(
        payload.contiguous(),
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        num_tokens_per_rank=num_tokens_per_rank,
        num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=num_tokens_per_expert,
        previous_event=event,
        config=config,
    )
    return (recv_x, recv_topk_idx, recv_topk_weights, handle)


def pack_slot_payload(hidden, topk_ids, topk_weights, source_rank: int):
    """One row per (token, k) slot: source fingerprint, annotations, NaN padding.

    ``hidden`` is the row the primary dispatch sends (BF16, or TOP's FP8 payload).
    Slot s = token * topk + k. Returns (payload, slot expert ids [S, 1] int64,
    slot FP32 weights [S, 1]).
    """
    words = _row_words(hidden)
    if words.shape[1] < FINGERPRINT_COLS:
        raise SlotOrderDeepEPError(
            f"[top] slot fingerprints need rows of >= {2 * FINGERPRINT_COLS} bytes"
        )
    num_tokens, topk = topk_ids.shape
    if num_tokens >= 256**3 or source_rank >= 256 * 256:
        raise SlotOrderDeepEPError(
            "[top] annotation overflow: expert < 65536, token < 2^24, rank < 65536"
        )
    num_slots = num_tokens * topk
    slot_experts = topk_ids.reshape(num_slots, 1).to(torch.int64).contiguous()
    slot_weights = topk_weights.reshape(num_slots, 1).to(torch.float32).contiguous()
    width = padded_width(FINGERPRINT_COLS + ANNOTATION_COLS)
    payload = torch.empty(
        (num_slots, width), dtype=torch.bfloat16, device=hidden.device
    )
    if num_slots:
        rows = 64
        _pack_slot_payload_kernel[(triton.cdiv(num_slots, rows),)](
            words,
            words.stride(0),
            slot_experts,
            slot_weights,
            payload.view(torch.int16),
            num_slots,
            source_rank,
            TOPK=topk,
            FP_COLS=FINGERPRINT_COLS,
            ANN_COLS=ANNOTATION_COLS,
            WIDTH=width,
            ROWS=rows,
        )
    return payload, slot_experts, slot_weights


def dispatch_slot_level(
    buffer,
    group,
    hidden,
    topk_ids,
    topk_weights,
    num_experts,
    *,
    annotate=True,
    config=None,
):
    """Dispatch one row per (token, k) slot, inactive slots included as unrouted rows.

    Slot s = token * topk + k, so the owner's slot layout has a fixed shape and
    the return needs no host-side compaction. ``annotate=False`` sends handle-only
    rows: the receive and return then carry nothing to validate.
    """
    if num_experts > 256 * 256:
        raise SlotOrderDeepEPError(
            "[top] annotation overflow: expert < 65536, token < 2^24, rank < 65536"
        )
    num_tokens, topk = topk_ids.shape
    if annotate:
        payload, slot_experts, slot_weights = pack_slot_payload(
            hidden, topk_ids, topk_weights, dist.get_rank(group)
        )
    else:
        payload = torch.empty(
            (num_tokens * topk, HANDLE_ONLY_COLS),
            dtype=torch.bfloat16,
            device=hidden.device,
        )
        slot_experts = topk_ids.reshape(-1, 1).to(torch.int64).contiguous()
        slot_weights = topk_weights.reshape(-1, 1).to(torch.float32).contiguous()
    recv_payload, recv_slot_idx, recv_slot_weights, handle = raw_dispatch(
        buffer, payload, slot_experts, slot_weights, num_experts, config=config
    )
    plan = SlotDispatchPlan(num_tokens, topk, handle, num_tokens * topk)
    return (recv_payload, recv_slot_idx, recv_slot_weights, plan)


def validate_expert_receive(
    *,
    recv_x,
    recv_topk_idx,
    recv_topk_weights,
    recv_payload,
    recv_slot_idx,
    recv_slot_weights,
    num_local_experts: int,
    ep_rank: int,
) -> ExpertSlotView:
    """Pair each slot-level row with its (row, k) in the primary receive.

    Both receives keep each source rank's rows in source order, so the slot rows
    follow the row-major order of the primary receive's local routes. The
    receipts prove this pairing; their failure bits are returned unread for
    owner_reduce to read with its own in one host sync.
    """
    num_slots = recv_payload.shape[0]
    recv_words = _row_words(recv_x)
    rows, topk = recv_topk_idx.shape
    device = recv_topk_idx.device
    flag = torch.zeros(1, dtype=torch.int32, device=device)
    primary_row = torch.empty(num_slots, dtype=torch.int64, device=device)
    primary_k = torch.empty(num_slots, dtype=torch.int64, device=device)
    topk_p2 = triton.next_power_of_2(topk)
    if rows:
        offsets = torch.empty(rows, dtype=torch.int32, device=device)
        recv_topk_idx = recv_topk_idx.contiguous()
        _route_offsets_kernel[(1,)](
            recv_topk_idx,
            rows,
            num_slots,
            offsets,
            flag,
            TOPK=topk,
            TOPK_P2=topk_p2,
            BLOCK=1024,
        )
        _validate_receive_kernel[(rows,)](
            recv_topk_idx,
            recv_topk_weights.contiguous(),
            offsets,
            recv_words,
            recv_words.stride(0),
            recv_payload,
            recv_slot_idx.reshape(-1).contiguous(),
            recv_slot_weights.reshape(-1).contiguous(),
            ep_rank * num_local_experts,
            num_slots,
            primary_row,
            primary_k,
            flag,
            TOPK=topk,
            TOPK_P2=topk_p2,
            FP_COLS=FINGERPRINT_COLS,
            ANN_COLS=ANNOTATION_COLS,
            WIDTH=recv_payload.shape[1],
        )
    elif num_slots:
        flag.fill_(_COUNT)
    return ExpertSlotView(
        primary_row=primary_row,
        primary_k=primary_k,
        annotations=recv_payload[
            :, FINGERPRINT_COLS : FINGERPRINT_COLS + ANNOTATION_COLS
        ],
        flag=flag,
    )


def pack_return(expert_out, output_index, view: ExpertSlotView) -> torch.Tensor:
    """Expert output row of every received slot, its annotations, NaN padding."""
    expert_out = _rows_contiguous(expert_out)
    output_index = _rows_contiguous(output_index)
    hidden = expert_out.shape[1]
    width = padded_width(hidden + ANNOTATION_COLS)
    num_slots = view.primary_row.numel()
    out = torch.empty(
        (num_slots, width), dtype=expert_out.dtype, device=expert_out.device
    )
    if num_slots:
        block = 1024
        _pack_return_kernel[(num_slots, triton.cdiv(width, block))](
            expert_out,
            expert_out.stride(0),
            output_index,
            output_index.stride(0),
            view.primary_row,
            view.primary_k,
            view.annotations,
            view.annotations.stride(0),
            out,
            HIDDEN=hidden,
            ANN_COLS=ANNOTATION_COLS,
            WIDTH=width,
            BLOCK=block,
        )
    return out


def validate_return(
    returned, *, hidden_cols: int, router_topk_ids, router_topk_weights, flag
) -> None:
    """OR the return receipts of rows [S, hidden + annotations] into ``flag``: dead padding, source (token, k)
    order, and expert ids and FP32 weight bytes equal to the router's. Slot s must be returned row s.
    """
    num_slots, topk = router_topk_ids.numel(), router_topk_ids.shape[1]
    if not num_slots:
        return
    if returned.stride(1) != 1 or returned.stride(0) != returned.shape[1]:
        returned = returned.contiguous()
    width = returned.shape[1]
    _validate_return_kernel[(num_slots,)](
        returned,
        router_topk_ids.contiguous(),
        router_topk_weights.to(torch.float32).contiguous(),
        flag,
        topk,
        HIDDEN=hidden_cols,
        ANN_COLS=ANNOTATION_COLS,
        WIDTH=width,
        PAD_BLOCK=triton.next_power_of_2(max(width - hidden_cols - ANNOTATION_COLS, 1)),
    )


def owner_reduce(
    returned: torch.Tensor,
    plan: SlotDispatchPlan,
    *,
    hidden_cols: int,
    router_topk_ids: torch.Tensor,
    router_topk_weights: torch.Tensor,
    out_dtype,
    routed_scaling_factor=1.0,
    receipt_flag=None,
    defer_receipts=False,
) -> torch.Tensor:
    from sglang.kernels.ops.moe.ep_moe_kernels import ep_gather

    if returned.shape[0] != plan.num_slots:
        raise SlotOrderDeepEPError(
            f"[top] return transport delivered {returned.shape[0]} rows, expected {plan.num_slots}"
        )
    num_tokens, topk = router_topk_ids.shape
    if (num_tokens, topk) != (plan.num_tokens, plan.topk):
        raise SlotOrderDeepEPError("[top] router shape differs from the slot dispatch")
    device = returned.device
    if returned.stride(1) != 1 or returned.stride(0) != returned.shape[1]:
        returned = returned.contiguous()
    flag = (
        receipt_flag
        if receipt_flag is not None
        else torch.zeros(1, dtype=torch.int32, device=device)
    )
    # Slot s returns to row s, so the gather index is the slot layout itself;
    # unrouted slots come back as zero rows that ep_gather skips by id.
    validate_return(
        returned,
        hidden_cols=hidden_cols,
        router_topk_ids=router_topk_ids,
        router_topk_weights=router_topk_weights,
        flag=flag,
    )
    if defer_receipts:
        defer_receipt_flag(flag)
    else:
        _raise_on_receipt_flag(flag)
    gather_weights = router_topk_weights.to(torch.float32)
    if routed_scaling_factor != 1.0:
        gather_weights = gather_weights * routed_scaling_factor
    index = torch.arange(plan.num_slots, dtype=torch.int32, device=device).view(
        num_tokens, topk
    )
    out = torch.empty(num_tokens, hidden_cols, device=device, dtype=out_dtype)
    ep_gather(returned[:, :hidden_cols], router_topk_ids, gather_weights, index, out)
    return out


class TopSlotOrderCombineInput(NamedTuple):
    hidden_states: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    output_index: torch.Tensor


def build_top_slot_order_deepep_dispatcher(
    moe_runner_config, *, fp8_dispatch: bool, **deepep_kwargs
):
    """``fp8_dispatch``: both modes send the FP8 expert's input bytes, payload and scales, so the expert skips its
    input quantization. Normal (prefill) dispatch sends quantize_expert_input's output; low-latency (decode) dispatch
    runs DeepEP's fused FP8 cast, whose bytes quantize_expert_input reproduces.
    """
    from sglang.srt.layers.moe.token_dispatcher import DispatchOutputChecker
    from sglang.srt.layers.moe.token_dispatcher.deepep import (
        DeepEPBuffer,
        DeepEPDispatcher,
        DeepEPLLCombineInput,
        _Stage,
    )
    from sglang.srt.layers.moe.utils import is_tbo_enabled

    if is_tbo_enabled():
        raise NotImplementedError(
            "[top] slot transport has one in-flight batch per dispatcher"
        )
    scaling = moe_runner_config.routed_scaling_factor
    scaling = 1.0 if scaling is None else float(scaling)
    import math

    if not math.isfinite(scaling) or scaling <= 0:
        raise ValueError("[top] routed scaling must be finite and positive")

    class TopSlotOrderDeepEPDispatcher(DeepEPDispatcher):
        def __init__(self):
            super().__init__(**deepep_kwargs)
            self._top_num_experts = deepep_kwargs["num_experts"]
            self._top_num_local = deepep_kwargs["num_local_experts"]
            self._top_group = deepep_kwargs["group"]
            self._top_dispatch_config, self._top_combine_config = (
                slot_transport_configs(self._top_group.size())
            )
            self._top_plan = self._top_view = None
            self._top_fp8 = fp8_dispatch
            self._top_buffer_sized = False
            self._top_combined = None
            self._top_set_low_latency_fp8()

        def set_quant_config(self, quant_config):
            # The stock setter re-derives the wire dtype from the server args (BF16 under TOP).
            super().set_quant_config(quant_config)
            self._top_set_low_latency_fp8()

        def _top_set_low_latency_fp8(self):
            if self._top_fp8 and self.deepep_mode.enable_low_latency():
                enable_low_latency_fp8(self._low_latency_dispatcher)

        def _top_presize_buffer(self):
            # A wider allocation would invalidate handles from an earlier dispatch.
            impl = self._normal_dispatcher
            DeepEPBuffer.get_deepep_buffer(
                impl.group,
                padded_width(deepep_kwargs["hidden_size"] + ANNOTATION_COLS),
                impl.params_bytes,
                self.deepep_mode,
                impl.num_max_dispatch_tokens_per_rank,
                self._top_num_experts,
                extra_configs=[
                    c
                    for c in (self._top_dispatch_config, self._top_combine_config)
                    if c is not None
                ],
            )
            self._top_buffer_sized = True

        def dispatch_a(self, hidden_states, topk_output, tbo_subbatch_index=None):
            if tbo_subbatch_index is not None:
                raise NotImplementedError(
                    "[top] slot transport has one in-flight batch per dispatcher"
                )
            if not self._top_buffer_sized:
                self._top_presize_buffer()
            payload = hidden_states
            if self._top_fp8 and self._get_impl() is self._normal_dispatcher:
                from sglang.srt.true_on_policy.fp8.fp8_expert_forward import (
                    quantize_expert_input,
                )

                # The quantizer reads each row on its own, so quantizing at the source gives every slot copy
                # the bytes and scales the expert would compute from the received BF16 row.
                payload = quantize_expert_input(hidden_states.contiguous())
            self._top_hidden = payload[0] if isinstance(payload, tuple) else payload
            self._top_router_ids = topk_output.topk_ids
            self._top_router_weights = topk_output.topk_weights
            # DeepEP requires an FP32 container; retain the router's BF16 rounding.
            super().dispatch_a(
                payload,
                topk_output._replace(topk_weights=topk_output.topk_weights.float()),
            )

        def dispatch_b(self, tbo_subbatch_index=None):
            if tbo_subbatch_index is not None:
                raise NotImplementedError(
                    "[top] slot transport has one in-flight batch per dispatcher"
                )
            out = super().dispatch_b()
            # DeepEP's host wait above means earlier layers' receipts are ready.
            read_pending_receipts(block=False)
            is_normal = DispatchOutputChecker.format_is_deepep_normal(out)
            _check_expert_input(out, fp8=self._top_fp8)
            if is_normal:
                if torch.cuda.is_current_stream_capturing():
                    raise NotImplementedError(
                        "[top] normal-mode receipt validation requires eager execution"
                    )
                buffer = self._normal_dispatcher._get_buffer()
                payload, ids, weights, plan = dispatch_slot_level(
                    buffer,
                    self._top_group,
                    self._top_hidden,
                    self._top_router_ids.long(),
                    self._top_router_weights.float(),
                    self._top_num_experts,
                    config=self._top_dispatch_config,
                )
                self._top_view = validate_expert_receive(
                    recv_x=out.hidden_states,
                    recv_topk_idx=out.topk_ids,
                    recv_topk_weights=out.topk_weights,
                    recv_payload=payload,
                    recv_slot_idx=ids,
                    recv_slot_weights=weights,
                    num_local_experts=self._top_num_local,
                    ep_rank=dist.get_rank(self._top_group),
                )
                self._top_plan = plan
            self._top_hidden = None
            return out

        def combine_a(self, combine_input, tbo_subbatch_index=None):
            if tbo_subbatch_index is not None:
                raise NotImplementedError(
                    "[top] slot transport has one in-flight batch per dispatcher"
                )
            if isinstance(combine_input, DeepEPLLCombineInput):
                return super().combine_a(
                    combine_input._replace(
                        topk_weights=(
                            combine_input.topk_weights.float()
                            if scaling == 1.0
                            else combine_input.topk_weights.float() * scaling
                        )
                    )
                )
            if not isinstance(combine_input, TopSlotOrderCombineInput):
                raise SlotOrderDeepEPError(
                    f"[top] unexpected combine input {type(combine_input).__name__}"
                )
            view, plan = self._top_view, self._top_plan
            if view is None or plan is None:
                raise SlotOrderDeepEPError(
                    "[top] combine requires a validated slot dispatch"
                )
            self._update_stage(_Stage.AFTER_DISPATCH_B, _Stage.AFTER_COMBINE_A)
            expert_out = combine_input.hidden_states
            payload = pack_return(expert_out, combine_input.output_index, view)
            returned, _, _ = self._normal_dispatcher._get_buffer().combine(
                payload, handle=plan.handle, config=self._top_combine_config
            )
            self._top_combined = owner_reduce(
                returned,
                plan,
                hidden_cols=expert_out.shape[1],
                router_topk_ids=self._top_router_ids,
                router_topk_weights=self._top_router_weights,
                out_dtype=expert_out.dtype,
                routed_scaling_factor=scaling,
                receipt_flag=view.flag,
                defer_receipts=True,
            )

        def combine_b(self, tbo_subbatch_index=None):
            if tbo_subbatch_index is not None:
                raise NotImplementedError(
                    "[top] slot transport has one in-flight batch per dispatcher"
                )
            if self._top_combined is None:
                return super().combine_b()
            out = self._top_combined
            self._top_combined = None
            self._top_plan = self._top_view = None
            self._normal_dispatcher.handle = None
            self._update_stage(_Stage.AFTER_COMBINE_A, _Stage.INITIAL)
            return out

    return TopSlotOrderDeepEPDispatcher()


def enable_low_latency_fp8(impl) -> None:
    """Send TOP's FP8 expert input through DeepEP's fused low-latency cast (quantize_expert_input's bytes).

    The cast takes power-of-two scales (round_scale) on both architectures, stored as the expert GEMM reads them:
    packed UE8M0 on Blackwell, FP32 on Hopper, where the stock setting would send FP32 amax/448 scales.
    """
    from sglang.srt.true_on_policy.fp8 import uses_hopper_scales

    if impl.fp8_use_ue8m0 == uses_hopper_scales():
        raise NotImplementedError(
            "[top] DeepEP low-latency FP8 scales must use the expert GEMM's scale format"
        )
    impl.use_fp8 = True
    impl.fp8_round_scale = True


def _check_expert_input(out, *, fp8: bool) -> None:
    """The receive carries what this dispatcher sent: TOP's FP8 payload and scales, or bare BF16 rows."""
    if fp8:
        if (
            out.hidden_states.dtype != torch.float8_e4m3fn
            or out.hidden_states_scale is None
        ):
            raise SlotOrderDeepEPError(
                "[top] DeepEP dispatch must deliver TOP's FP8 payload and scales"
            )
    elif (
        out.hidden_states.dtype != torch.bfloat16 or out.hidden_states_scale is not None
    ):
        raise SlotOrderDeepEPError("[top] DeepEP must deliver BF16 expert inputs")


def maybe_slot_order_combine_input(hidden_states, topk_ids, topk_weights, output_index):
    """Owner-side slot-order combine input under TOP; None for the stock gather."""
    from sglang.srt.true_on_policy.moe import should_use_slot_order_moe_combine

    if not should_use_slot_order_moe_combine():
        return None
    return TopSlotOrderCombineInput(hidden_states, topk_ids, topk_weights, output_index)
