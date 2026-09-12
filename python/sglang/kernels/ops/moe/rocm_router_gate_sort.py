"""The top-k gate of :mod:`rocm_router_gate` followed by aiter's MoE sorting in one launch, bitwise
the two launches'.

One program per row runs the gate; program 0 sorts every ``(token, slot)`` entry once the other
rows have published theirs through device-scope atomic ``int64`` slots (coherent across XCDs without
an L2 writeback), trapping the wave if a row never publishes rather than sorting zeros for it."""

from __future__ import annotations

import threading
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.moe.rocm_router_gate import (
    _GATE_NUM_EXPERTS,
    _MAX_TOPK,
    _gate_row,
)

# above this many rows the two-launch front (per-row gate, chunked sort) is faster
ROCM_GATE_SORT_MAX_TOKENS = 2
_HANDOFF_ROWS = 64
_HANDOFF_SLOTS = 16  # >= next power of two of any TOPK


@triton.jit
def _chunk(x2, cid, c):
    # row c of a [NC, JC] tensor as a [JC] vector (tiny one-hot reduction, stays in registers)
    return tl.sum(tl.where(cid[:, None] == c, x2, 0), axis=0)


@triton.jit
def _sort_entries(
    e,  # [N] int32 global expert ids
    w,  # [N] fp32 weights
    t,  # [N] int32 token of the entry
    k,  # [N] int32 slot of the entry
    valid,  # [N] bool
    local_expert_ids_ptr,
    sorted_ids_ptr,
    sorted_weights_ptr,
    sorted_expert_ids_ptr,
    num_valid_ids_ptr,
    num_tokens,
    TOPK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    NB_PER_E: tl.constexpr,
    JC: tl.constexpr,  # entries compared per chunk: [N, JC] tensors stay in registers
):
    """aiter's moe_sorting of ``N`` token-major (token, slot) entries by pairwise comparison:
    three passes over ``N // JC`` chunks."""
    N: tl.constexpr = e.shape[0]
    NC: tl.constexpr = N // JC
    i = tl.arange(0, N)
    cid = tl.arange(0, NC)
    loc = tl.load(local_expert_ids_ptr + e, mask=valid, other=-1)
    loc = tl.where(valid, loc, -1)  # -1 = not local
    is_local = loc >= 0
    loc2 = tl.reshape(loc, (NC, JC))
    t2 = tl.reshape(t, (NC, JC))
    k2 = tl.reshape(k, (NC, JC))
    # Opus keeps one entry per (token, expert): a later slot with the same expert wins
    dup = tl.zeros([N], dtype=tl.int32)
    for c in tl.static_range(NC):
        loc_j = _chunk(loc2, cid, c)
        t_j = _chunk(t2, cid, c)
        k_j = _chunk(k2, cid, c)
        same = (loc[:, None] == loc_j[None, :]) & is_local[:, None]
        dup += tl.sum(
            (same & (t_j[None, :] == t[:, None]) & (k_j[None, :] > k[:, None])).to(
                tl.int32
            ),
            axis=1,
        )
    keep = is_local & (dup == 0)
    keep2 = tl.reshape(keep.to(tl.int32), (NC, JC))
    # the first kept entry of an expert stands for the expert: its count, blocks and padding
    earlier = tl.zeros([N], dtype=tl.int32)
    cnt = tl.zeros([N], dtype=tl.int32)
    rank = tl.zeros([N], dtype=tl.int32)
    for c in tl.static_range(NC):
        jj = c * JC + tl.arange(0, JC)
        loc_j = _chunk(loc2, cid, c)
        t_j = _chunk(t2, cid, c)
        keep_j = _chunk(keep2, cid, c)
        same_k = (
            (loc[:, None] == loc_j[None, :])
            & is_local[:, None]
            & (keep_j[None, :] != 0)
        )
        earlier += tl.sum((same_k & (jj[None, :] < i[:, None])).to(tl.int32), axis=1)
        cnt += tl.sum(same_k.to(tl.int32), axis=1)
        # Tokens ascend inside an expert: rank = earlier tokens on the same expert.
        rank += tl.sum((same_k & (t_j[None, :] < t[:, None])).to(tl.int32), axis=1)
    first = keep & (earlier == 0)
    padded = ((cnt + BLOCK_M - 1) // BLOCK_M) * BLOCK_M
    first2 = tl.reshape(first.to(tl.int32), (NC, JC))
    padded2 = tl.reshape(padded, (NC, JC))
    # experts ascend: an expert starts after the padded blocks of every lower present expert
    start = tl.zeros([N], dtype=tl.int32)
    for c in tl.static_range(NC):
        loc_j = _chunk(loc2, cid, c)
        first_j = _chunk(first2, cid, c)
        padded_j = _chunk(padded2, cid, c)
        start += tl.sum(
            tl.where(
                (first_j[None, :] != 0) & (loc_j[None, :] < loc[:, None]),
                padded_j[None, :],
                0,
            ),
            axis=1,
        )
    pos = start + rank
    tl.store(sorted_ids_ptr + pos, (k << 24) | t, mask=keep)
    tl.store(sorted_weights_ptr + pos, w, mask=keep)
    # Block padding after each expert's entries.
    r = tl.arange(0, BLOCK_M)
    pad_pos = start[:, None] + cnt[:, None] + r[None, :]
    pad_mask = first[:, None] & (r[None, :] < (padded - cnt)[:, None])
    tl.store(
        sorted_ids_ptr + pad_pos,
        tl.full(pad_pos.shape, (TOPK << 24) | num_tokens, tl.int32),
        mask=pad_mask,
    )
    tl.store(
        sorted_weights_ptr + pad_pos, tl.zeros(pad_pos.shape, tl.float32), mask=pad_mask
    )
    # One local expert id per block.
    b = tl.arange(0, NB_PER_E)
    blk = start[:, None] // BLOCK_M + b[None, :]
    blk_mask = first[:, None] & (b[None, :] < (padded // BLOCK_M)[:, None])
    tl.store(
        sorted_expert_ids_ptr + blk,
        tl.broadcast_to(loc[:, None], blk.shape),
        mask=blk_mask,
    )
    tl.store(num_valid_ids_ptr, tl.sum(tl.where(first, padded, 0), axis=0))
    tl.store(num_valid_ids_ptr + 1, num_tokens)


@triton.jit
def _router_gate_sort_kernel(
    logits_ptr,  # [M, 384] read when SPLIT_K == 0, written when WRITE_LOGITS
    part_ptr,  # [SPLIT_K, M, 384] fp32 (SPLIT_K > 0)
    bias_ptr,  # [384] fp32 or bf16 (HAS_BIAS)
    weights_ptr,  # [M, TOPK] fp32
    ids_ptr,  # [M, TOPK] int32
    local_expert_ids_ptr,  # [384] int32: local index, -1 when not local
    num_token_non_padded_ptr,  # [1] int32 (HAS_PAD_COUNT)
    sorted_ids_ptr,  # [max_padded] int32
    sorted_weights_ptr,  # [max_padded] fp32
    sorted_expert_ids_ptr,  # [max_blocks] int32
    num_valid_ids_ptr,  # [2] int32
    moe_buf_ptr,  # [M, model_dim] (zeroed when ZERO_MOE_BUF)
    handoff_ptr,  # [HANDOFF_ROWS, HANDOFF_SLOTS] int64: (weight bits << 32) | (id << 1) | valid
    num_tokens,
    model_dim,
    stride_lm,
    stride_ps,
    stride_pm,
    stride_om,
    routed_scaling_factor,
    SPLIT_K: tl.constexpr,
    WRITE_LOGITS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    RENORM: tl.constexpr,
    TOPK: tl.constexpr,
    TOPK_PAD: tl.constexpr,
    M_PAD: tl.constexpr,
    BLOCK_M: tl.constexpr,  # aiter block_m: rows per GEMM tile
    NB_PER_E: tl.constexpr,  # >= max blocks one expert can fill
    HANDOFF_SLOTS: tl.constexpr,
    HAS_PAD_COUNT: tl.constexpr,
    ZERO_MOE_BUF: tl.constexpr,
    SINGLE_ROW: tl.constexpr,  # M == 1: program 0 sorts its own lanes, no hand-off
    BLOCK_D: tl.constexpr,
    MAX_SPINS: tl.constexpr,
    JC: tl.constexpr,  # entries per comparison chunk of the sort
):
    pid = tl.program_id(0)
    lane = tl.arange(0, 64)
    if ZERO_MOE_BUF:
        offs_d = tl.arange(0, BLOCK_D)
        for d0 in range(0, model_dim, BLOCK_D):
            tl.store(
                moe_buf_ptr + pid * model_dim + d0 + offs_d,
                tl.zeros((BLOCK_D,), dtype=moe_buf_ptr.dtype.element_ty),
                mask=d0 + offs_d < model_dim,
            )
    w_lane, e_lane = _gate_row(
        logits_ptr,
        part_ptr,
        bias_ptr,
        pid,
        stride_lm,
        stride_ps,
        stride_pm,
        routed_scaling_factor,
        SPLIT_K,
        WRITE_LOGITS,
        HAS_BIAS,
        RENORM,
        TOPK,
    )
    if HAS_PAD_COUNT:
        # rows at and past the count are padding: expert 0, weight 0, in the outputs and in the sort
        if pid >= tl.load(num_token_non_padded_ptr):
            w_lane = tl.zeros([64], dtype=tl.float32)
            e_lane = tl.zeros([64], dtype=tl.int32)
    out_mask = lane < TOPK
    tl.store(weights_ptr + pid * stride_om + lane, w_lane, mask=out_mask)
    tl.store(ids_ptr + pid * stride_om + lane, e_lane, mask=out_mask)
    if SINGLE_ROW:
        kk = tl.arange(0, TOPK_PAD)
        e_row = tl.gather(e_lane, kk, 0)
        w_row = tl.gather(w_lane, kk, 0)
        _sort_entries(
            e_row,
            w_row,
            tl.zeros([TOPK_PAD], dtype=tl.int32),
            kk,
            kk < TOPK,
            local_expert_ids_ptr,
            sorted_ids_ptr,
            sorted_weights_ptr,
            sorted_expert_ids_ptr,
            num_valid_ids_ptr,
            num_tokens,
            TOPK,
            BLOCK_M,
            NB_PER_E,
            TOPK_PAD,
        )
        return
    # the slot is zero (cleared by the previous launch), so adding is publishing
    word = (w_lane.to(tl.int32, bitcast=True).to(tl.int64) << 32) | (
        (e_lane.to(tl.int64) << 1) | 1
    )
    tl.atomic_add(
        handoff_ptr + pid * HANDOFF_SLOTS + lane,
        word,
        mask=out_mask,
        sem="relaxed",
        scope="gpu",
    )
    if pid == 0:
        t2 = tl.arange(0, M_PAD)
        k2 = tl.arange(0, TOPK_PAD)
        in_range = (t2[:, None] < num_tokens) & (k2[None, :] < TOPK)
        slot = handoff_ptr + t2[:, None] * HANDOFF_SLOTS + k2[None, :]
        zero = tl.zeros([M_PAD, TOPK_PAD], dtype=tl.int64)
        got = tl.atomic_add(slot, zero, mask=in_range, sem="relaxed", scope="gpu")
        ready = tl.min(
            tl.min(tl.where(in_range, (got & 1).to(tl.int32), 1), axis=1), axis=0
        )
        spins = 0
        while ready == 0 and spins < MAX_SPINS:
            got = tl.atomic_add(slot, zero, mask=in_range, sem="relaxed", scope="gpu")
            ready = tl.min(
                tl.min(tl.where(in_range, (got & 1).to(tl.int32), 1), axis=1), axis=0
            )
            spins += 1
        if ready == 0:
            # a row that never published is a broken hand-off: trap rather than sort zeros for it
            # (tl.device_assert is compiled out without TRITON_DEBUG, so it cannot be the guard)
            tl.inline_asm_elementwise(
                "s_trap 2", "=v,v", [ready], dtype=tl.int32, is_pure=False, pack=1
            )
        # clear the slots for the next launch (this program is the only reader)
        tl.atomic_add(slot, -got, mask=in_range, sem="relaxed", scope="gpu")
        N: tl.constexpr = M_PAD * TOPK_PAD
        i = tl.arange(0, N)
        flat = tl.reshape(got, (N,))
        valid = tl.reshape(in_range, (N,))
        e = tl.where(valid, ((flat >> 1) & 0x7FFFFFFF).to(tl.int32), 0)
        w = tl.where(valid, (flat >> 32).to(tl.int32).to(tl.float32, bitcast=True), 0.0)
        _sort_entries(
            e,
            w,
            i // TOPK_PAD,
            i % TOPK_PAD,
            valid,
            local_expert_ids_ptr,
            sorted_ids_ptr,
            sorted_weights_ptr,
            sorted_expert_ids_ptr,
            num_valid_ids_ptr,
            num_tokens,
            TOPK,
            BLOCK_M,
            NB_PER_E,
            JC,
        )


# one hand-off buffer per device: a launch leaves it zeroed for the next
_handoff_state: dict = {}
_handoff_lock = threading.Lock()


def _handoff_buffer(device: torch.device) -> torch.Tensor:
    """The per-device hand-off slots. A launch leaves them zeroed, so consecutive launches on
    one stream share them; concurrent launches on two streams of one device would race."""
    key = (device.type, device.index)
    buf = _handoff_state.get(key)
    if buf is None:
        with _handoff_lock:
            buf = _handoff_state.get(key)
            if buf is None:
                assert not torch.cuda.is_current_stream_capturing(), (
                    "rocm_router_gate_sort: the hand-off buffer must be allocated by an "
                    "eager launch, or its zero fill is captured instead of executed"
                )
                buf = torch.zeros(
                    _HANDOFF_ROWS * _HANDOFF_SLOTS, dtype=torch.int64, device=device
                )
                _handoff_state[key] = buf
    return buf


def rocm_router_gate_sort(
    gating_output: torch.Tensor,
    correction_bias: Optional[torch.Tensor],
    topk: int,
    renormalize: bool,
    routed_scaling_factor: Optional[float],
    partials: Optional[torch.Tensor],
    local_expert_ids: torch.Tensor,
    num_experts: int,
    model_dim: int,
    moe_buf_dtype: torch.dtype,
    block_size: int,
    zero_moe_buf: bool,
    num_token_non_padded: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, ...]:
    """:func:`rocm_router_gate` followed by :func:`fused_aiter_moe_sorting` in one launch:
    ``(weights, ids, sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf)``."""
    M, n_experts = gating_output.shape
    assert n_experts == _GATE_NUM_EXPERTS == num_experts and 0 < topk <= _MAX_TOPK
    assert 0 < M <= ROCM_GATE_SORT_MAX_TOKENS, (
        f"{M} rows: the fused gate + sort serves at most {ROCM_GATE_SORT_MAX_TOKENS}"
    )
    assert gating_output.stride(1) == 1
    if partials is not None:
        assert partials.shape[1:] == (M, n_experts) and partials.dtype == torch.float32
        assert gating_output.dtype == torch.float32
        split_k = partials.shape[0]
        stride_ps, stride_pm = partials.stride(0), partials.stride(1)
    else:
        partials = gating_output
        split_k = 0
        stride_ps = stride_pm = 0
    if correction_bias is not None:
        assert correction_bias.shape == (n_experts,) and correction_bias.is_contiguous()
    if block_size & (block_size - 1) or block_size <= 0:
        raise ValueError(f"block_size must be a power of two, got {block_size}")
    if local_expert_ids.dtype != torch.int32 or local_expert_ids.numel() != num_experts:
        raise TypeError("local_expert_ids must be int32 with one entry per expert")
    device = gating_output.device
    weights = torch.empty((M, topk), dtype=torch.float32, device=device)
    ids = torch.empty((M, topk), dtype=torch.int32, device=device)
    max_num_tokens_padded = M * topk + num_experts * block_size - topk
    max_num_m_blocks = (max_num_tokens_padded + block_size - 1) // block_size
    sorted_ids = torch.empty(max_num_tokens_padded, dtype=torch.int32, device=device)
    sorted_weights = torch.empty(
        max_num_tokens_padded, dtype=torch.float32, device=device
    )
    sorted_expert_ids = torch.empty(max_num_m_blocks, dtype=torch.int32, device=device)
    num_valid_ids = torch.empty(2, dtype=torch.int32, device=device)
    if zero_moe_buf:
        moe_buf = torch.empty((M, model_dim), dtype=moe_buf_dtype, device=device)
    else:
        moe_buf = torch.empty((0, 0), dtype=moe_buf_dtype, device=device)
    m_pad = max(2, triton.next_power_of_2(M))
    assert m_pad <= _HANDOFF_ROWS, (m_pad, _HANDOFF_ROWS)
    topk_pad = triton.next_power_of_2(topk)
    n = m_pad * topk_pad
    jc = n if n <= 32 else 16
    _router_gate_sort_kernel[(M,)](
        gating_output,
        partials,
        correction_bias if correction_bias is not None else gating_output,
        weights,
        ids,
        local_expert_ids,
        num_token_non_padded if num_token_non_padded is not None else num_valid_ids,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        moe_buf,
        _handoff_buffer(device),
        M,
        model_dim,
        gating_output.stride(0),
        stride_ps,
        stride_pm,
        topk,
        float(1.0 if routed_scaling_factor is None else routed_scaling_factor),
        SPLIT_K=split_k,
        WRITE_LOGITS=split_k > 0,
        HAS_BIAS=correction_bias is not None,
        RENORM=bool(renormalize),
        TOPK=topk,
        TOPK_PAD=topk_pad,
        M_PAD=m_pad,
        BLOCK_M=block_size,
        NB_PER_E=max(2, triton.next_power_of_2((M + block_size - 1) // block_size)),
        HANDOFF_SLOTS=_HANDOFF_SLOTS,
        HAS_PAD_COUNT=num_token_non_padded is not None,
        ZERO_MOE_BUF=zero_moe_buf,
        SINGLE_ROW=M == 1,
        BLOCK_D=1024,
        MAX_SPINS=1 << 20,
        JC=jc,
        num_warps=1,
    )
    return (
        weights,
        ids,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        moe_buf,
    )
