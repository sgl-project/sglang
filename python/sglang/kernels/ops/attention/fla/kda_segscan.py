# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# K3.1 D-SEG: segment-parallel form of chunk_gated_delta_rule_fwd_h
# for long single-sequence prefill chunks. Default off (SGLANG_K31_KDA_SEGSCAN=0).
#
# Each [V, K] state row of a head evolves chunk by chunk as x <- x A_c + b_c, so
# over a chunk-aligned segment p: X_end = X_start @ M_p + C_p.
#   Phase A: the served kernel on P pseudo-sequences, twice, on two streams:
#            C pass (real u, zero start) and M pass (u = 0, identity start).
#   Phase B: FP32 scan X_{p+1} = X_p @ M_p + C_p (one Triton launch).
#   Phase C: the served kernel from X_p on every segment; it writes h / v_new in
#            the served layout (segments are chunk-aligned, so chunk offsets are
#            the global ones), and the last segment's end state goes to the pool.
# Segment 0 is bitwise equal to the stock call; later segments change the FP32
# summation order, so this path changes numerics (check accuracy before enabling
# it). The segment-parallel method and its scan kernel are based on earlier NVIDIA
# work, re-expressed on SGLang's FLA kernel.
#
# Scope: one sequence (cu_seqlens of 2 entries, B == 1), K == V == 128, per-channel
# exp2 gates, a state pool with (H, V, K)-contiguous slots, eager mode. Anything else
# (and CUDA-graph capture) runs the served call unchanged. No host sync: the plan
# uses only host-known shapes (chunk_indices length), the segment bounds are clamped
# on the device to cu_seqlens[1], and the -1 state-index sentinel is honoured on the
# device (zero start, no state store), as in the served kernel.
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.fla.chunk_delta_h import (
    CHUNK_SIZE,
    chunk_gated_delta_rule_fwd_h,
)
from sglang.srt.environ import envs

# Number of segments (0 / 1 = off) and the shortest chunk (tokens) that uses D-SEG.
KDA_SEGSCAN_P = envs.SGLANG_K31_KDA_SEGSCAN.get()
KDA_SEGSCAN_MIN_T = envs.SGLANG_K31_KDA_SEGSCAN_MIN_T.get()
# Fused setup glue (default): one prep launch (pseudo cu_seqlens / chunk offsets,
# C = 0, M = I), the segment-0 start state loaded from the pool slot inside the scan,
# one store launch for the final state; bitwise equal to the torch glue
# (SGLANG_K31_KDA_SEGSCAN_TORCH_GLUE=1), which stays for non-FP32 pools and as the
# A/B path.
KDA_SEGSCAN_GLUE = 0 if envs.SGLANG_K31_KDA_SEGSCAN_TORCH_GLUE.get() else 1

_SCAN_BV = 16
_SCAN_BK = 32


@triton.jit
def _dseg_scan_kernel(
    x0,
    m,
    c,
    x,
    P,
    pool,
    pool_stride0,
    idx,
    H: tl.constexpr,
    D: tl.constexpr,
    BV: tl.constexpr,
    BK: tl.constexpr,
    FROM_POOL: tl.constexpr,
):
    # One program per (head, BV state rows). Step p reads the rows this CTA wrote
    # at step p - 1 back from global memory in BK column blocks; the CTA barrier
    # orders that write before the read (same-CTA hand-off). FP32 throughout
    # (ieee dot). The barrier is load-bearing twice: it orders the hand-off, and it
    # keeps Triton from software-pipelining the loop (a pipelined loop would
    # prefetch x[p + 1] before this CTA stores it). Both launches also pass
    # num_stages=1, so the scan stays un-pipelined without relying on the barrier.
    hh = tl.program_id(0)
    rv = tl.program_id(1) * BV + tl.arange(0, BV)
    rn = tl.arange(0, D)
    hdd = H * D * D
    row = hh * D * D + rv[:, None] * D
    if FROM_POOL:
        # segment-0 start = pool slot idx[0] ((H, V, K)-contiguous slot), zero for a
        # -1 sentinel row (== torch.where(valid, pool[slot].float(), 0.0))
        slot = tl.load(idx)
        s64 = tl.maximum(slot, 0).to(tl.int64)
        ok = (rv[:, None] >= 0) & (slot >= 0)
        xs = tl.load(pool + s64 * pool_stride0 + row + rn[None, :], mask=ok, other=0.0)
        tl.store(x + row + rn[None, :], xs.to(tl.float32))
    else:
        tl.store(x + row + rn[None, :], tl.load(x0 + row + rn[None, :]))
    for p in range(P - 1):
        tl.debug_barrier()
        acc = tl.load(c + p * hdd + row + rn[None, :])
        for kb in tl.static_range(0, D, BK):
            rk = kb + tl.arange(0, BK)
            xa = tl.load(x + p * hdd + row + rk[None, :])
            mm = tl.load(m + p * hdd + hh * D * D + rk[:, None] * D + rn[None, :])
            acc += tl.dot(xa, mm, input_precision="ieee")
        tl.store(x + (p + 1) * hdd + row + rn[None, :], acc)


def _scan(x0: torch.Tensor, m: torch.Tensor, c: torch.Tensor, x: torch.Tensor) -> None:
    n_seg, H, D, _ = x.shape
    _dseg_scan_kernel[(H, D // _SCAN_BV)](
        x0,
        m,
        c,
        x,
        n_seg,
        x0,
        0,
        x0,
        H=H,
        D=D,
        BV=_SCAN_BV,
        BK=_SCAN_BK,
        FROM_POOL=False,
        num_warps=4,
        num_stages=1,
    )


def _scan_pool(
    pool: torch.Tensor,
    idx: torch.Tensor,
    m: torch.Tensor,
    c: torch.Tensor,
    x: torch.Tensor,
) -> None:
    n_seg, H, D, _ = x.shape
    _dseg_scan_kernel[(H, D // _SCAN_BV)](
        x,
        m,
        c,
        x,
        n_seg,
        pool,
        pool.stride(0),
        idx,
        H=H,
        D=D,
        BV=_SCAN_BV,
        BK=_SCAN_BK,
        FROM_POOL=True,
        num_warps=4,
        num_stages=1,
    )


@triton.jit
def _dseg_prep_kernel(
    base,
    cu_seqlens,
    cu,
    co,
    c,
    mb,
    NSEG,
    NP2: tl.constexpr,
    D: tl.constexpr,
    BV: tl.constexpr,
    BT: tl.constexpr,
):
    # Program 0: pseudo bounds cu = min(base, cu_seqlens[1]) and chunk offsets
    # co = [0, cumsum(cdiv(diff(cu), BT))] (== _pseudo_cu + the torch ops). Every
    # program: one [BV, D] row block of C (= 0) and M (= I) for one (segment, head).
    pid = tl.program_id(0)
    if pid == 0:
        r = tl.arange(0, NP2)
        end = tl.load(cu_seqlens + 1)
        lo = tl.minimum(tl.load(base + r, mask=r <= NSEG, other=0), end)
        tl.store(cu + r, lo, mask=r <= NSEG)
        hi = tl.minimum(tl.load(base + r + 1, mask=r < NSEG, other=0), end)
        n = tl.where(r < NSEG, (hi - lo + BT - 1) // BT, 0)
        cs = tl.cumsum(n, 0)
        tl.store(co + r + 1, cs.to(co.dtype.element_ty), mask=r < NSEG)
        tl.store(co + r, (cs * 0).to(co.dtype.element_ty), mask=r == 0)
    nb = D // BV
    sh = pid // nb
    rv = (pid % nb) * BV + tl.arange(0, BV)
    rn = tl.arange(0, D)
    off = sh * D * D + rv[:, None] * D + rn[None, :]
    tl.store(c + off, tl.zeros([BV, D], dtype=tl.float32))
    tl.store(mb + off, tl.where(rv[:, None] == rn[None, :], 1.0, 0.0).to(tl.float32))


@triton.jit
def _dseg_store_kernel(pool, pool_stride0, idx, xl, D: tl.constexpr, BV: tl.constexpr):
    # pool[idx[0]] = x_last for a valid row; a -1 sentinel row stores nothing
    # (== _store_final for an FP32 pool).
    hh = tl.program_id(0)
    rv = tl.program_id(1) * BV + tl.arange(0, BV)
    rn = tl.arange(0, D)
    row = hh * D * D + rv[:, None] * D + rn[None, :]
    slot = tl.load(idx)
    if slot >= 0:
        tl.store(pool + slot.to(tl.int64) * pool_stride0 + row, tl.load(xl + row))


def _store_final(
    pool: torch.Tensor, slot: torch.Tensor, valid: torch.Tensor, x_last: torch.Tensor
) -> None:
    # Sentinel rows (index < 0) keep the slot's own value (served: no store).
    cur = pool.index_select(0, slot)
    pool.index_put_(
        (slot,),
        torch.where(valid.view(1, 1, 1, 1), x_last.unsqueeze(0).to(pool.dtype), cur),
    )


def _pseudo_cu(base: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
    # Segment bounds clamped to the sequence end (cu_seqlens[1] may be < T).
    return torch.minimum(base, cu_seqlens[1:2])


def _track_local(tci: torch.Tensor, c0: torch.Tensor, c1: torch.Tensor):
    # Which segment holds the tracked chunk, and its index inside that segment.
    hit = (tci >= c0) & (tci < c1)
    return hit, torch.where(hit, tci - c0, torch.full_like(c0, -1))


_PLANS = {}
_CONST = {}
_SIDE = {}


def _plan(nt: int, num_segments: int, cu_dtype: torch.dtype, device: torch.device):
    key = (nt, num_segments, cu_dtype, device)
    plan = _PLANS.get(key)
    if plan is None:
        seg_chunks = triton.cdiv(nt, num_segments)
        n_seg = triton.cdiv(nt, seg_chunks)
        c0 = [s * seg_chunks for s in range(n_seg)]
        c1 = [min(nt, (s + 1) * seg_chunks) for s in range(n_seg)]
        plan = (
            n_seg,
            # token bounds before the clamp to cu_seqlens[1]; the last is open-ended
            torch.tensor(
                [b * CHUNK_SIZE for b in c0] + [torch.iinfo(cu_dtype).max],
                dtype=cu_dtype,
                device=device,
            ),
            torch.arange(n_seg, dtype=torch.int32, device=device),
            torch.tensor(c0, dtype=torch.int32, device=device),
            torch.tensor(c1, dtype=torch.int32, device=device),
        )
        _PLANS[key] = plan
    return plan


def _identity(n_seg: int, H: int, K: int, device: torch.device) -> torch.Tensor:
    key = ("eye", n_seg, H, K, device)
    t = _CONST.get(key)
    if t is None:
        t = (
            torch.eye(K, dtype=torch.float32, device=device)
            .expand(n_seg, H, K, K)
            .contiguous()
        )
        _CONST[key] = t
    return t


def _zero_u(u: torch.Tensor) -> torch.Tensor:
    # One zero buffer per (dtype, device), grown to the largest u seen; never written.
    key = ("zu", u.dtype, u.device)
    t = _CONST.get(key)
    if t is None or t.numel() < u.numel():
        t = torch.zeros(u.numel(), dtype=u.dtype, device=u.device)
        _CONST[key] = t
    return t[: u.numel()].view(u.shape)


def _side_stream(device: torch.device) -> torch.cuda.Stream:
    s = _SIDE.get(device)
    if s is None:
        s = torch.cuda.Stream(device=device)
        _SIDE[device] = s
    return s


def dseg_eligible(
    k: torch.Tensor,
    u: torch.Tensor,
    gk: Optional[torch.Tensor],
    initial_state: Optional[torch.Tensor],
    initial_state_indices: Optional[torch.Tensor],
    cu_seqlens: Optional[torch.Tensor],
    chunk_indices: Optional[torch.Tensor],
    num_segments: int,
    min_tokens: int,
    use_exp2: bool,
) -> bool:
    if num_segments < 2 or not use_exp2 or gk is None:
        return False
    if initial_state is None or initial_state_indices is None:
        return False
    if cu_seqlens is None or cu_seqlens.numel() != 2 or chunk_indices is None:
        return False
    B, _, _, K = k.shape
    H, V = u.shape[-2], u.shape[-1]
    nt = chunk_indices.shape[0]
    if (
        B != 1
        or K != 128
        or V != 128
        or nt * CHUNK_SIZE < max(min_tokens, 2 * CHUNK_SIZE)
    ):
        return False
    if initial_state.dim() != 4 or tuple(initial_state.shape[1:]) != (H, V, K):
        return False
    if tuple(initial_state.stride()[1:]) != (V * K, K, 1):
        return False
    if not k.is_cuda or torch.cuda.is_current_stream_capturing():
        return False
    return True


def chunk_gated_delta_rule_fwd_h_dseg(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    gk: Optional[torch.Tensor],
    initial_state: Optional[torch.Tensor],
    initial_state_indices: Optional[torch.Tensor],
    cu_seqlens: Optional[torch.Tensor],
    chunk_indices: Optional[torch.Tensor],
    chunk_offsets: Optional[torch.Tensor] = None,
    use_exp2: bool = True,
    track_state: Optional[torch.Tensor] = None,
    track_chunk_idx: Optional[torch.Tensor] = None,
    num_segments: int = KDA_SEGSCAN_P,
    min_tokens: int = KDA_SEGSCAN_MIN_T,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Drop-in for chunk_gated_delta_rule_fwd_h (same outputs and pool update);
    runs the served call when the inputs are outside D-SEG's scope."""
    if not dseg_eligible(
        k,
        u,
        gk,
        initial_state,
        initial_state_indices,
        cu_seqlens,
        chunk_indices,
        num_segments,
        min_tokens,
        use_exp2,
    ):
        return chunk_gated_delta_rule_fwd_h(
            k=k,
            w=w,
            u=u,
            gk=gk,
            initial_state=initial_state,
            initial_state_indices=initial_state_indices,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            use_exp2=use_exp2,
            track_state=track_state,
            track_chunk_idx=track_chunk_idx,
            chunk_offsets=chunk_offsets,
        )
    H, V, K = u.shape[-2], u.shape[-1], k.shape[-1]
    dev = k.device
    nt = chunk_indices.shape[0]
    n_seg, base, seg_ids, c0, c1 = _plan(nt, num_segments, cu_seqlens.dtype, dev)
    shp = (n_seg, H, V, K)
    glue = bool(KDA_SEGSCAN_GLUE) and initial_state.dtype == torch.float32
    if glue:
        # one launch: pseudo bounds, chunk offsets, C = 0, M = I (no host sync)
        cu = torch.empty(n_seg + 1, dtype=base.dtype, device=dev)
        co = torch.empty_like(cu)
        C = torch.empty(shp, dtype=torch.float32, device=dev)
        Mb = torch.empty(shp, dtype=torch.float32, device=dev)
        _dseg_prep_kernel[(n_seg * H * (K // _SCAN_BV),)](
            base,
            cu_seqlens,
            cu,
            co,
            C,
            Mb,
            n_seg,
            NP2=triton.next_power_of_2(n_seg + 1),
            D=K,
            BV=_SCAN_BV,
            BT=CHUNK_SIZE,
            num_warps=4,
        )
    else:
        # pseudo-sequence bounds and chunk offsets on the device (no host sync)
        cu = _pseudo_cu(base, cu_seqlens)
        lens = cu[1:] - cu[:-1]
        co = torch.cat(
            [cu.new_zeros(1), torch.cumsum(triton.cdiv(lens, CHUNK_SIZE), 0)]
        ).to(cu.dtype)
        C = torch.zeros(shp, dtype=torch.float32, device=dev)
        Mb = _identity(n_seg, H, K, dev).clone()
    zu = _zero_u(u)

    # The h prefetch (SGLANG_K31_KDA_H_PREFETCH) stays off on every D-SEG pass: it
    # slows these segment calls.
    def run(uu, state, **kw):
        return chunk_gated_delta_rule_fwd_h(
            k=k,
            w=w,
            u=uu,
            gk=gk,
            initial_state=state,
            initial_state_indices=seg_ids,
            cu_seqlens=cu,
            chunk_indices=chunk_indices,
            chunk_offsets=co,
            use_exp2=use_exp2,
            prefetch_k=False,
            **kw,
        )

    # Phase A: C pass on the current stream, M pass on a side stream.
    main = torch.cuda.current_stream(dev)
    side = _side_stream(dev)
    side.wait_stream(main)
    run(u, C, save_new_value=False)
    with torch.cuda.stream(side):
        run(zu, Mb, save_new_value=False)
    main.wait_stream(side)

    # Phase B: segment start states from the pool slot (zero start for a sentinel row).
    idx = initial_state_indices[:1]
    X = torch.empty(shp, dtype=torch.float32, device=dev)
    if glue:
        _scan_pool(initial_state, idx, Mb, C, X)
    else:
        valid = idx >= 0
        slot = idx.clamp(min=0).long()
        x0 = torch.where(
            valid.view(1, 1, 1), initial_state.index_select(0, slot)[0].float(), 0.0
        ).contiguous()
        _scan(x0, Mb, C, X)
    Xin = X

    # Phase C: served kernel from X_p on every segment (writes X_p's end state in place).
    if track_state is not None:
        tci = track_chunk_idx[:1].to(torch.int32)
        hit, loc = _track_local(tci, c0, c1)
        ts = torch.empty(shp, dtype=torch.float32, device=dev)
        h, v_new = run(u, Xin, track_state=ts, track_chunk_idx=loc)
        sel = hit.to(torch.int32).argmax().view(1)
        track_state[0].copy_(
            torch.where(hit.any(), ts.index_select(0, sel)[0], track_state[0])
        )
    else:
        h, v_new = run(u, Xin)
    if glue:
        _dseg_store_kernel[(H, K // _SCAN_BV)](
            initial_state,
            initial_state.stride(0),
            idx,
            Xin[n_seg - 1],
            D=K,
            BV=_SCAN_BV,
            num_warps=4,
        )
    else:
        _store_final(initial_state, slot, valid, Xin[n_seg - 1])
    return h, v_new
