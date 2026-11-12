# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""CuTe DSL device kernel KDA conv-MTP.

conv enabled, no bias, optional fused gated RMSNorm, lower_bound gate, Q/K
L2 norm, beta sigmoid, ILP=2, W=4. Recurrent-state tiles are cp.async'd into
NUM_STATE_STAGES smem stages. Phase 2 walks the V // TILE_V state tiles in
passes of TILES_PER_PASS: a multi-token verify keeps them all register-resident
across the token chain, a single-token step streams one at a time.
"""

import os

import cuda.bindings.driver as cuda
import cutlass
from cutlass import cute
from cutlass._mlir.dialects import nvvm
from cutlass.cute.nvgpu import cpasync

from sglang.kernels.arch import current_arch
from sglang.kernels.jit.cute_aot_cache import get_jit_cache

WARP_SIZE = 32
TILE_K = 128
# K31HT_KDA_P1ILP: 1 = the q / k / g phase-1 warps issue every global load of their token walk
# up front (same addresses, clamps and FP order as stock; bitwise == stock). Unset / 0 = stock. Engaged only for num_spec > 0.
# LAST_P1ILP records the form the last call used (tests read it).
LAST_P1ILP = 0
KERNEL_WIDTH = 4
# Phase 1 conv jobs: q, k, g. Each gets its own equal-sized warp group; the
# v-conv gets whatever is left over.
P1_NUM_JOBS = 3
# Share of the block's warps handed to the v-conv, as a divisor: 4 -> a quarter.
# Tuned, not structural — v is one conv over V channels with no norm or gate, so
# it needs fewer warps than q/k/g, but the exact split is empirical.
P1_V_WARP_DIVISOR = 4
# Recurrent-state smem stages, i.e. how many of the V // TILE_V state tiles are
# in flight at once. Three measured worse than two even for the streaming path,
# where a deeper pipeline should have helped: the transfer is not the bottleneck
# and the extra stage only costs smem.
NUM_STATE_STAGES = 2

# Block size is chosen per launch, not fixed: below one wave (H*N <= SM count)
# every SM gets one block regardless, so a wider block is free warps. Above it,
# two narrow blocks per SM let one issue while the other sits at a barrier,
# which a single wide block cannot do. See _block_threads.
BLOCK_THREADS_NARROW = 256
BLOCK_THREADS_WIDE = 512

# Recurrence lane split: K spans a subgroup instead of all 32 lanes, so several
# v-rows reduce concurrently and each butterfly is shorter. The resident path
# uses eight lanes; the ns0 streaming path uses sixteen because halving its
# repeated q/k/decay LDS sequence outweighs the one extra shuffle.
P2_LANES_RESIDENT = 8
P2_LANES_STREAM = 16

HEAD_DIM = TILE_K
VEC_SIZE = HEAD_DIM // WARP_SIZE


def _stream_state(*, block_threads: int, num_spec: int) -> bool:
    """Whether phase 2 streams the recurrent state instead of holding it.

    See the TILES_PER_PASS comment in the kernel: streaming trades register
    residency for occupancy, which needs a single token (nothing to lose) and a
    grid above one wave (something to gain, signalled by the narrow block).
    """
    return num_spec == 0 and block_threads == BLOCK_THREADS_NARROW


def _p2_lanes(*, block_threads: int, num_spec: int) -> int:
    return (
        P2_LANES_STREAM
        if _stream_state(block_threads=block_threads, num_spec=num_spec)
        else P2_LANES_RESIDENT
    )


def _qk_smem(*, block_threads: int, num_spec: int) -> tuple[tuple[int, ...], int]:
    """Stride tuple and per-token element count for sQ/sK/sG.

    Both modes index the same (token, lane row, vector, element) shape; only the
    strides differ. Lane-major (see P2_QK_ROW) makes a lane's P2_VEC channels
    contiguous so they load as 128-bit vectors, at the cost of the bank pad.
    Channel-major places element (k_grp, jv, c) at plain channel
    k_grp + (jv * 4 + c) * P2_LANES_K, which is what the scalar path wants: no
    pad, and the same conflict-free access as an unpermuted [token, K] tile.
    """
    p2_lanes = _p2_lanes(block_threads=block_threads, num_spec=num_spec)
    p2_vec = TILE_K // p2_lanes
    p2_qk_row = p2_vec + 4
    if _stream_state(block_threads=block_threads, num_spec=num_spec):
        return (p2_lanes * p2_qk_row, p2_qk_row, 4, 1), p2_lanes * p2_qk_row
    return (TILE_K, 1, 4 * p2_lanes, p2_lanes), TILE_K


def _state_tile_v(*, block_threads: int, num_spec: int) -> int:
    """Height of the recurrent-state tile staged in smem, in value rows.

    Never below P2_ROWS_LANE rows per warp: a warp's P2_LANES_K-lane groups
    reduce that many rows concurrently, so a shorter tile leaves lanes idle.

    Streaming wants exactly that floor -- both the smem stage and the
    register-resident slice scale with the tile, and a consumer is waiting per
    tile, so a short tile pipelines. Residency wants the opposite: no token can
    start until every tile has landed, so there is nothing to overlap and the
    best shape is the fewest, largest tiles that keep all NUM_STATE_STAGES in
    flight at once with no stage reuse.
    """
    p2_rows_lane = WARP_SIZE // _p2_lanes(
        block_threads=block_threads, num_spec=num_spec
    )
    min_rows = (block_threads // WARP_SIZE) * p2_rows_lane
    if _stream_state(block_threads=block_threads, num_spec=num_spec):
        # Sixteen rows doubles the stage waits and loses ~12% at B32. A 32-row
        # tile retains four phases while the sixteen-lane split still halves
        # the q/k/decay register arrays and LDS sequence.
        return max(min_rows, 32)
    return max(min_rows, HEAD_DIM // NUM_STATE_STAGES)


def _issue_state_tile(
    state_g2s_copy: cute.TiledCopy,
    thr_state_copy,
    gStateTiles: cute.Tensor,
    sState: cute.Tensor,
    i_v: int,
    num_stages: int,
) -> None:
    """cp.async state tile ``i_v`` into the stage it maps to, and commit it."""
    cute.copy(
        state_g2s_copy,
        thr_state_copy.partition_S(gStateTiles[(None, None, i_v)]),
        thr_state_copy.partition_D(sState[(None, None, i_v % num_stages)]),
    )
    cute.arch.cp_async_commit_group()


@cute.kernel
def kda_decode_mtp_kernel(
    state_g2s_copy: cute.TiledCopy,
    h0: cute.Tensor,
    x_q: cute.Tensor,
    x_k: cute.Tensor,
    x_v: cute.Tensor,
    w_q: cute.Tensor,
    w_k: cute.Tensor,
    w_v: cute.Tensor,
    cs_q: cute.Tensor,
    cs_k: cute.Tensor,
    cs_v: cute.Tensor,
    A_log: cute.Tensor,
    g: cute.Tensor,
    dt_bias: cute.Tensor,
    beta: cute.Tensor,
    o: cute.Tensor,
    ht: cute.Tensor,
    intermediate_state_indices: cute.Tensor,
    intermediate_conv_q: cute.Tensor,
    intermediate_conv_k: cute.Tensor,
    intermediate_conv_v: cute.Tensor,
    ring_rawv: cute.Tensor,
    ring_rawk: cute.Tensor,
    ring_g: cute.Tensor,
    ring_beta: cute.Tensor,
    onorm_g: cute.Tensor,
    onorm_weight: cute.Tensor,
    fold_pending: cute.Tensor,
    smem_qk_layout: cute.Layout,
    smem_state_layout: cute.Layout,
    ssm_state_indices: cute.Tensor,
    cu_seqlens: cute.Tensor,
    scale: cutlass.Constexpr[float],
    NUM_SPEC: cutlass.Constexpr[int],
    BLOCK_THREADS: cutlass.Constexpr[int],
    P2_LANES_K: cutlass.Constexpr[int],
    lower_bound: cutlass.Constexpr[float],
    CACHE_RING: cutlass.Constexpr[bool],
    APPLY_ONORM: cutlass.Constexpr[bool],
    onorm_eps: cutlass.Constexpr[float],
    FOLD: cutlass.Constexpr[bool],
    RING_HALF: cutlass.Constexpr[int],
    P1ILP: cutlass.Constexpr[int],
):
    """KDA MTP decode — SMEM pre-compute + register-resident state.

    One block owns all 128 value rows, so APPLY_ONORM can reduce the RMS
    denominator without cross-block synchronization.

    FOLD (commit-then-verify, opt-in): ``temporal[slot]`` still holds the state
    before the previous verify window; ``fold_pending[slot]`` holds how many of
    that window's ring records were accepted and which ring half holds them.
    The block replays them into the register-resident state with the stock
    commit's method (division-form L2 norm, exp gate decay, delta rule), writes
    the committed state back to ``temporal[slot]``, verifies the new window from
    registers, and writes the new window's records to the other ring half. The
    state is read once per step instead of twice (verify + commit).
    """
    tidx, _, _ = cute.arch.thread_idx()
    in_warp_tid = tidx % WARP_SIZE
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    i_hv, i_n, _ = cute.arch.block_idx()
    head_off = i_hv * HEAD_DIM
    T_LOOP = 1 + NUM_SPEC
    # Conv-precompute warp budget: q/k/g each get P1_JOB_WARPS warps and split
    # the token dimension across them; the rest carry the v-conv.
    NUM_WARPS = BLOCK_THREADS // WARP_SIZE
    P1_V_WARPS = max(1, NUM_WARPS // P1_V_WARP_DIVISOR)
    P1_JOB_WARPS = (NUM_WARPS - P1_V_WARPS) // P1_NUM_JOBS
    P1_QKG_WARPS = P1_NUM_JOBS * P1_JOB_WARPS
    V_CH_PER_THREAD = TILE_K // (P1_V_WARPS * WARP_SIZE)
    P2_ROWS_LANE = WARP_SIZE // P2_LANES_K
    P2_VEC = TILE_K // P2_LANES_K
    TILE_V = _state_tile_v(block_threads=BLOCK_THREADS, num_spec=NUM_SPEC)
    NUM_V_TILES = HEAD_DIM // TILE_V
    STATE_STAGES = min(NUM_STATE_STAGES, NUM_V_TILES)
    NUM_V_ROWS = TILE_V // NUM_WARPS
    P2_BATCHES = NUM_V_ROWS // P2_ROWS_LANE
    # How many state tiles phase 2 holds in registers at once. Holding all of
    # them keeps the whole 128x128 state resident across the token chain, which
    # is what makes multi-token verify cheap -- but it costs
    # HEAD_DIM * TILE_K / BLOCK_THREADS registers however the lanes are split,
    # and that is what caps the kernel at two blocks per SM. Streaming one tile
    # at a time spends those registers on occupancy instead, which is worth it
    # only when there is nothing to lose (a single token, so no reuse) and
    # something to gain (a grid above one wave, which is exactly what the narrow
    # block width signals -- see _block_threads).
    STREAM_STATE = _stream_state(block_threads=BLOCK_THREADS, num_spec=NUM_SPEC)
    TILES_PER_PASS = 1 if STREAM_STATE else NUM_V_TILES
    smem = cutlass.utils.SmemAllocator()
    sQ = smem.allocate_tensor(cutlass.Float32, smem_qk_layout, 16)
    sK = smem.allocate_tensor(cutlass.Float32, smem_qk_layout, 16)
    sG = smem.allocate_tensor(cutlass.Float32, smem_qk_layout, 16)
    sBeta = smem.allocate_tensor(cutlass.Float32, cute.make_layout((T_LOOP,)), 16)
    sVall = smem.allocate_tensor(
        cutlass.Float32, cute.make_layout((T_LOOP * HEAD_DIM,)), 16
    )
    sState = smem.allocate_tensor(h0.element_type, smem_state_layout, 16)
    if cutlass.const_expr(FOLD):
        # Previous window's accepted ring records, staged before this block
        # overwrites the ring: normalized k, exp(gate), v, beta.
        sRK = smem.allocate_tensor(cutlass.Float32, smem_qk_layout, 16)
        sRG = smem.allocate_tensor(cutlass.Float32, smem_qk_layout, 16)
        sRB = smem.allocate_tensor(cutlass.Float32, cute.make_layout((T_LOOP,)), 16)
        sRV = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout((T_LOOP * HEAD_DIM,)), 16
        )
    if cutlass.const_expr(APPLY_ONORM):
        sOall = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout((T_LOOP * HEAD_DIM,)), 16
        )
    else:
        # Compile-time-dead placeholder; avoids charging the non-norm path
        # for an output tile it never touches.
        sOall = sVall
    if cutlass.const_expr(APPLY_ONORM):
        sOnormGate = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout((T_LOOP * HEAD_DIM,)), 16
        )
        sOnormWeight = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout((HEAD_DIM,)), 16
        )
    r_q = cute.make_rmem_tensor(
        cute.make_layout((P2_VEC,), stride=(1,)), cutlass.Float32
    )
    r_k = cute.make_rmem_tensor(
        cute.make_layout((P2_VEC,), stride=(1,)), cutlass.Float32
    )
    r_decay = cute.make_rmem_tensor(
        cute.make_layout((P2_VEC,), stride=(1,)), cutlass.Float32
    )
    # Sized for whichever is larger: one pass of recurrent state (phase 2) or
    # the two conv windows (phase 1). The phases never overlap.
    R_STATE_ELEMS = max(
        TILES_PER_PASS * P2_BATCHES * P2_VEC, 2 * (KERNEL_WIDTH - 1) * VEC_SIZE
    )
    r_state = cute.make_rmem_tensor(
        cute.make_layout((R_STATE_ELEMS,), stride=(1,)), cutlass.Float32
    )
    r_wq = cute.make_rmem_tensor(
        cute.make_layout((KERNEL_WIDTH * VEC_SIZE,), stride=(1,)), cutlass.Float32
    )
    # One channel's KERNEL_WIDTH conv taps are contiguous in the [dim, W]
    # weights, so they load as a single 16-byte vector instead of W strided
    # scalars. These run once per block, i.e. entirely in the fixed cost.
    r_w4 = cute.make_rmem_tensor(
        cute.make_layout((KERNEL_WIDTH,), stride=(1,)), cutlass.Float32
    )
    if cutlass.const_expr(P1ILP > 0):
        # K31HT_KDA_P1ILP buffers: per producer step the token value (r_pix), per (step, advance) the advance value (r_pia), the
        # pre-advance values (r_pip), beta per step (r_pib).
        PI_STEPS = (T_LOOP + P1_JOB_WARPS - 1) // P1_JOB_WARPS
        PI_ADV = P1_JOB_WARPS - 1
        r_pix = cute.make_rmem_tensor(cute.make_layout((PI_STEPS * VEC_SIZE,), stride=(1,)), cutlass.Float32)
        r_pia = cute.make_rmem_tensor(cute.make_layout((max(1, PI_STEPS * PI_ADV) * VEC_SIZE,), stride=(1,)), cutlass.Float32)
        r_pip = cute.make_rmem_tensor(cute.make_layout((max(1, PI_ADV) * VEC_SIZE,), stride=(1,)), cutlass.Float32)
        r_pib = cute.make_rmem_tensor(cute.make_layout((PI_STEPS,), stride=(1,)), cutlass.Float32)
    if cutlass.const_expr(STREAM_STATE):
        r_v4q = cute.make_rmem_tensor(
            cute.make_layout((4,), stride=(1,)), cutlass.Float32
        )
        r_v4k = cute.make_rmem_tensor(
            cute.make_layout((4,), stride=(1,)), cutlass.Float32
        )
        r_v4g = cute.make_rmem_tensor(
            cute.make_layout((4,), stride=(1,)), cutlass.Float32
        )

    slot = ssm_state_indices[i_n]
    # CUDA-graph padding rows use slot == -1.
    if slot < 0:
        cute.arch.griddepcontrol_wait()
        pad_bos = cu_seqlens[i_n]
        for i_t in cutlass.range_constexpr(T_LOOP):
            if tidx < HEAD_DIM:
                o[0, pad_bos + i_t, i_hv, tidx] = cutlass.BFloat16(0.0)
        cute.arch.griddepcontrol_launch_dependents()
        # nvvm.exit, not `return`: the DSL rejects an early return out of a
        # staged if (UNSUP_EARLY_EXIT).
        nvvm.exit()

    # int64: `slot * stride` overflows int32 on envelope-strided pools.
    slot = cutlass.Int64(slot)

    # q/k/g each run on P1_JOB_WARPS warps split by token parity and the v-conv
    # takes the rest. Each token's conv is an independent window over globals,
    # so the split needs no cross-warp communication.
    p1_job = warp_idx % P1_NUM_JOBS
    p1_par = warp_idx // P1_NUM_JOBS
    # Only the q-conv warps read the q window and only the k-conv warps the k
    # window; the g and v warps overwrite r_state in phase 2 without reading it.
    if warp_idx < P1_QKG_WARPS:
        if p1_job == 0:
            for i in range(VEC_SIZE):
                k_idx = i * 32 + in_warp_tid
                for w in range(KERNEL_WIDTH - 1):
                    r_state[w * VEC_SIZE + i] = cutlass.Float32(
                        cs_q[slot, head_off + k_idx, w]
                    )
        elif p1_job == 1:
            for i in range(VEC_SIZE):
                k_idx = i * 32 + in_warp_tid
                for w in range(KERNEL_WIDTH - 1):
                    r_state[(KERNEL_WIDTH - 1) * VEC_SIZE + w * VEC_SIZE + i] = (
                        cutlass.Float32(cs_k[slot, head_off + k_idx, w])
                    )

    if warp_idx < P1_QKG_WARPS:
        if p1_job == 0:
            for i in range(VEC_SIZE):
                cute.autovec_copy(w_q[(head_off + i * 32 + in_warp_tid, None)], r_w4)
                for w in range(KERNEL_WIDTH):
                    r_wq[w * VEC_SIZE + i] = r_w4[w]
        elif p1_job == 1:
            for i in range(VEC_SIZE):
                cute.autovec_copy(w_k[(head_off + i * 32 + in_warp_tid, None)], r_w4)
                for w in range(KERNEL_WIDTH):
                    r_wq[w * VEC_SIZE + i] = r_w4[w]

    # The 128-bit copy atom maps each thread to coalesced K vectors.
    gState = h0[(slot, i_hv, None, None)]
    gStateTiles = cute.local_tile(gState, (TILE_V, TILE_K), (None, 0))
    thr_state_copy = state_g2s_copy.get_slice(tidx)
    for i_v in cutlass.range_constexpr(STATE_STAGES):
        _issue_state_tile(
            state_g2s_copy, thr_state_copy, gStateTiles, sState, i_v, STATE_STAGES
        )

    # Ring row offset of this window's records (ping-pong halves in FOLD mode).
    ring_base = 0
    if cutlass.const_expr(FOLD):
        # fold_pending[slot] = accepted records | (ring half holding them) << 8.
        # The previous window sits in one half of the ring and this verify
        # writes the other, so nothing orders the reads below against the conv
        # phase's ring writes: warp w loads record w into registers now (like
        # `temporal` and the conv state above, a per-slot persistent buffer read
        # before griddepcontrol.wait; only the previous step's post-verify
        # launches write `fold_pending`), and stages it to shared memory after
        # the conv phase, off the critical path.
        pv = cutlass.Int32(fold_pending[slot])
        a_prev = cutlass.min(pv % 256, T_LOOP)
        half_old = (pv // 256) % 2
        base_old = half_old * RING_HALF
        ring_base = (1 - half_old) * RING_HALF
        r_rk = cute.make_rmem_tensor(
            cute.make_layout((VEC_SIZE,), stride=(1,)), cutlass.Float32
        )
        r_rg = cute.make_rmem_tensor(
            cute.make_layout((VEC_SIZE,), stride=(1,)), cutlass.Float32
        )
        r_rv = cute.make_rmem_tensor(
            cute.make_layout((VEC_SIZE,), stride=(1,)), cutlass.Float32
        )
        r_rb = cutlass.Float32(0.0)
        if warp_idx < a_prev:
            for i in range(VEC_SIZE):
                k_idx = i * WARP_SIZE + in_warp_tid
                r_rk[i] = cutlass.Float32(
                    ring_rawk[slot, i_hv, base_old + warp_idx, k_idx]
                )
                r_rg[i] = cutlass.Float32(ring_g[slot, i_hv, base_old + warp_idx, k_idx])
                r_rv[i] = cutlass.Float32(
                    ring_rawv[slot, i_hv, base_old + warp_idx, k_idx]
                )
            r_rb = cutlass.Float32(ring_beta[slot, i_hv, base_old + warp_idx])

    cute.arch.griddepcontrol_wait()

    bos = cu_seqlens[i_n]
    eos = cu_seqlens[i_n + 1]
    n_tok = eos - bos
    scratch_row = intermediate_state_indices[i_n]
    r_exp_A = cutlass.Float32(0.0)

    # Private weights remove the initial cross-warp dependency.

    if warp_idx < P1_QKG_WARPS:
        if cutlass.const_expr(P1ILP > 0):
            # K31HT_KDA_P1ILP: every global load of this warp's token walk, same addresses and clamps as the stock code below
            # (which then reads these registers instead); issued before any of the walk's stores so they overlap.
            if p1_job == 0:
                for _pi in cutlass.range_constexpr(PI_ADV):
                    for i in cutlass.range_constexpr(VEC_SIZE):
                        r_pip[_pi * VEC_SIZE + i] = cutlass.Float32(
                            x_q[0, bos + cutlass.min(_pi, T_LOOP - 1), i_hv, i * 32 + in_warp_tid]
                        )
                for _s in cutlass.range_constexpr(PI_STEPS):
                    _it = p1_par + _s * P1_JOB_WARPS
                    _tok = bos + cutlass.min(_it, n_tok - 1)
                    for i in cutlass.range_constexpr(VEC_SIZE):
                        r_pix[_s * VEC_SIZE + i] = cutlass.Float32(
                            x_q[0, _tok, i_hv, i * 32 + in_warp_tid]
                        )
                    for _a in cutlass.range_constexpr(PI_ADV):
                        _nxp = bos + cutlass.min(_it + 1 + _a, T_LOOP - 1)
                        for i in cutlass.range_constexpr(VEC_SIZE):
                            r_pia[(_s * PI_ADV + _a) * VEC_SIZE + i] = cutlass.Float32(
                                x_q[0, _nxp, i_hv, i * 32 + in_warp_tid]
                            )
            elif p1_job == 1:
                for _pi in cutlass.range_constexpr(PI_ADV):
                    for i in cutlass.range_constexpr(VEC_SIZE):
                        r_pip[_pi * VEC_SIZE + i] = cutlass.Float32(
                            x_k[0, bos + cutlass.min(_pi, T_LOOP - 1), i_hv, i * 32 + in_warp_tid]
                        )
                for _s in cutlass.range_constexpr(PI_STEPS):
                    _it = p1_par + _s * P1_JOB_WARPS
                    _tok = bos + cutlass.min(_it, n_tok - 1)
                    for i in cutlass.range_constexpr(VEC_SIZE):
                        r_pix[_s * VEC_SIZE + i] = cutlass.Float32(
                            x_k[0, _tok, i_hv, i * 32 + in_warp_tid]
                        )
                    r_pib[_s] = cutlass.Float32(beta[0, _tok, i_hv])
                    for _a in cutlass.range_constexpr(PI_ADV):
                        _nxp = bos + cutlass.min(_it + 1 + _a, T_LOOP - 1)
                        for i in cutlass.range_constexpr(VEC_SIZE):
                            r_pia[(_s * PI_ADV + _a) * VEC_SIZE + i] = cutlass.Float32(
                                x_k[0, _nxp, i_hv, i * 32 + in_warp_tid]
                            )
            else:
                for _s in cutlass.range_constexpr(PI_STEPS):
                    _it = p1_par + _s * P1_JOB_WARPS
                    _tok = bos + cutlass.min(_it, n_tok - 1)
                    for i in cutlass.range_constexpr(VEC_SIZE):
                        r_pix[_s * VEC_SIZE + i] = cutlass.Float32(
                            g[0, _tok, i_hv, i * 32 + in_warp_tid]
                        )
        if p1_job == 2:
            r_exp_A = cute.math.exp(cutlass.Float32(A_log[i_hv]), fastmath=True)
            # The gate warps do not use conv-window registers. Hold invariant
            # bias values there until phase 2 overwrites the state registers.
            for i in range(VEC_SIZE):
                r_state[i] = cutlass.Float32(
                    dt_bias[i_hv * HEAD_DIM + i * WARP_SIZE + in_warp_tid]
                )
        # Warp starting at token p1_par needs its window advanced that
        # many steps; the g path is pointwise and needs none.
        if p1_job == 0:
            for _pi in cutlass.range_constexpr(P1_JOB_WARPS - 1):
                if _pi < cutlass.min(p1_par, n_tok):
                    for i in range(VEC_SIZE):
                        if cutlass.const_expr(P1ILP > 0):
                            _xn = r_pip[_pi * VEC_SIZE + i]
                        else:
                            _xn = cutlass.Float32(
                                x_q[0, bos + _pi, i_hv, i * 32 + in_warp_tid]
                            )
                        r_state[0 * VEC_SIZE + i] = r_state[1 * VEC_SIZE + i]
                        r_state[1 * VEC_SIZE + i] = r_state[2 * VEC_SIZE + i]
                        r_state[2 * VEC_SIZE + i] = _xn
        elif p1_job == 1:
            for _pi in cutlass.range_constexpr(P1_JOB_WARPS - 1):
                if _pi < cutlass.min(p1_par, n_tok):
                    for i in range(VEC_SIZE):
                        if cutlass.const_expr(P1ILP > 0):
                            _xn = r_pip[_pi * VEC_SIZE + i]
                        else:
                            _xn = cutlass.Float32(
                                x_k[0, bos + _pi, i_hv, i * 32 + in_warp_tid]
                            )
                        r_state[(KERNEL_WIDTH - 1) * VEC_SIZE + 0 * VEC_SIZE + i] = (
                            r_state[(KERNEL_WIDTH - 1) * VEC_SIZE + 1 * VEC_SIZE + i]
                        )
                        r_state[(KERNEL_WIDTH - 1) * VEC_SIZE + 1 * VEC_SIZE + i] = (
                            r_state[(KERNEL_WIDTH - 1) * VEC_SIZE + 2 * VEC_SIZE + i]
                        )
                        r_state[(KERNEL_WIDTH - 1) * VEC_SIZE + 2 * VEC_SIZE + i] = _xn
        # Unroll only the short producer token loop; recurrence stays rolled.
        for producer_step in cutlass.range_constexpr(
            (T_LOOP + P1_JOB_WARPS - 1) // P1_JOB_WARPS
        ):
            i_t = p1_par + producer_step * P1_JOB_WARPS
            if i_t < T_LOOP:
                token = bos + cutlass.min(i_t, n_tok - 1)
                if p1_job == 0:
                    for i_pair in range(VEC_SIZE // 2):
                        i0 = i_pair * 2
                        i1 = i_pair * 2 + 1
                        k_idx0 = i0 * 32 + in_warp_tid
                        k_idx1 = i1 * 32 + in_warp_tid
                        r_conv_0 = 0.0
                        r_conv_1 = 0.0
                        for w in range(KERNEL_WIDTH - 1):
                            r_conv_0 += (
                                r_state[w * VEC_SIZE + i0] * r_wq[w * VEC_SIZE + i0]
                            )
                            r_conv_1 += (
                                r_state[w * VEC_SIZE + i1] * r_wq[w * VEC_SIZE + i1]
                            )
                        if cutlass.const_expr(P1ILP > 0):
                            r_xq_0 = r_pix[producer_step * VEC_SIZE + i0]
                            r_xq_1 = r_pix[producer_step * VEC_SIZE + i1]
                        else:
                            r_xq_0 = cutlass.Float32(x_q[0, token, i_hv, k_idx0])
                            r_xq_1 = cutlass.Float32(x_q[0, token, i_hv, k_idx1])
                        _cwq_last_0 = r_wq[(KERNEL_WIDTH - 1) * VEC_SIZE + i0]
                        _cwq_last_1 = r_wq[(KERNEL_WIDTH - 1) * VEC_SIZE + i1]
                        r_conv_0 += r_xq_0 * _cwq_last_0
                        r_conv_1 += r_xq_1 * _cwq_last_1
                        e0 = cute.math.exp(-r_conv_0, fastmath=True)
                        e1 = cute.math.exp(-r_conv_1, fastmath=True)
                        sig_0 = cute.arch.rcp_approx(cutlass.Float32(1.0) + e0)
                        sig_1 = cute.arch.rcp_approx(cutlass.Float32(1.0) + e1)
                        r_q[i0] = r_conv_0 * sig_0
                        r_q[i1] = r_conv_1 * sig_1
                        r_state[0 * VEC_SIZE + i0] = r_state[1 * VEC_SIZE + i0]
                        r_state[0 * VEC_SIZE + i1] = r_state[1 * VEC_SIZE + i1]
                        r_state[1 * VEC_SIZE + i0] = r_state[2 * VEC_SIZE + i0]
                        r_state[1 * VEC_SIZE + i1] = r_state[2 * VEC_SIZE + i1]
                        r_state[2 * VEC_SIZE + i0] = r_xq_0
                        r_state[2 * VEC_SIZE + i1] = r_xq_1
                    sum_q = 0.0
                    for i in range(VEC_SIZE):
                        sum_q += r_q[i] * r_q[i]
                    for offset in [16, 8, 4, 2, 1]:
                        sum_q += cute.arch.shuffle_sync_bfly(
                            sum_q, offset=offset, mask=-1, mask_and_clamp=31
                        )
                    rnorm_q_scaled = (
                        cute.math.rsqrt(sum_q + 1e-06, fastmath=True) * scale
                    )
                    for i in range(VEC_SIZE):
                        r_q[i] = r_q[i] * rnorm_q_scaled
                    for i in range(VEC_SIZE):
                        k_idx = i * WARP_SIZE + in_warp_tid
                        qk_grp = k_idx % P2_LANES_K
                        qk_j = k_idx // P2_LANES_K
                        sQ[
                            i_t,
                            qk_grp,
                            qk_j // 4,
                            qk_j % 4,
                        ] = r_q[i]
                elif p1_job == 1:
                    r_b_raw = cutlass.Float32(0.0)
                    if in_warp_tid == 0:
                        if cutlass.const_expr(P1ILP > 0):
                            r_b_raw = r_pib[producer_step]
                        else:
                            r_b_raw = cutlass.Float32(beta[0, token, i_hv])
                    for i in range(VEC_SIZE):
                        k_idx = i * 32 + in_warp_tid
                        r_conv = (
                            r_state[(KERNEL_WIDTH - 1) * VEC_SIZE + 0 * VEC_SIZE + i]
                            * r_wq[0 * VEC_SIZE + i]
                        )
                        r_conv += (
                            r_state[(KERNEL_WIDTH - 1) * VEC_SIZE + 1 * VEC_SIZE + i]
                            * r_wq[1 * VEC_SIZE + i]
                        )
                        r_conv += (
                            r_state[(KERNEL_WIDTH - 1) * VEC_SIZE + 2 * VEC_SIZE + i]
                            * r_wq[2 * VEC_SIZE + i]
                        )
                        if cutlass.const_expr(P1ILP > 0):
                            r_xk = r_pix[producer_step * VEC_SIZE + i]
                        else:
                            r_xk = cutlass.Float32(x_k[0, token, i_hv, k_idx])
                        r_conv += r_xk * r_wq[(KERNEL_WIDTH - 1) * VEC_SIZE + i]
                        r_conv = r_conv * cute.arch.rcp_approx(
                            cutlass.Float32(1.0) + cute.math.exp(-r_conv, fastmath=True)
                        )
                        r_k[i] = r_conv
                        r_state[(KERNEL_WIDTH - 1) * VEC_SIZE + 0 * VEC_SIZE + i] = (
                            r_state[(KERNEL_WIDTH - 1) * VEC_SIZE + 1 * VEC_SIZE + i]
                        )
                        r_state[(KERNEL_WIDTH - 1) * VEC_SIZE + 1 * VEC_SIZE + i] = (
                            r_state[(KERNEL_WIDTH - 1) * VEC_SIZE + 2 * VEC_SIZE + i]
                        )
                        r_state[(KERNEL_WIDTH - 1) * VEC_SIZE + 2 * VEC_SIZE + i] = r_xk
                    # Complete independent key chains before raw-ring stores.
                    if cutlass.const_expr(CACHE_RING):
                        for i in range(VEC_SIZE):
                            k_idx = i * WARP_SIZE + in_warp_tid
                            ring_rawk[slot, i_hv, ring_base + i_t, k_idx] = (
                                cutlass.BFloat16(r_k[i])
                            )
                    sum_k = 0.0
                    for i in range(VEC_SIZE):
                        sum_k += r_k[i] * r_k[i]
                    for offset in [16, 8, 4, 2, 1]:
                        sum_k += cute.arch.shuffle_sync_bfly(
                            sum_k, offset=offset, mask=-1, mask_and_clamp=31
                        )
                    rnorm_k = cute.math.rsqrt(sum_k + 1e-06, fastmath=True)
                    for i in range(VEC_SIZE):
                        r_k[i] = r_k[i] * rnorm_k
                    for i in range(VEC_SIZE):
                        k_idx = i * WARP_SIZE + in_warp_tid
                        qk_grp = k_idx % P2_LANES_K
                        qk_j = k_idx // P2_LANES_K
                        sK[
                            i_t,
                            qk_grp,
                            qk_j // 4,
                            qk_j % 4,
                        ] = r_k[i]
                    if in_warp_tid == 0:
                        sBeta[i_t] = cute.arch.rcp_approx(
                            cutlass.Float32(1.0)
                            + cute.math.exp(-r_b_raw, fastmath=True)
                        )
                        if cutlass.const_expr(CACHE_RING):
                            ring_beta[slot, i_hv, ring_base + i_t] = sBeta[i_t]
                else:
                    # Issue the independent channel loads before any ring store.
                    # Gate warps do not otherwise use r_q in this phase.
                    for i in range(VEC_SIZE):
                        if cutlass.const_expr(P1ILP > 0):
                            r_q[i] = r_pix[producer_step * VEC_SIZE + i]
                        else:
                            r_q[i] = cutlass.Float32(
                                g[0, token, i_hv, i * 32 + in_warp_tid]
                            )
                    for i in range(VEC_SIZE):
                        k_idx = i * 32 + in_warp_tid
                        r_g_raw = r_q[i] + r_state[i]
                        exp_A_x = r_exp_A * r_g_raw
                        sigmoid_val = cute.arch.rcp_approx(
                            cutlass.Float32(1.0)
                            + cute.math.exp(-exp_A_x, fastmath=True)
                        )
                        r_gk = lower_bound * sigmoid_val
                        qk_grp = k_idx % P2_LANES_K
                        qk_j = k_idx // P2_LANES_K
                        sG[
                            i_t,
                            qk_grp,
                            qk_j // 4,
                            qk_j % 4,
                        ] = cute.math.exp(r_gk, fastmath=True)
                        if cutlass.const_expr(CACHE_RING):
                            ring_g[slot, i_hv, ring_base + i_t, k_idx] = r_gk
                if p1_job == 0:
                    for i in range(VEC_SIZE):
                        k_idx = i * 32 + in_warp_tid
                        for w in range(KERNEL_WIDTH - 1):
                            intermediate_conv_q[
                                scratch_row, i_t, head_off + k_idx, w
                            ] = cutlass.BFloat16(r_state[w * VEC_SIZE + i])
                elif p1_job == 1:
                    for i in range(VEC_SIZE):
                        k_idx = i * 32 + in_warp_tid
                        for w in range(KERNEL_WIDTH - 1):
                            intermediate_conv_k[
                                scratch_row, i_t, head_off + k_idx, w
                            ] = cutlass.BFloat16(
                                r_state[
                                    (KERNEL_WIDTH - 1) * VEC_SIZE + w * VEC_SIZE + i
                                ]
                            )
                # Reach token i_t + P1_JOB_WARPS. The conv body already
                # advanced one step; clamp the rest so the final iteration's
                # unread advance stays in bounds at the last request.
                if p1_job == 0:
                    for _a in range(P1_JOB_WARPS - 1):
                        _nx = bos + cutlass.min(i_t + 1 + _a, T_LOOP - 1)
                        for i in range(VEC_SIZE):
                            if cutlass.const_expr(P1ILP > 0):
                                _xn = r_pia[(producer_step * PI_ADV + _a) * VEC_SIZE + i]
                            else:
                                _xn = cutlass.Float32(
                                    x_q[0, _nx, i_hv, i * 32 + in_warp_tid]
                                )
                            r_state[0 * VEC_SIZE + i] = r_state[1 * VEC_SIZE + i]
                            r_state[1 * VEC_SIZE + i] = r_state[2 * VEC_SIZE + i]
                            r_state[2 * VEC_SIZE + i] = _xn
                elif p1_job == 1:
                    for _a in range(P1_JOB_WARPS - 1):
                        _nx = bos + cutlass.min(i_t + 1 + _a, T_LOOP - 1)
                        for i in range(VEC_SIZE):
                            if cutlass.const_expr(P1ILP > 0):
                                _xn = r_pia[(producer_step * PI_ADV + _a) * VEC_SIZE + i]
                            else:
                                _xn = cutlass.Float32(
                                    x_k[0, _nx, i_hv, i * 32 + in_warp_tid]
                                )
                            r_state[
                                (KERNEL_WIDTH - 1) * VEC_SIZE + 0 * VEC_SIZE + i
                            ] = r_state[
                                (KERNEL_WIDTH - 1) * VEC_SIZE + 1 * VEC_SIZE + i
                            ]
                            r_state[
                                (KERNEL_WIDTH - 1) * VEC_SIZE + 1 * VEC_SIZE + i
                            ] = r_state[
                                (KERNEL_WIDTH - 1) * VEC_SIZE + 2 * VEC_SIZE + i
                            ]
                            r_state[
                                (KERNEL_WIDTH - 1) * VEC_SIZE + 2 * VEC_SIZE + i
                            ] = _xn
    else:
        for _c in range(V_CH_PER_THREAD):
            _v_idx = (tidx - P1_QKG_WARPS * 32) + _c * (HEAD_DIM // V_CH_PER_THREAD)
            _csv0 = cutlass.Float32(cs_v[slot, head_off + _v_idx, 0])
            _csv1 = cutlass.Float32(cs_v[slot, head_off + _v_idx, 1])
            _csv2 = cutlass.Float32(cs_v[slot, head_off + _v_idx, 2])
            cute.autovec_copy(w_v[(head_off + _v_idx, None)], r_w4)
            _wv = [r_w4[w] for w in range(KERNEL_WIDTH)]
            # The v warps finish precompute early. Use their spare interval
            # to stage the independent onorm inputs before the existing barrier.
            if cutlass.const_expr(APPLY_ONORM):
                sOnormWeight[_v_idx] = cutlass.Float32(onorm_weight[_v_idx])
            # Sliding conv window, oldest -> newest.
            _win = [_csv0, _csv1, _csv2]
            for _t in cutlass.range_constexpr(T_LOOP):
                _win.append(
                    cutlass.Float32(
                        x_v[0, bos + cutlass.min(_t, n_tok - 1), i_hv, _v_idx]
                    )
                )
            for _t in cutlass.range_constexpr(T_LOOP):
                _vconv = _win[_t] * _wv[0]
                for _w in cutlass.range_constexpr(1, KERNEL_WIDTH):
                    _vconv += _win[_t + _w] * _wv[_w]
                _vconv = _vconv * cute.arch.rcp_approx(
                    cutlass.Float32(1.0) + cute.math.exp(-_vconv, fastmath=True)
                )
                sVall[_t * HEAD_DIM + _v_idx] = _vconv
                if cutlass.const_expr(APPLY_ONORM):
                    gate_raw = cutlass.Float32(
                        onorm_g[0, bos + cutlass.min(_t, n_tok - 1), i_hv, _v_idx]
                    )
                    sOnormGate[_t * HEAD_DIM + _v_idx] = cute.arch.rcp_approx(
                        cutlass.Float32(1.0) + cute.math.exp(-gate_raw, fastmath=True)
                    )
                if cutlass.const_expr(CACHE_RING):
                    ring_rawv[slot, i_hv, ring_base + _t, _v_idx] = cutlass.BFloat16(
                        _vconv
                    )
                for _w in cutlass.range_constexpr(KERNEL_WIDTH - 1):
                    intermediate_conv_v[scratch_row, _t, head_off + _v_idx, _w] = (
                        cutlass.BFloat16(_win[_t + 1 + _w])
                    )

    if cutlass.const_expr(FOLD):
        # Stage the previous window's records (prefetched before the conv
        # phase): k normalized in the commit's division form with eps inside
        # the sqrt, exp(gate). Phase 2's barrier publishes them to the replay.
        if warp_idx < a_prev:
            sum_rk = cutlass.Float32(0.0)
            for i in range(VEC_SIZE):
                sum_rk += r_rk[i] * r_rk[i]
            for offset in [16, 8, 4, 2, 1]:
                sum_rk += cute.arch.shuffle_sync_bfly(
                    sum_rk, offset=offset, mask=-1, mask_and_clamp=31
                )
            denom_rk = cute.math.sqrt(sum_rk + 1e-06, fastmath=True)
            for i in range(VEC_SIZE):
                k_idx = i * WARP_SIZE + in_warp_tid
                qk_grp = k_idx % P2_LANES_K
                qk_j = k_idx // P2_LANES_K
                sRK[warp_idx, qk_grp, qk_j // 4, qk_j % 4] = r_rk[i] / denom_rk
                sRG[warp_idx, qk_grp, qk_j // 4, qk_j % 4] = cute.math.exp(
                    r_rg[i], fastmath=True
                )
                sRV[warp_idx * HEAD_DIM + k_idx] = r_rv[i]
            if in_warp_tid == 0:
                sRB[warp_idx] = r_rb

    k_grp = in_warp_tid % P2_LANES_K
    row_grp = in_warp_tid // P2_LANES_K

    staged = cutlass.const_expr(STATE_STAGES < NUM_V_TILES)
    if cutlass.const_expr(not staged):
        # State-copy writers and consumers belong to the same warp.
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_warp()
    else:
        # Publish the cross-warp precompute before streamed recurrence.
        cute.arch.barrier()
    r_beta_val = cutlass.Float32(0.0)
    for i_p in cutlass.range_constexpr(NUM_V_TILES // TILES_PER_PASS):
        for i_l in cutlass.range_constexpr(TILES_PER_PASS):
            i_v = i_p * TILES_PER_PASS + i_l
            if cutlass.const_expr(staged):
                cute.arch.cp_async_wait_group(
                    min(STATE_STAGES + i_v, NUM_V_TILES) - 1 - i_v
                )
                cute.arch.sync_warp()
            for b in range(P2_BATCHES):
                _st = (i_l * P2_BATCHES + b) * P2_VEC
                _row = warp_idx * NUM_V_ROWS + b * P2_ROWS_LANE + row_grp
                for j in range(P2_VEC):
                    r_state[_st + j] = cutlass.Float32(
                        sState[
                            _row,
                            j * P2_LANES_K + k_grp,
                            i_v % STATE_STAGES,
                        ]
                    )
            if cutlass.const_expr(staged) and i_v + STATE_STAGES < NUM_V_TILES:
                cute.arch.sync_warp()
                _issue_state_tile(
                    state_g2s_copy,
                    thr_state_copy,
                    gStateTiles,
                    sState,
                    i_v + STATE_STAGES,
                    STATE_STAGES,
                )
        if cutlass.const_expr(i_p == NUM_V_TILES // TILES_PER_PASS - 1):
            if cutlass.const_expr(not staged):
                # State loads can precede the cross-warp precompute barrier.
                cute.arch.barrier()
            # Every global read this block makes has now landed in registers.
            cute.arch.griddepcontrol_launch_dependents()

        if cutlass.const_expr(FOLD):
            # Commit the previous window before verifying this one: replay its
            # a_prev accepted records into the resident state in the stock
            # commit's op order (gate decay, k-dot, delta, rank-1 update; no q,
            # no output), then store the committed state. FOLD implies the
            # resident layout (one pass holds every state tile).
            for r_t in cutlass.range(a_prev):
                r_beta_val = sRB[r_t]
                for j in range(P2_VEC):
                    r_k[j] = sRK[r_t, k_grp, j // 4, j % 4]
                    r_decay[j] = sRG[r_t, k_grp, j // 4, j % 4]
                for i_l in range(TILES_PER_PASS):
                    v_base = (i_p * TILES_PER_PASS + i_l) * TILE_V
                    for b in range(P2_BATCHES):
                        _st = (i_l * P2_BATCHES + b) * P2_VEC
                        v_row = warp_idx * NUM_V_ROWS + b * P2_ROWS_LANE + row_grp
                        r_v = sRV[r_t * HEAD_DIM + v_base + v_row]
                        shk_1 = 0.0
                        shk_2 = 0.0
                        for jp in range(P2_VEC // 2):
                            _p = jp * 2
                            r_state[_st + _p], r_state[_st + _p + 1] = (
                                cute.arch.mul_packed_f32x2(
                                    (r_decay[_p], r_decay[_p + 1]),
                                    (r_state[_st + _p], r_state[_st + _p + 1]),
                                )
                            )
                            shk_1, shk_2 = cute.arch.fma_packed_f32x2(
                                src_a=(r_state[_st + _p], r_state[_st + _p + 1]),
                                src_b=(r_k[_p], r_k[_p + 1]),
                                src_c=(shk_1, shk_2),
                            )
                        shk = shk_1 + shk_2
                        for offset in [16, 8, 4, 2, 1]:
                            if cutlass.const_expr(offset < P2_LANES_K):
                                shk += cute.arch.shuffle_sync_bfly(
                                    shk, offset=offset, mask=-1, mask_and_clamp=31
                                )
                        vnb = (r_v - shk) * r_beta_val
                        for jp in range(P2_VEC // 2):
                            _p = jp * 2
                            r_state[_st + _p], r_state[_st + _p + 1] = (
                                cute.arch.fma_packed_f32x2(
                                    src_a=(vnb, vnb),
                                    src_b=(r_k[_p], r_k[_p + 1]),
                                    src_c=(r_state[_st + _p], r_state[_st + _p + 1]),
                                )
                            )
            if a_prev > 0:
                for i_l in range(TILES_PER_PASS):
                    v_base = (i_p * TILES_PER_PASS + i_l) * TILE_V
                    for b in range(P2_BATCHES):
                        _st = (i_l * P2_BATCHES + b) * P2_VEC
                        v_row = warp_idx * NUM_V_ROWS + b * P2_ROWS_LANE + row_grp
                        for j in range(P2_VEC):
                            if cutlass.const_expr(h0.element_type.width == 32):
                                gState[v_base + v_row, j * P2_LANES_K + k_grp] = (
                                    r_state[_st + j]
                                )
                            else:
                                # BF16 state: round once (RNE), as the stock
                                # commit's store does, and verify from the
                                # stored value, so this window sees exactly
                                # the state a separate commit would leave.
                                r_h = h0.element_type(r_state[_st + j])
                                gState[v_base + v_row, j * P2_LANES_K + k_grp] = r_h
                                r_state[_st + j] = cutlass.Float32(r_h)

        for i_t in cutlass.range(T_LOOP):
            if cutlass.const_expr(STREAM_STATE):
                # ns0 uses the same q/k/decay for every state tile. Load it on
                # the first pass and keep it live across the remaining three.
                if cutlass.const_expr(i_p == 0):
                    r_beta_val = sBeta[i_t]
                    # Distinct staging tensors avoid making the three vector
                    # loads one dependent chain.
                    for jv in range(P2_VEC // 4):
                        cute.autovec_copy(sQ[(i_t, k_grp, jv, None)], r_v4q)
                        cute.autovec_copy(sK[(i_t, k_grp, jv, None)], r_v4k)
                        cute.autovec_copy(sG[(i_t, k_grp, jv, None)], r_v4g)
                        for c in range(4):
                            r_q[jv * 4 + c] = r_v4q[c]
                            r_k[jv * 4 + c] = r_v4k[c]
                            r_decay[jv * 4 + c] = r_v4g[c]
            else:
                # Read once per token, and the resident state already claims
                # every spare register, so don't pay for vector staging.
                r_beta_val = sBeta[i_t]
                for j in range(P2_VEC):
                    r_q[j] = sQ[i_t, k_grp, j // 4, j % 4]
                    r_k[j] = sK[i_t, k_grp, j // 4, j % 4]
                    r_decay[j] = sG[i_t, k_grp, j // 4, j % 4]
            for i_l in range(TILES_PER_PASS):
                v_base = (i_p * TILES_PER_PASS + i_l) * TILE_V
                for b in range(P2_BATCHES):
                    _st = (i_l * P2_BATCHES + b) * P2_VEC
                    v_row = warp_idx * NUM_V_ROWS + b * P2_ROWS_LANE + row_grp
                    # Every lane of a group wants the same v; smem broadcasts it.
                    r_v = sVall[i_t * HEAD_DIM + v_base + v_row]
                    shk_1 = 0.0
                    shk_2 = 0.0
                    for jp in range(P2_VEC // 2):
                        _p = jp * 2
                        r_state[_st + _p], r_state[_st + _p + 1] = (
                            cute.arch.mul_packed_f32x2(
                                (r_decay[_p], r_decay[_p + 1]),
                                (r_state[_st + _p], r_state[_st + _p + 1]),
                            )
                        )
                        shk_1, shk_2 = cute.arch.fma_packed_f32x2(
                            src_a=(r_state[_st + _p], r_state[_st + _p + 1]),
                            src_b=(r_k[_p], r_k[_p + 1]),
                            src_c=(shk_1, shk_2),
                        )
                    shk = shk_1 + shk_2
                    for offset in [16, 8, 4, 2, 1]:
                        if cutlass.const_expr(offset < P2_LANES_K):
                            shk += cute.arch.shuffle_sync_bfly(
                                shk, offset=offset, mask=-1, mask_and_clamp=31
                            )
                    vnb = (r_v - shk) * r_beta_val
                    shq_1 = 0.0
                    shq_2 = 0.0
                    for jp in range(P2_VEC // 2):
                        _p = jp * 2
                        r_state[_st + _p], r_state[_st + _p + 1] = (
                            cute.arch.fma_packed_f32x2(
                                src_a=(vnb, vnb),
                                src_b=(r_k[_p], r_k[_p + 1]),
                                src_c=(r_state[_st + _p], r_state[_st + _p + 1]),
                            )
                        )
                        shq_1, shq_2 = cute.arch.fma_packed_f32x2(
                            src_a=(r_state[_st + _p], r_state[_st + _p + 1]),
                            src_b=(r_q[_p], r_q[_p + 1]),
                            src_c=(shq_1, shq_2),
                        )
                    shq = shq_1 + shq_2
                    for offset in [16, 8, 4, 2, 1]:
                        if cutlass.const_expr(offset < P2_LANES_K):
                            shq += cute.arch.shuffle_sync_bfly(
                                shq, offset=offset, mask=-1, mask_and_clamp=31
                            )
                    if k_grp == 0 and i_t < n_tok:
                        if cutlass.const_expr(APPLY_ONORM):
                            sOall[i_t * HEAD_DIM + v_base + v_row] = shq
                        else:
                            o[0, bos + i_t, i_hv, v_base + v_row] = cutlass.BFloat16(
                                shq
                            )
                if cutlass.const_expr(not CACHE_RING):
                    for b in range(P2_BATCHES):
                        _st = (i_l * P2_BATCHES + b) * P2_VEC
                        v_row = warp_idx * NUM_V_ROWS + b * P2_ROWS_LANE + row_grp
                        for j in range(P2_VEC):
                            ht[
                                scratch_row,
                                i_t,
                                i_hv,
                                v_base + v_row,
                                j * P2_LANES_K + k_grp,
                            ] = r_state[_st + j]

    if cutlass.const_expr(APPLY_ONORM):
        cute.arch.barrier()
        for i_t in cutlass.range(warp_idx, T_LOOP, NUM_WARPS):
            sumsq = cutlass.Float32(0.0)
            for i in range(VEC_SIZE):
                _o = sOall[i_t * HEAD_DIM + i * 32 + in_warp_tid]
                sumsq += _o * _o
            for offset in [16, 8, 4, 2, 1]:
                sumsq += cute.arch.shuffle_sync_bfly(
                    sumsq, offset=offset, mask=-1, mask_and_clamp=31
                )
            rms = cute.math.rsqrt(
                sumsq / cutlass.Float32(HEAD_DIM) + onorm_eps, fastmath=True
            )
            for i in range(VEC_SIZE):
                v_idx = i * 32 + in_warp_tid
                raw_o = sOall[i_t * HEAD_DIM + v_idx]
                gate = sOnormGate[i_t * HEAD_DIM + v_idx]
                weight = sOnormWeight[v_idx]
                if i_t < n_tok:
                    o[0, bos + i_t, i_hv, v_idx] = cutlass.BFloat16(
                        raw_o * rms * weight * gate
                    )


@cute.jit
def _run_kda_decode_mtp_dspark(
    h0: cute.Tensor,
    x_q: cute.Tensor,
    x_k: cute.Tensor,
    x_v: cute.Tensor,
    w_q: cute.Tensor,
    w_k: cute.Tensor,
    w_v: cute.Tensor,
    cs_q: cute.Tensor,
    cs_k: cute.Tensor,
    cs_v: cute.Tensor,
    A_log: cute.Tensor,
    g: cute.Tensor,
    dt_bias: cute.Tensor,
    beta: cute.Tensor,
    out: cute.Tensor,
    intermediate_ssm: cute.Tensor,
    intermediate_state_indices: cute.Tensor,
    intermediate_conv_q: cute.Tensor,
    intermediate_conv_k: cute.Tensor,
    intermediate_conv_v: cute.Tensor,
    ssm_state_indices: cute.Tensor,
    cu_seqlens: cute.Tensor,
    ring_rawv: cute.Tensor,
    ring_rawk: cute.Tensor,
    ring_g: cute.Tensor,
    ring_beta: cute.Tensor,
    onorm_g: cute.Tensor,
    onorm_weight: cute.Tensor,
    fold_pending: cute.Tensor,
    scale: cutlass.Constexpr[float],
    H: cutlass.Constexpr[int],
    N: cutlass.Constexpr[int],
    NUM_SPEC: cutlass.Constexpr[int],
    BLOCK_THREADS: cutlass.Constexpr[int],
    lower_bound: cutlass.Constexpr[float],
    CACHE_RING: cutlass.Constexpr[bool],
    APPLY_ONORM: cutlass.Constexpr[bool],
    onorm_eps: cutlass.Constexpr[float],
    FOLD: cutlass.Constexpr[bool],
    RING_HALF: cutlass.Constexpr[int],
    P1ILP: cutlass.Constexpr[int],
    stream: cuda.CUstream,
):
    """Launch the fixed Kimi-K3/DSpARK bonus + NUM_SPEC-draft specialization."""
    p2_lanes = _p2_lanes(block_threads=BLOCK_THREADS, num_spec=NUM_SPEC)
    p2_vec = TILE_K // p2_lanes
    # (token, lane row, vector, element within vector); see _qk_smem.
    qk_stride, qk_elems = _qk_smem(block_threads=BLOCK_THREADS, num_spec=NUM_SPEC)
    smem_qk_layout = cute.make_layout(
        (1 + NUM_SPEC, p2_lanes, p2_vec // 4, 4), stride=qk_stride
    )
    # The state stages in its own dtype (FP32, or BF16 under ReplaySSM rings);
    # phase 2 widens each element to FP32 when it moves to registers.
    state_dtype = h0.element_type
    state_bits = state_dtype.width
    # Padding by the subgroup width shifts concurrent v-rows onto disjoint bank
    # groups: +8 tiles four eight-bank windows; +16 tiles two half-warps. The pad
    # is counted in 4-byte words, so a BF16 stage keeps the FP32 byte skew.
    smem_state_stride = TILE_K + p2_lanes * (32 // state_bits)
    TILE_V = _state_tile_v(block_threads=BLOCK_THREADS, num_spec=NUM_SPEC)
    state_stages = min(NUM_STATE_STAGES, TILE_K // TILE_V)
    state_stage_elems = TILE_V * smem_state_stride
    state_smem_elems = state_stages * state_stage_elems
    smem_state_layout = cute.make_layout(
        (TILE_V, TILE_K, state_stages),
        stride=(smem_state_stride, 1, state_stage_elems),
    )
    state_cache_mode = (
        cpasync.LoadCacheMode.ALWAYS
        if NUM_SPEC == 0 and BLOCK_THREADS == BLOCK_THREADS_WIDE
        else cpasync.LoadCacheMode.GLOBAL
    )
    # 128-bit copies: 4 FP32 or 8 BF16 values per thread, narrowed only if one
    # pass of a thread row would overrun TILE_K.
    state_vals = min(128 // state_bits, TILE_K // (BLOCK_THREADS // TILE_V))
    state_copy_atom = cute.make_copy_atom(
        cpasync.CopyG2SOp(cache_mode=state_cache_mode),
        state_dtype,
        num_bits_per_copy=state_vals * state_bits,
    )
    state_g2s_copy = cute.make_tiled_copy_tv(
        state_copy_atom,
        thr_layout=cute.make_layout(
            (TILE_V, BLOCK_THREADS // TILE_V),
            stride=(BLOCK_THREADS // TILE_V, 1),
        ),
        val_layout=cute.make_layout((1, state_vals)),
    )
    t_loop = 1 + NUM_SPEC
    smem_bytes = (
        # sQ, sK, sG, sBeta, and sVall.
        (3 * t_loop * qk_elems + t_loop + t_loop * TILE_K) * 4
        # State stages and the raw output tile for normalization.
        + state_smem_elems * (state_bits // 8)
        + (t_loop * TILE_K * 4 if cutlass.const_expr(APPLY_ONORM) else 0)
        + ((t_loop + 1) * TILE_K * 4 if APPLY_ONORM else 0)
        # FOLD: staged previous-window records (sRK, sRG, sRB, sRV).
        + ((2 * t_loop * qk_elems + t_loop + t_loop * TILE_K) * 4 if FOLD else 0)
        + 256
    )
    kda_decode_mtp_kernel(
        state_g2s_copy,
        h0,
        x_q,
        x_k,
        x_v,
        w_q,
        w_k,
        w_v,
        cs_q,
        cs_k,
        cs_v,
        A_log,
        g,
        dt_bias,
        beta,
        out,
        intermediate_ssm,
        intermediate_state_indices,
        intermediate_conv_q,
        intermediate_conv_k,
        intermediate_conv_v,
        ring_rawv,
        ring_rawk,
        ring_g,
        ring_beta,
        onorm_g,
        onorm_weight,
        fold_pending,
        smem_qk_layout,
        smem_state_layout,
        ssm_state_indices,
        cu_seqlens,
        scale,
        NUM_SPEC,
        BLOCK_THREADS,
        p2_lanes,
        lower_bound,
        CACHE_RING,
        APPLY_ONORM,
        onorm_eps,
        FOLD,
        RING_HALF,
        P1ILP,
    ).launch(
        grid=(H, N, 1),
        block=[BLOCK_THREADS, 1, 1],
        smem=smem_bytes,
        stream=stream,
        use_pdl=True,
    )


def _block_threads(*, H: int, N: int) -> int:
    """Pick the block width for this grid.

    The grid is (H, N). Below one wave every SM gets a single block whatever
    its width, so the wide block is free warps and roughly halves the exposed
    latency of the serial token chain. At or above one wave the narrow block
    wins instead, because two resident blocks let one issue while the other
    waits at a barrier -- warps inside a single block all wait together.

    Measured on GB300 (152 SMs, H=12, num_spec=5): N=1 -8.8%, N=8 -11.3%,
    but N=64 +47.3%.
    """
    num_sms = current_arch().num_sms
    return BLOCK_THREADS_WIDE if H * N <= num_sms else BLOCK_THREADS_NARROW


_DSPARK_COMPILED = get_jit_cache(
    "kimi_k3_kda_mtp_verify",
    source_paths=(__file__,),
    enable_tvm_ffi=False,
)


def _tensor_layout_key(tensor, fits_32bit_stride):
    return (
        tensor.device,
        tensor.dtype,
        tuple(tensor.shape),
        tuple(tensor.stride()),
        fits_32bit_stride,
    )


def _fits_32bit_stride(tensor):
    max_offset = int(tensor.storage_offset())
    for size, stride in zip(tensor.shape, tensor.stride()):
        if abs(int(stride)) > 2**31 - 1:
            return False
        if size:
            max_offset += (int(size) - 1) * abs(int(stride))
            if max_offset > 2**31 - 1:
                return False
    return True


def _cute_tensor(tensor, *, dynamic=True, fits_32bit_stride=None):
    from cutlass.cute.runtime import from_dlpack

    if tensor.requires_grad:
        tensor = tensor.detach()
    if fits_32bit_stride is None:
        fits_32bit_stride = _fits_32bit_stride(tensor)
    value = from_dlpack(
        tensor,
        assumed_align=16,
        use_32bit_stride=fits_32bit_stride,
    )
    leading_dim = next(
        (dim for dim, stride in enumerate(tensor.stride()) if stride == 1), None
    )
    if not dynamic or leading_dim is None:
        return value
    return value.mark_layout_dynamic(leading_dim)


def fused_kda_decode_mtp_dspark(
    *,
    x_q,
    x_k,
    x_v,
    w_q,
    w_k,
    w_v,
    cs_q,
    cs_k,
    cs_v,
    g,
    beta,
    A_log,
    dt_bias,
    recurrent_state,
    intermediate_ssm,
    intermediate_state_indices,
    intermediate_conv_q,
    intermediate_conv_k,
    intermediate_conv_v,
    ssm_state_indices,
    cu_seqlens,
    lower_bound,
    scale=None,
    replayssm_rawv=None,
    replayssm_rawk=None,
    replayssm_g=None,
    replayssm_beta=None,
    onorm_gate=None,
    onorm_weight=None,
    onorm_eps=None,
    fold_pending=None,
    p1ilp=None,
):
    """Run Kimi-K3 KDA verify while preserving DSpARK rollback semantics.

    Persistent recurrent/conv states are read-only.  Every post-token state and
    convolution window is written to DSpARK's existing intermediate buffers.
    The caller is responsible for enforcing the fixed dense token contract:
    every request contributes exactly 1 + num_spec tokens (num_spec ==
    --speculative-dspark-block-size), inferred here from T // N - 1.

    ReplaySSM: passing the four replayssm_* rings switches the kernel to
    CACHE_RING mode — per-step raw inputs go to the rings (consumed by the
    commit-time exact fold, see kda_replayssm_spec_decode.py) and the per-step
    intermediate_ssm state snapshots are skipped, so intermediate_ssm may be
    None.

    Passing all three onorm_* arguments fuses gated RMSNorm into the recurrence
    kernel.

    fold_pending (int32 [pool slots], ReplaySSM only): commit-then-verify. Each
    request's ``recurrent_state[slot]`` is taken as the state *before* the
    previous verify window; ``fold_pending[slot] = a | half << 8`` says its
    first ``a`` records, stored in ring half ``half`` (rows half * L/2 ...),
    were accepted. The kernel replays them (the stock commit's method), writes
    the committed state back, verifies, and writes this window's records to
    half ``1 - half``. The caller owns ``fold_pending``
    (kda_replayssm_spec_decode fold helpers).
    """
    import torch

    H = x_q.shape[2]
    N = cu_seqlens.numel() - 1
    T = x_q.shape[1]
    expected_shape = (1, T, H, TILE_K)
    if tuple(x_q.shape) != expected_shape or tuple(x_k.shape) != expected_shape:
        raise ValueError(f"expected q/k shape {expected_shape}")
    if tuple(x_v.shape) != expected_shape or tuple(g.shape) != expected_shape:
        raise ValueError(f"expected v/g shape {expected_shape}")
    if tuple(beta.shape) != (1, T, H):
        raise ValueError(f"expected beta shape {(1, T, H)}")
    # T // N == 1 is num_spec == 0: one token per request, i.e. a plain decode
    # step. The backend never dispatches here for it (that is the dedicated
    # decode kernel's job), but the layout is legal and benchmarks compare the
    # two at this point, so the wrapper accepts it.
    if N <= 0 or T % N != 0 or T // N < 1:
        raise ValueError(
            f"DSpARK KDA MTP requires a fixed 1 + num_spec dense tokens per "
            f"request; got T={T}, N={N}"
        )
    num_spec = T // N - 1
    if recurrent_state.shape[1:] != (H, TILE_K, TILE_K):
        raise ValueError("expected recurrent state layout [pool, H, V=128, K=128]")
    rings = (replayssm_rawv, replayssm_rawk, replayssm_g, replayssm_beta)
    cache_ring = all(ring is not None for ring in rings)
    # BF16 state needs the ring path: CACHE_RING skips the per-step FP32
    # snapshot stores. Without fold_pending the verify only reads the state;
    # with it, the kernel also stores the committed state, rounded once to
    # the state dtype (round to nearest even), and verifies from that value.
    state_dtypes = (torch.float32, torch.bfloat16) if cache_ring else (torch.float32,)
    align = 16 // recurrent_state.element_size()
    if (
        recurrent_state.dtype not in state_dtypes
        or tuple(recurrent_state.stride()[-3:]) != (TILE_K * TILE_K, TILE_K, 1)
        or recurrent_state.stride(0) % align != 0
        or recurrent_state.storage_offset() % align != 0
    ):
        raise ValueError(
            "cp.async recurrent state requires fp32 (or bf16 with the ReplaySSM "
            "rings) contiguous [H, V, K] inner layout and 16-byte-aligned slot "
            "offsets"
        )
    if any(ring is not None for ring in rings) and not cache_ring:
        raise ValueError("ReplaySSM requires all four replayssm_* rings")
    if cache_ring:
        ring_len = replayssm_rawv.shape[2]
        if (
            ring_len < 1 + num_spec
            or replayssm_rawv.shape[1:] != (H, ring_len, TILE_K)
            or replayssm_rawk.shape[1:] != (H, ring_len, TILE_K)
            or replayssm_g.shape[1:] != (H, ring_len, TILE_K)
            or replayssm_beta.shape[1:] != (H, ring_len)
        ):
            raise ValueError(
                f"expected ReplaySSM ring layouts [slots, H={H}, L>={1 + num_spec}, "
                f"{TILE_K}] / [slots, H, L]"
            )
        if (
            replayssm_rawv.dtype != torch.bfloat16
            or replayssm_rawk.dtype != torch.bfloat16
            or replayssm_g.dtype != torch.float32
            or replayssm_beta.dtype != torch.float32
        ):
            raise ValueError(
                "expected ReplaySSM ring dtypes rawv/rawk=bf16, g/beta=fp32"
            )
        if intermediate_ssm is None:
            # Dead-branch placeholder: CACHE_RING skips every ht snapshot
            # write, but CuTe still type-checks the rank-5 indexing.
            intermediate_ssm = recurrent_state.unsqueeze(1)
    if not cache_ring and (
        intermediate_ssm.shape[1] < 1 + num_spec
        or intermediate_ssm.shape[2:5] != (H, TILE_K, TILE_K)
    ):
        raise ValueError(
            f"expected intermediate SSM layout [scratch, >={1 + num_spec}, H, "
            f"V=128, K=128]"
        )
    if scale is None:
        scale = TILE_K**-0.5
    onorm_args = (onorm_gate, onorm_weight, onorm_eps)
    apply_onorm = all(value is not None for value in onorm_args)
    if any(value is not None for value in onorm_args) and not apply_onorm:
        raise ValueError(
            "fused output norm requires onorm_gate, onorm_weight, and onorm_eps"
        )
    if apply_onorm:
        if tuple(onorm_gate.shape) != expected_shape:
            raise ValueError(f"expected output-norm gate shape {expected_shape}")
        if tuple(onorm_weight.shape) != (TILE_K,):
            raise ValueError(f"expected output-norm weight shape {(TILE_K,)}")
        if onorm_gate.dtype != torch.bfloat16 or onorm_weight.dtype != torch.float32:
            raise ValueError("expected output-norm gate=bf16 and weight=fp32")
    fold = fold_pending is not None
    ring_half = 0
    if fold:
        if not cache_ring:
            raise ValueError("fold_pending requires the ReplaySSM rings")
        if num_spec < 1:
            raise ValueError("fold_pending requires a multi-token verify window")
        ring_half = replayssm_rawv.shape[2] // 2
        if ring_half < 1 + num_spec:
            raise ValueError(
                "fold_pending needs a ring of two verify windows (ping-pong halves)"
            )
        if (
            fold_pending.dtype != torch.int32
            or fold_pending.ndim != 1
            or not fold_pending.is_contiguous()
            or fold_pending.numel() < recurrent_state.shape[0]
            or fold_pending.device != recurrent_state.device
        ):
            raise ValueError(
                "fold_pending must be a contiguous int32 [>= pool slots] tensor "
                "on the state's device"
            )
    block_threads = _block_threads(H=H, N=N)
    if fold and 1 + num_spec > block_threads // WARP_SIZE:
        raise ValueError("fold_pending stages one record per warp")
    global LAST_P1ILP
    if p1ilp is None:
        _env_p1 = os.environ.get("K31HT_KDA_P1ILP", "").strip()
        p1ilp = int(_env_p1) if _env_p1 else 0
    if p1ilp not in (0, 1):
        raise ValueError(f"K31HT_KDA_P1ILP must be 0 or 1; got {p1ilp}")
    p1_on = p1ilp if num_spec > 0 else 0
    LAST_P1ILP = p1_on
    out = torch.empty_like(x_v)
    args = (
        recurrent_state,
        x_q,
        x_k,
        x_v,
        w_q,
        w_k,
        w_v,
        cs_q,
        cs_k,
        cs_v,
        A_log,
        g,
        dt_bias,
        beta,
        out,
        intermediate_ssm,
        intermediate_state_indices,
        intermediate_conv_q,
        intermediate_conv_k,
        intermediate_conv_v,
        ssm_state_indices,
        cu_seqlens,
        # ReplaySSM rings; same placeholder trick when CACHE_RING is off.
        replayssm_rawv if cache_ring else intermediate_conv_q,
        replayssm_rawk if cache_ring else intermediate_conv_q,
        replayssm_g if cache_ring else intermediate_conv_q,
        replayssm_beta if cache_ring else intermediate_conv_q[..., 0],
        onorm_gate if apply_onorm else x_v,
        onorm_weight if apply_onorm else dt_bias,
        # Compile-time-dead placeholder when FOLD is off.
        fold_pending if fold else cu_seqlens,
    )
    fits_32bit = tuple(_fits_32bit_stride(tensor) for tensor in args)
    cuda_arch = current_arch().cuda
    key = (
        cuda_arch.capability,
        H,
        N,
        num_spec,
        block_threads,
        cache_ring,
        apply_onorm,
        fold,
        ring_half,
        float(onorm_eps) if apply_onorm else 0.0,
        float(scale),
        float(lower_bound),
        p1_on,
        *(_tensor_layout_key(tensor, fits) for tensor, fits in zip(args, fits_32bit)),
    )
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = _DSPARK_COMPILED[key] if key in _DSPARK_COMPILED else None
    if compiled is None:
        cute_args = tuple(
            _cute_tensor(tensor, dynamic=False, fits_32bit_stride=fits)
            for tensor, fits in zip(args, fits_32bit)
        )
        compiled = cute.compile(
            _run_kda_decode_mtp_dspark,
            *cute_args,
            scale=float(scale),
            H=H,
            N=N,
            NUM_SPEC=num_spec,
            BLOCK_THREADS=block_threads,
            lower_bound=float(lower_bound),
            CACHE_RING=cache_ring,
            APPLY_ONORM=apply_onorm,
            onorm_eps=float(onorm_eps) if apply_onorm else 0.0,
            FOLD=fold,
            RING_HALF=ring_half,
            P1ILP=p1_on,
            stream=stream,
        )
        _DSPARK_COMPILED[key] = compiled
    compiled(
        *(
            _cute_tensor(tensor, fits_32bit_stride=fits)
            for tensor, fits in zip(args, fits_32bit)
        ),
        stream,
    )
    return out
