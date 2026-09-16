"""CuTe DSL device kernel KDA conv-MTP.

conv enabled, no bias, optional fused gated RMSNorm, lower_bound gate, Q/K
L2 norm, beta sigmoid, ILP=2, W=4. Recurrent-state tiles are cp.async'd into
NUM_STATE_STAGES smem stages. Phase 2 walks the V // TILE_V state tiles in
passes of TILES_PER_PASS: a multi-token verify keeps them all register-resident
across the token chain, a single-token step streams one at a time.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import nvvm
from cutlass.cute.nvgpu import cpasync

from sglang.kernels.jit.cute_aot_cache import get_jit_cache

WARP_SIZE = 32
TILE_K = 128
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
# Conv weights live in one smem array: [W, K] for q/k, then [W, V] for v.
V_WEIGHT_BASE = KERNEL_WIDTH * HEAD_DIM
CONV_WEIGHT_ELEMS = 2 * V_WEIGHT_BASE


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
):
    """KDA MTP decode — SMEM pre-compute + register-resident state.

    One block owns all 128 value rows, so APPLY_ONORM can reduce the RMS
    denominator without cross-block synchronization.
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
    sConvW = smem.allocate_tensor(
        cutlass.Float32,
        cute.make_layout((CONV_WEIGHT_ELEMS,)),
        16,
    )
    sState = smem.allocate_tensor(cutlass.Float32, smem_state_layout, 16)
    if cutlass.const_expr(APPLY_ONORM):
        sOall = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout((T_LOOP * HEAD_DIM,)), 16
        )
    else:
        # Compile-time-dead placeholder; avoids charging the non-norm path
        # for an output tile it never touches.
        sOall = sVall
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

    if tidx < HEAD_DIM:
        cute.autovec_copy(w_k[(head_off + tidx, None)], r_w4)
        for w in range(KERNEL_WIDTH):
            sConvW[w * HEAD_DIM + tidx] = r_w4[w]
    if tidx < HEAD_DIM:
        cute.autovec_copy(w_v[(head_off + tidx, None)], r_w4)
        for w in range(KERNEL_WIDTH):
            sConvW[V_WEIGHT_BASE + w * HEAD_DIM + tidx] = r_w4[w]
    if warp_idx < P1_QKG_WARPS and warp_idx % P1_NUM_JOBS == 0:
        for i in range(VEC_SIZE):
            cute.autovec_copy(w_q[(head_off + i * 32 + in_warp_tid, None)], r_w4)
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

    cute.arch.griddepcontrol_wait()

    bos = cu_seqlens[i_n]
    eos = cu_seqlens[i_n + 1]
    n_tok = eos - bos
    scratch_row = intermediate_state_indices[i_n]
    r_exp_A = cutlass.Float32(0.0)

    cute.arch.barrier()

    if warp_idx < P1_QKG_WARPS:
        if p1_job == 2:
            r_exp_A = cute.math.exp(cutlass.Float32(A_log[i_hv]), fastmath=True)
        # Warp starting at token p1_par needs its window advanced that
        # many steps; the g path is pointwise and needs none.
        if p1_job == 0:
            for _pi in cutlass.range(cutlass.min(p1_par, n_tok)):
                for i in range(VEC_SIZE):
                    _xn = cutlass.Float32(x_q[0, bos + _pi, i_hv, i * 32 + in_warp_tid])
                    r_state[0 * VEC_SIZE + i] = r_state[1 * VEC_SIZE + i]
                    r_state[1 * VEC_SIZE + i] = r_state[2 * VEC_SIZE + i]
                    r_state[2 * VEC_SIZE + i] = _xn
        elif p1_job == 1:
            for _pi in cutlass.range(cutlass.min(p1_par, n_tok)):
                for i in range(VEC_SIZE):
                    _xn = cutlass.Float32(x_k[0, bos + _pi, i_hv, i * 32 + in_warp_tid])
                    r_state[(KERNEL_WIDTH - 1) * VEC_SIZE + 0 * VEC_SIZE + i] = r_state[
                        (KERNEL_WIDTH - 1) * VEC_SIZE + 1 * VEC_SIZE + i
                    ]
                    r_state[(KERNEL_WIDTH - 1) * VEC_SIZE + 1 * VEC_SIZE + i] = r_state[
                        (KERNEL_WIDTH - 1) * VEC_SIZE + 2 * VEC_SIZE + i
                    ]
                    r_state[(KERNEL_WIDTH - 1) * VEC_SIZE + 2 * VEC_SIZE + i] = _xn
        for i_t in cutlass.range(p1_par, T_LOOP, P1_JOB_WARPS):
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
                        r_conv_0 += r_state[w * VEC_SIZE + i0] * r_wq[w * VEC_SIZE + i0]
                        r_conv_1 += r_state[w * VEC_SIZE + i1] * r_wq[w * VEC_SIZE + i1]
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
                rnorm_q_scaled = cute.math.rsqrt(sum_q + 1e-06, fastmath=True) * scale
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
                    r_b_raw = cutlass.Float32(beta[0, token, i_hv])
                for i in range(VEC_SIZE):
                    k_idx = i * 32 + in_warp_tid
                    r_conv = (
                        r_state[(KERNEL_WIDTH - 1) * VEC_SIZE + 0 * VEC_SIZE + i]
                        * sConvW[0 * HEAD_DIM + i * 32 + in_warp_tid]
                    )
                    r_conv += (
                        r_state[(KERNEL_WIDTH - 1) * VEC_SIZE + 1 * VEC_SIZE + i]
                        * sConvW[1 * HEAD_DIM + i * 32 + in_warp_tid]
                    )
                    r_conv += (
                        r_state[(KERNEL_WIDTH - 1) * VEC_SIZE + 2 * VEC_SIZE + i]
                        * sConvW[2 * HEAD_DIM + i * 32 + in_warp_tid]
                    )
                    r_xk = cutlass.Float32(x_k[0, token, i_hv, k_idx])
                    r_conv += (
                        r_xk
                        * sConvW[(KERNEL_WIDTH - 1) * HEAD_DIM + i * 32 + in_warp_tid]
                    )
                    r_conv = r_conv * cute.arch.rcp_approx(
                        cutlass.Float32(1.0) + cute.math.exp(-r_conv, fastmath=True)
                    )
                    r_k[i] = r_conv
                    if cutlass.const_expr(CACHE_RING):
                        ring_rawk[slot, i_hv, i_t, k_idx] = cutlass.BFloat16(r_conv)
                    r_state[(KERNEL_WIDTH - 1) * VEC_SIZE + 0 * VEC_SIZE + i] = r_state[
                        (KERNEL_WIDTH - 1) * VEC_SIZE + 1 * VEC_SIZE + i
                    ]
                    r_state[(KERNEL_WIDTH - 1) * VEC_SIZE + 1 * VEC_SIZE + i] = r_state[
                        (KERNEL_WIDTH - 1) * VEC_SIZE + 2 * VEC_SIZE + i
                    ]
                    r_state[(KERNEL_WIDTH - 1) * VEC_SIZE + 2 * VEC_SIZE + i] = r_xk
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
                        cutlass.Float32(1.0) + cute.math.exp(-r_b_raw, fastmath=True)
                    )
                    if cutlass.const_expr(CACHE_RING):
                        ring_beta[slot, i_hv, i_t] = sBeta[i_t]
            else:
                for i in range(VEC_SIZE):
                    k_idx = i * 32 + in_warp_tid
                    r_g_raw = cutlass.Float32(g[0, token, i_hv, k_idx])
                    r_g_raw = r_g_raw + cutlass.Float32(
                        dt_bias[i_hv * HEAD_DIM + k_idx]
                    )
                    exp_A_x = r_exp_A * r_g_raw
                    sigmoid_val = cute.arch.rcp_approx(
                        cutlass.Float32(1.0) + cute.math.exp(-exp_A_x, fastmath=True)
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
                        ring_g[slot, i_hv, i_t, k_idx] = r_gk
            if p1_job == 0:
                for i in range(VEC_SIZE):
                    k_idx = i * 32 + in_warp_tid
                    for w in range(KERNEL_WIDTH - 1):
                        intermediate_conv_q[scratch_row, i_t, head_off + k_idx, w] = (
                            cutlass.BFloat16(r_state[w * VEC_SIZE + i])
                        )
            elif p1_job == 1:
                for i in range(VEC_SIZE):
                    k_idx = i * 32 + in_warp_tid
                    for w in range(KERNEL_WIDTH - 1):
                        intermediate_conv_k[scratch_row, i_t, head_off + k_idx, w] = (
                            cutlass.BFloat16(
                                r_state[
                                    (KERNEL_WIDTH - 1) * VEC_SIZE + w * VEC_SIZE + i
                                ]
                            )
                        )
            # Reach token i_t + P1_JOB_WARPS. The conv body already
            # advanced one step; clamp the rest so the final iteration's
            # unread advance stays in bounds at the last request.
            if p1_job == 0:
                for _a in range(P1_JOB_WARPS - 1):
                    _nx = bos + cutlass.min(i_t + 1 + _a, T_LOOP - 1)
                    for i in range(VEC_SIZE):
                        _xn = cutlass.Float32(x_q[0, _nx, i_hv, i * 32 + in_warp_tid])
                        r_state[0 * VEC_SIZE + i] = r_state[1 * VEC_SIZE + i]
                        r_state[1 * VEC_SIZE + i] = r_state[2 * VEC_SIZE + i]
                        r_state[2 * VEC_SIZE + i] = _xn
            elif p1_job == 1:
                for _a in range(P1_JOB_WARPS - 1):
                    _nx = bos + cutlass.min(i_t + 1 + _a, T_LOOP - 1)
                    for i in range(VEC_SIZE):
                        _xn = cutlass.Float32(x_k[0, _nx, i_hv, i * 32 + in_warp_tid])
                        r_state[(KERNEL_WIDTH - 1) * VEC_SIZE + 0 * VEC_SIZE + i] = (
                            r_state[(KERNEL_WIDTH - 1) * VEC_SIZE + 1 * VEC_SIZE + i]
                        )
                        r_state[(KERNEL_WIDTH - 1) * VEC_SIZE + 1 * VEC_SIZE + i] = (
                            r_state[(KERNEL_WIDTH - 1) * VEC_SIZE + 2 * VEC_SIZE + i]
                        )
                        r_state[(KERNEL_WIDTH - 1) * VEC_SIZE + 2 * VEC_SIZE + i] = _xn
    else:
        for _c in range(V_CH_PER_THREAD):
            _v_idx = (tidx - P1_QKG_WARPS * 32) + _c * (HEAD_DIM // V_CH_PER_THREAD)
            _csv0 = cutlass.Float32(cs_v[slot, head_off + _v_idx, 0])
            _csv1 = cutlass.Float32(cs_v[slot, head_off + _v_idx, 1])
            _csv2 = cutlass.Float32(cs_v[slot, head_off + _v_idx, 2])
            _wv = [
                sConvW[V_WEIGHT_BASE + w * HEAD_DIM + _v_idx]
                for w in range(KERNEL_WIDTH)
            ]
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
                if cutlass.const_expr(CACHE_RING):
                    ring_rawv[slot, i_hv, _t, _v_idx] = cutlass.BFloat16(_vconv)
                for _w in cutlass.range_constexpr(KERNEL_WIDTH - 1):
                    intermediate_conv_v[scratch_row, _t, head_off + _v_idx, _w] = (
                        cutlass.BFloat16(_win[_t + 1 + _w])
                    )

    k_grp = in_warp_tid % P2_LANES_K
    row_grp = in_warp_tid // P2_LANES_K

    staged = cutlass.const_expr(STATE_STAGES < NUM_V_TILES)
    if cutlass.const_expr(not staged):
        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()
    r_beta_val = cutlass.Float32(0.0)
    for i_p in cutlass.range_constexpr(NUM_V_TILES // TILES_PER_PASS):
        for i_l in cutlass.range_constexpr(TILES_PER_PASS):
            i_v = i_p * TILES_PER_PASS + i_l
            if cutlass.const_expr(staged):
                cute.arch.cp_async_wait_group(
                    min(STATE_STAGES + i_v, NUM_V_TILES) - 1 - i_v
                )
                cute.arch.barrier()
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
                cute.arch.barrier()
                _issue_state_tile(
                    state_g2s_copy,
                    thr_state_copy,
                    gStateTiles,
                    sState,
                    i_v + STATE_STAGES,
                    STATE_STAGES,
                )
        if cutlass.const_expr(i_p == NUM_V_TILES // TILES_PER_PASS - 1):
            # Every global read this block makes has now landed in registers.
            cute.arch.griddepcontrol_launch_dependents()

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
                _tok = bos + cutlass.min(i_t, n_tok - 1)
                gate_raw = cutlass.Float32(onorm_g[0, _tok, i_hv, v_idx])
                gate = cute.arch.rcp_approx(
                    cutlass.Float32(1.0) + cute.math.exp(-gate_raw, fastmath=True)
                )
                if i_t < n_tok:
                    o[0, bos + i_t, i_hv, v_idx] = cutlass.BFloat16(
                        raw_o * rms * cutlass.Float32(onorm_weight[v_idx]) * gate
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
    scale: cutlass.Constexpr[float],
    H: cutlass.Constexpr[int],
    N: cutlass.Constexpr[int],
    NUM_SPEC: cutlass.Constexpr[int],
    BLOCK_THREADS: cutlass.Constexpr[int],
    lower_bound: cutlass.Constexpr[float],
    CACHE_RING: cutlass.Constexpr[bool],
    APPLY_ONORM: cutlass.Constexpr[bool],
    onorm_eps: cutlass.Constexpr[float],
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
    # Padding by the subgroup width shifts concurrent v-rows onto disjoint bank
    # groups: +8 tiles four eight-bank windows; +16 tiles two half-warps.
    smem_state_stride = TILE_K + p2_lanes
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
    state_copy_atom = cute.make_copy_atom(
        cpasync.CopyG2SOp(cache_mode=state_cache_mode),
        cutlass.Float32,
        num_bits_per_copy=128,
    )
    state_g2s_copy = cute.make_tiled_copy_tv(
        state_copy_atom,
        thr_layout=cute.make_layout(
            (TILE_V, BLOCK_THREADS // TILE_V),
            stride=(BLOCK_THREADS // TILE_V, 1),
        ),
        val_layout=cute.make_layout((1, 4)),
    )
    t_loop = 1 + NUM_SPEC
    smem_bytes = (
        # sQ, sK, sG, sBeta, sVall, and sConvW.
        (3 * t_loop * qk_elems + t_loop + t_loop * TILE_K + 8 * TILE_K) * 4
        # State stages and the raw output tile for normalization.
        + state_smem_elems * 4
        + (t_loop * TILE_K * 4 if cutlass.const_expr(APPLY_ONORM) else 0)
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
    import torch

    num_sms = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count
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
    if (
        recurrent_state.dtype != torch.float32
        or tuple(recurrent_state.stride()[-3:]) != (TILE_K * TILE_K, TILE_K, 1)
        or recurrent_state.stride(0) % 4 != 0
        or recurrent_state.storage_offset() % 4 != 0
    ):
        raise ValueError(
            "cp.async recurrent state requires fp32 contiguous [H, V, K] "
            "inner layout and 16-byte-aligned slot offsets"
        )
    rings = (replayssm_rawv, replayssm_rawk, replayssm_g, replayssm_beta)
    cache_ring = all(ring is not None for ring in rings)
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
    block_threads = _block_threads(H=H, N=N)
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
    )
    fits_32bit = tuple(_fits_32bit_stride(tensor) for tensor in args)
    key = (
        torch.cuda.get_device_capability(),
        H,
        N,
        num_spec,
        block_threads,
        cache_ring,
        apply_onorm,
        float(onorm_eps) if apply_onorm else 0.0,
        float(scale),
        float(lower_bound),
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
