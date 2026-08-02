"""CuTe DSL device kernel KDA conv-MTP.

conv enabled, no bias, optional fused gated RMSNorm, lower_bound gate, Q/K
L2 norm, beta sigmoid, ILP=2, W=4. TILE_V is 64 with a serial two-tile
loop. Recurrent-state tiles are cp.async'd into NUM_STATE_STAGES smem stages.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import nvvm
from cutlass.cute.nvgpu import cpasync

WARP_SIZE = 32
TILE_K = 128
TILE_V = 64
SPLIT_TILE_V = 16
KERNEL_WIDTH = 4
# Serial phase-1 q/k/g jobs use equal-sized warp groups; the rest carry V.
P1_NUM_JOBS = 3
# Share of the block's warps handed to the v-conv, as a divisor: 4 -> a quarter.
# Tuned, not structural — v is one conv over V channels with no norm or gate, so
# it needs fewer warps than q/k/g, but the exact split is empirical.
P1_V_WARP_DIVISOR = 4
# Width-16 split-V warp allocation. Six warps per interleaved Q/K/G job
# balance the precise-math path while leaving one warp for V convolution.
P1_SPLIT_QK_WARPS = 6
P1_SPLIT_G_WARPS = 6
P1_SPLIT_V_WARPS = 1
# Recurrent-state smem stages, i.e. how many of the V // TILE_V state tiles are
# resident at once.
NUM_STATE_STAGES = 2

# Block size is chosen per launch, not fixed: below one wave (H*N <= SM count)
# every SM gets one block regardless, so a wider block is free warps. Above it,
# two narrow blocks per SM let one issue while the other sits at a barrier,
# which a single wide block cannot do. See _block_threads.
BLOCK_THREADS_NARROW = 256
BLOCK_THREADS_WIDE = 512
BLOCK_THREADS_SPLIT = 608
STATE_COPY_THREADS_SPLIT = 512

# K spans one recurrence lane group. The serial topology prefers eight lanes,
# whereas the final eight-CTA width-16 split-V topology prefers a full warp.
# Both choices were measured independently on B300.
P2_LANES_K_SERIAL = 8
P2_LANES_K_SPLIT = 16
P2_WARPS_SPLIT = 8

HEAD_DIM = TILE_K
VEC_SIZE = HEAD_DIM // WARP_SIZE
NUM_V_TILES = HEAD_DIM // TILE_V
NUM_SPLIT_V_CTAS = HEAD_DIM // SPLIT_TILE_V
# Conv weights live in one smem array: [W, K] for q/k, then [W, V] for v.
V_WEIGHT_BASE = KERNEL_WIDTH * HEAD_DIM
CONV_WEIGHT_ELEMS = 2 * V_WEIGHT_BASE


def _issue_state_tile(
    state_g2s_copy: cute.TiledCopy,
    thr_state_copy,
    gStateTiles: cute.Tensor,
    sState: cute.Tensor,
    i_v: int,
) -> None:
    """cp.async state tile ``i_v`` into the stage it maps to, and commit it."""
    cute.copy(
        state_g2s_copy,
        thr_state_copy.partition_S(gStateTiles[(None, None, i_v)]),
        thr_state_copy.partition_D(sState[(None, None, i_v % NUM_STATE_STAGES)]),
    )
    cute.arch.cp_async_commit_group()


def _float_float_add(
    acc_hi,
    acc_lo,
    value_hi,
    value_lo,
):
    """Accumulate two FP32 pairs without eagerly renormalizing the residual.

    The recurrent contractions contain only four lane-local products followed
    by five butterfly combines.  Keeping the exact TwoSum residual in the low
    component across that short reduction avoids a second TwoSum after every
    addition while preserving both components until the final ``hi + lo``.
    """
    sum_hi = acc_hi + value_hi
    delta = sum_hi - acc_hi
    sum_error = (acc_hi - (sum_hi - delta)) + (value_hi - delta)
    sum_lo = acc_lo + value_lo + sum_error
    return sum_hi, sum_lo


def _float_float_add_product(acc_hi, acc_lo, lhs, rhs):
    """Accumulate one FP32 product without discarding its FMA residual."""
    product = lhs * rhs
    product_error = cute.fma(lhs, rhs, -product)
    return _float_float_add(acc_hi, acc_lo, product, product_error)


def _float_float_add_packed(
    acc_hi,
    acc_lo,
    value_hi,
    value_lo,
):
    """Renormalized float-float addition for two independent packed lanes."""
    sum_hi = cute.arch.add_packed_f32x2(acc_hi, value_hi)
    delta = cute.arch.sub_packed_f32x2(sum_hi, acc_hi)
    sum_minus_delta = cute.arch.sub_packed_f32x2(sum_hi, delta)
    left_error = cute.arch.sub_packed_f32x2(acc_hi, sum_minus_delta)
    right_error = cute.arch.sub_packed_f32x2(value_hi, delta)
    sum_error = cute.arch.add_packed_f32x2(left_error, right_error)
    correction = cute.arch.add_packed_f32x2(acc_lo, value_lo)
    correction = cute.arch.add_packed_f32x2(correction, sum_error)
    renormalized = cute.arch.add_packed_f32x2(sum_hi, correction)
    renormalized_delta = cute.arch.sub_packed_f32x2(renormalized, sum_hi)
    renorm_minus_delta = cute.arch.sub_packed_f32x2(renormalized, renormalized_delta)
    left_residual = cute.arch.sub_packed_f32x2(sum_hi, renorm_minus_delta)
    right_residual = cute.arch.sub_packed_f32x2(correction, renormalized_delta)
    residual = cute.arch.add_packed_f32x2(left_residual, right_residual)
    return renormalized, residual


def _float_float_add_product_packed(acc_hi, acc_lo, lhs, rhs):
    """Packed TwoProduct followed by a renormalized packed float-float add."""
    product = cute.arch.mul_packed_f32x2(lhs, rhs)
    product_error = cute.arch.fma_packed_f32x2(
        lhs,
        rhs,
        (-product[0], -product[1]),
    )
    return _float_float_add_packed(acc_hi, acc_lo, product, product_error)


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
    onorm_partials: cute.Tensor,
    smem_qk_layout: cute.Layout,
    smem_state_layout: cute.Layout,
    ssm_state_indices: cute.Tensor,
    cu_seqlens: cute.Tensor,
    scale: cutlass.Constexpr[float],
    NUM_SPEC: cutlass.Constexpr[int],
    BLOCK_THREADS: cutlass.Constexpr[int],
    SPLIT_V: cutlass.Constexpr[bool],
    lower_bound: cutlass.Constexpr[float],
    CACHE_RING: cutlass.Constexpr[bool],
    APPLY_ONORM: cutlass.Constexpr[bool],
    onorm_eps: cutlass.Constexpr[float],
):
    """KDA MTP decode — SMEM pre-compute + register-resident state."""
    tidx, _, _ = cute.arch.thread_idx()
    in_warp_tid = tidx % WARP_SIZE
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    i_hv, i_n, i_z = cute.arch.block_idx()
    head_off = i_hv * HEAD_DIM
    T_LOOP = 1 + NUM_SPEC
    CTA_V_ROWS = SPLIT_TILE_V if cutlass.const_expr(SPLIT_V) else TILE_V
    ACTIVE_V_TILES = 1 if cutlass.const_expr(SPLIT_V) else NUM_V_TILES
    ACTIVE_V_CHANNELS = CTA_V_ROWS if cutlass.const_expr(SPLIT_V) else HEAD_DIM
    STATE_STAGES = 1 if cutlass.const_expr(SPLIT_V) else NUM_STATE_STAGES
    # Conv-precompute warp budget.
    NUM_WARPS = BLOCK_THREADS // WARP_SIZE
    P2_NUM_WARPS = P2_WARPS_SPLIT if cutlass.const_expr(SPLIT_V) else NUM_WARPS
    P1_SERIAL_V_WARPS = max(1, NUM_WARPS // P1_V_WARP_DIVISOR)
    P1_SERIAL_JOB_WARPS = (NUM_WARPS - P1_SERIAL_V_WARPS) // P1_NUM_JOBS
    P1_QK_WARPS = (
        P1_SPLIT_QK_WARPS if cutlass.const_expr(SPLIT_V) else P1_SERIAL_JOB_WARPS
    )
    P1_G_WARPS = (
        P1_SPLIT_G_WARPS if cutlass.const_expr(SPLIT_V) else P1_SERIAL_JOB_WARPS
    )
    P1_V_WARPS = P1_SPLIT_V_WARPS if cutlass.const_expr(SPLIT_V) else P1_SERIAL_V_WARPS
    P1_INTERLEAVED_WARPS = P1_NUM_JOBS * P1_QK_WARPS
    P1_QKG_WARPS = P1_INTERLEAVED_WARPS + (P1_G_WARPS - P1_QK_WARPS)
    V_CH_PER_THREAD = max(1, ACTIVE_V_CHANNELS // (P1_V_WARPS * WARP_SIZE))
    P2_LANES_K = P2_LANES_K_SPLIT if cutlass.const_expr(SPLIT_V) else P2_LANES_K_SERIAL
    P2_ROWS_LANE = WARP_SIZE // P2_LANES_K
    P2_VEC = TILE_K // P2_LANES_K
    P2_BFLY = [8, 4, 2, 1] if cutlass.const_expr(SPLIT_V) else [4, 2, 1]
    P2_SHUFFLE_CLAMP = 0x100F if cutlass.const_expr(SPLIT_V) else 31
    P2_TOKEN_UNROLL = 4 if cutlass.const_expr(SPLIT_V) else 0
    QK_SCRATCH_BASE = CONV_WEIGHT_ELEMS - T_LOOP
    NUM_V_ROWS = CTA_V_ROWS // P2_NUM_WARPS
    P2_BATCHES = NUM_V_ROWS // P2_ROWS_LANE
    smem = cutlass.utils.SmemAllocator()
    sQ = smem.allocate_tensor(cutlass.Float32, smem_qk_layout, 16)
    sK = smem.allocate_tensor(cutlass.Float32, smem_qk_layout, 16)
    sG = smem.allocate_tensor(cutlass.Float32, smem_qk_layout, 16)
    sBeta = smem.allocate_tensor(cutlass.Float32, cute.make_layout((T_LOOP,)), 16)
    sExpA = smem.allocate_tensor(cutlass.Float32, cute.make_layout((1,)), 16)
    sVall = smem.allocate_tensor(
        cutlass.Float32, cute.make_layout((T_LOOP * ACTIVE_V_CHANNELS,)), 16
    )
    sConvW = smem.allocate_tensor(
        cutlass.Float32,
        cute.make_layout((CONV_WEIGHT_ELEMS,)),
        16,
    )
    sState = smem.allocate_tensor(cutlass.Float32, smem_state_layout, 16)
    if cutlass.const_expr(APPLY_ONORM):
        sOall = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout((T_LOOP * ACTIVE_V_CHANNELS,)), 16
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
    r_bk = cute.make_rmem_tensor(
        cute.make_layout((P2_VEC,), stride=(1,)), cutlass.Float32
    )
    # Sized for whichever is larger: the active recurrent-state tiles (phase 2)
    # or the two conv windows (phase 1). The phases never overlap.
    R_STATE_ELEMS = max(
        ACTIVE_V_TILES * P2_BATCHES * P2_VEC,
        2 * (KERNEL_WIDTH - 1) * VEC_SIZE,
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

    slot = ssm_state_indices[i_n]
    # CUDA-graph padding rows use slot == -1.
    if slot < 0:
        cute.arch.griddepcontrol_wait()
        pad_bos = cu_seqlens[i_n]
        pad_v_base = i_z * SPLIT_TILE_V if cutlass.const_expr(SPLIT_V) else 0
        for i_t in cutlass.range_constexpr(T_LOOP):
            if tidx < ACTIVE_V_CHANNELS:
                o[0, pad_bos + i_t, i_hv, pad_v_base + tidx] = cutlass.BFloat16(0.0)
        cute.arch.griddepcontrol_launch_dependents()
        # nvvm.exit, not `return`: the DSL rejects an early return out of a
        # staged if (UNSUP_EARLY_EXIT).
        nvvm.exit()

    if cutlass.const_expr(not SPLIT_V):
        for i in range(VEC_SIZE):
            k_idx = i * 32 + in_warp_tid
            for w in range(KERNEL_WIDTH - 1):
                r_state[w * VEC_SIZE + i] = cutlass.Float32(
                    cs_q[slot, head_off + k_idx, w]
                )
                r_state[(KERNEL_WIDTH - 1) * VEC_SIZE + w * VEC_SIZE + i] = (
                    cutlass.Float32(cs_k[slot, head_off + k_idx, w])
                )

    if tidx < HEAD_DIM:
        cute.autovec_copy(w_k[(head_off + tidx, None)], r_w4)
        for w in range(KERNEL_WIDTH):
            sConvW[w * HEAD_DIM + tidx] = r_w4[w]
    if tidx < ACTIVE_V_CHANNELS:
        v_idx = i_z * SPLIT_TILE_V + tidx if cutlass.const_expr(SPLIT_V) else tidx
        cute.autovec_copy(w_v[(head_off + v_idx, None)], r_w4)
        for w in range(KERNEL_WIDTH):
            sConvW[V_WEIGHT_BASE + w * HEAD_DIM + v_idx] = r_w4[w]
    if warp_idx < P1_INTERLEAVED_WARPS and warp_idx % P1_NUM_JOBS == 0:
        for i in range(VEC_SIZE):
            cute.autovec_copy(w_q[(head_off + i * 32 + in_warp_tid, None)], r_w4)
            for w in range(KERNEL_WIDTH):
                r_wq[w * VEC_SIZE + i] = r_w4[w]

    # The 128-bit copy atom maps each thread to coalesced K vectors.
    gState = h0[(slot, i_hv, None, None)]
    gStateTiles = cute.local_tile(gState, (CTA_V_ROWS, TILE_K), (None, 0))
    state_copy_tid = (
        cutlass.min(tidx, STATE_COPY_THREADS_SPLIT - 1)
        if cutlass.const_expr(SPLIT_V)
        else tidx
    )
    thr_state_copy = state_g2s_copy.get_slice(state_copy_tid)

    if cutlass.const_expr(SPLIT_V):
        if tidx < STATE_COPY_THREADS_SPLIT:
            cute.copy(
                state_g2s_copy,
                thr_state_copy.partition_S(gStateTiles[(None, None, i_z)]),
                thr_state_copy.partition_D(sState[(None, None, 0)]),
            )
        cute.arch.cp_async_commit_group()
    else:
        for i_v in cutlass.range_constexpr(min(NUM_STATE_STAGES, NUM_V_TILES)):
            _issue_state_tile(state_g2s_copy, thr_state_copy, gStateTiles, sState, i_v)

    cute.arch.griddepcontrol_wait()

    bos = cu_seqlens[i_n]
    eos = cu_seqlens[i_n + 1]
    n_tok = eos - bos
    scratch_row = intermediate_state_indices[i_n]
    r_exp_A = cutlass.Float32(0.0)
    if tidx == 0:
        sExpA[0] = cute.math.exp(
            cutlass.Float32(A_log[i_hv]),
            fastmath=False,
        )

    cute.arch.barrier()

    # Split-V uses 6/6/6/1 Q/K/G/V warps; serial keeps equal groups.
    p1_job = warp_idx % P1_NUM_JOBS
    p1_par = warp_idx // P1_NUM_JOBS
    if cutlass.const_expr(SPLIT_V):
        extra_g = warp_idx >= P1_INTERLEAVED_WARPS
        p1_job = cutlass.Int32(cutlass.select_(extra_g, 2, p1_job))
        p1_par = cutlass.Int32(
            cutlass.select_(
                extra_g,
                P1_QK_WARPS + warp_idx - P1_INTERLEAVED_WARPS,
                p1_par,
            )
        )
    if warp_idx < P1_QKG_WARPS and p1_job == 2:
        r_exp_A = sExpA[0]
        for i_t in cutlass.range(p1_par, T_LOOP, P1_G_WARPS):
            token = bos + cutlass.min(i_t, n_tok - 1)
            for i in range(VEC_SIZE):
                k_idx = i * 32 + in_warp_tid
                r_g_raw = cutlass.Float32(g[0, token, i_hv, k_idx])
                r_g_raw = r_g_raw + cutlass.Float32(dt_bias[i_hv * HEAD_DIM + k_idx])
                exp_A_x = r_exp_A * r_g_raw
                sigmoid_val = cute.arch.rcp_approx(
                    cutlass.Float32(1.0) + cute.math.exp(-exp_A_x, fastmath=False)
                )
                r_gk = lower_bound * sigmoid_val
                sG[i_t, k_idx] = cute.math.exp(r_gk, fastmath=False)
                if cutlass.const_expr(CACHE_RING):
                    if cutlass.const_expr(not SPLIT_V) or i_z == 0:
                        ring_g[slot, i_hv, i_t, k_idx] = r_gk
    elif warp_idx < P1_QKG_WARPS:
        # Warp starting at token p1_par needs its window advanced that
        # many steps.
        if cutlass.const_expr(not SPLIT_V):
            if p1_job == 0:
                for _pi in cutlass.range(cutlass.min(p1_par, n_tok)):
                    for i in range(VEC_SIZE):
                        _xn = cutlass.Float32(
                            x_q[0, bos + _pi, i_hv, i * 32 + in_warp_tid]
                        )
                        r_state[0 * VEC_SIZE + i] = r_state[1 * VEC_SIZE + i]
                        r_state[1 * VEC_SIZE + i] = r_state[2 * VEC_SIZE + i]
                        r_state[2 * VEC_SIZE + i] = _xn
            elif p1_job == 1:
                for _pi in cutlass.range(cutlass.min(p1_par, n_tok)):
                    for i in range(VEC_SIZE):
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
        for i_t in cutlass.range(p1_par, T_LOOP, P1_QK_WARPS):
            token = bos + cutlass.min(i_t, n_tok - 1)
            if cutlass.const_expr(SPLIT_V):
                if p1_job == 0:
                    for i in range(VEC_SIZE):
                        k_idx = i * 32 + in_warp_tid
                        for w in range(KERNEL_WIDTH - 1):
                            history_index = i_t + w
                            history_value = cutlass.Float32(0.0)
                            if history_index < KERNEL_WIDTH - 1:
                                history_value = cutlass.Float32(
                                    cs_q[
                                        slot,
                                        head_off + k_idx,
                                        history_index,
                                    ]
                                )
                            else:
                                history_value = cutlass.Float32(
                                    x_q[
                                        0,
                                        bos + history_index - (KERNEL_WIDTH - 1),
                                        i_hv,
                                        k_idx,
                                    ]
                                )
                            r_state[w * VEC_SIZE + i] = history_value
                elif p1_job == 1:
                    for i in range(VEC_SIZE):
                        k_idx = i * 32 + in_warp_tid
                        for w in range(KERNEL_WIDTH - 1):
                            history_index = i_t + w
                            history_value = cutlass.Float32(0.0)
                            if history_index < KERNEL_WIDTH - 1:
                                history_value = cutlass.Float32(
                                    cs_k[
                                        slot,
                                        head_off + k_idx,
                                        history_index,
                                    ]
                                )
                            else:
                                history_value = cutlass.Float32(
                                    x_k[
                                        0,
                                        bos + history_index - (KERNEL_WIDTH - 1),
                                        i_hv,
                                        k_idx,
                                    ]
                                )
                            r_state[
                                (KERNEL_WIDTH - 1) * VEC_SIZE + w * VEC_SIZE + i
                            ] = history_value
            if p1_job == 0:
                for i_pair in range(VEC_SIZE // 2):
                    i0 = i_pair * 2
                    i1 = i_pair * 2 + 1
                    k_idx0 = i0 * 32 + in_warp_tid
                    k_idx1 = i1 * 32 + in_warp_tid
                    _q_state_0 = r_state[i0]
                    _q_state_1 = r_state[i1]
                    _q_weight_0 = r_wq[i0]
                    _q_weight_1 = r_wq[i1]
                    r_conv = cute.arch.mul_packed_f32x2(
                        (_q_state_0, _q_state_1),
                        (_q_weight_0, _q_weight_1),
                    )
                    r_conv_err = cute.arch.fma_packed_f32x2(
                        (_q_state_0, _q_state_1),
                        (_q_weight_0, _q_weight_1),
                        (-r_conv[0], -r_conv[1]),
                    )
                    for w in range(1, KERNEL_WIDTH - 1):
                        _q_state_0 = r_state[w * VEC_SIZE + i0]
                        _q_state_1 = r_state[w * VEC_SIZE + i1]
                        _q_weight_0 = r_wq[w * VEC_SIZE + i0]
                        _q_weight_1 = r_wq[w * VEC_SIZE + i1]
                        r_conv, r_conv_err = _float_float_add_product_packed(
                            r_conv,
                            r_conv_err,
                            (_q_state_0, _q_state_1),
                            (_q_weight_0, _q_weight_1),
                        )
                    r_xq_0 = cutlass.Float32(x_q[0, token, i_hv, k_idx0])
                    r_xq_1 = cutlass.Float32(x_q[0, token, i_hv, k_idx1])
                    _cwq_last_0 = r_wq[(KERNEL_WIDTH - 1) * VEC_SIZE + i0]
                    _cwq_last_1 = r_wq[(KERNEL_WIDTH - 1) * VEC_SIZE + i1]
                    r_conv, r_conv_err = _float_float_add_product_packed(
                        r_conv,
                        r_conv_err,
                        (r_xq_0, r_xq_1),
                        (_cwq_last_0, _cwq_last_1),
                    )
                    r_conv = cute.arch.add_packed_f32x2(r_conv, r_conv_err)
                    r_conv_0, r_conv_1 = r_conv
                    e0 = cute.math.exp(-r_conv_0, fastmath=False)
                    e1 = cute.math.exp(-r_conv_1, fastmath=False)
                    sig_0 = cute.math.rcp(cutlass.Float32(1.0) + e0)
                    sig_1 = cute.math.rcp(cutlass.Float32(1.0) + e1)
                    # Materialize the model's BF16 convolution output before KDA.
                    r_q[i0] = cutlass.Float32(cutlass.BFloat16(r_conv_0 * sig_0))
                    r_q[i1] = cutlass.Float32(cutlass.BFloat16(r_conv_1 * sig_1))
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
                rnorm_q_scaled = cute.math.rsqrt(sum_q + 1e-06, fastmath=False) * scale
                for i in range(VEC_SIZE):
                    r_q[i] = r_q[i] * rnorm_q_scaled
                for i in range(VEC_SIZE):
                    k_idx = i * 32 + in_warp_tid
                    sQ[i_t, k_idx] = r_q[i]
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
                    r_conv = r_conv * cute.math.rcp(
                        cutlass.Float32(1.0) + cute.math.exp(-r_conv, fastmath=False)
                    )
                    r_conv = cutlass.Float32(cutlass.BFloat16(r_conv))
                    r_k[i] = r_conv
                    if cutlass.const_expr(CACHE_RING):
                        if cutlass.const_expr(not SPLIT_V) or i_z == 0:
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
                rnorm_k = cute.math.rsqrt(sum_k + 1e-06, fastmath=False)
                for i in range(VEC_SIZE):
                    r_k[i] = r_k[i] * rnorm_k
                for i in range(VEC_SIZE):
                    k_idx = i * 32 + in_warp_tid
                    sK[i_t, k_idx] = r_k[i]
                if cutlass.const_expr(not SPLIT_V) and in_warp_tid == 0:
                    sBeta[i_t] = cute.math.rcp(
                        cutlass.Float32(1.0) + cute.math.exp(-r_b_raw, fastmath=False)
                    )
                    if cutlass.const_expr(CACHE_RING):
                        if cutlass.const_expr(not SPLIT_V) or i_z == 0:
                            ring_beta[slot, i_hv, i_t] = sBeta[i_t]
            # z=0 owns shared K/G/beta rings and Q/K rollback writes.
            if p1_job == 0:
                if cutlass.const_expr(not SPLIT_V) or i_z == 0:
                    for i in range(VEC_SIZE):
                        k_idx = i * 32 + in_warp_tid
                        for w in range(KERNEL_WIDTH - 1):
                            intermediate_conv_q[
                                scratch_row, i_t, head_off + k_idx, w
                            ] = cutlass.BFloat16(r_state[w * VEC_SIZE + i])
            elif p1_job == 1:
                if cutlass.const_expr(not SPLIT_V) or i_z == 0:
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
            # Reach token i_t + P1_QK_WARPS. The conv body already
            # advanced one step; clamp the rest so the final iteration's
            # unread advance stays in bounds at the last request.
            if cutlass.const_expr(not SPLIT_V):
                if p1_job == 0:
                    for _a in range(P1_QK_WARPS - 1):
                        _nx = bos + cutlass.min(i_t + 1 + _a, T_LOOP - 1)
                        for i in range(VEC_SIZE):
                            _xn = cutlass.Float32(
                                x_q[0, _nx, i_hv, i * 32 + in_warp_tid]
                            )
                            r_state[0 * VEC_SIZE + i] = r_state[1 * VEC_SIZE + i]
                            r_state[1 * VEC_SIZE + i] = r_state[2 * VEC_SIZE + i]
                            r_state[2 * VEC_SIZE + i] = _xn
                elif p1_job == 1:
                    for _a in range(P1_QK_WARPS - 1):
                        _nx = bos + cutlass.min(i_t + 1 + _a, T_LOOP - 1)
                        for i in range(VEC_SIZE):
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
        v_thread = tidx - P1_QKG_WARPS * WARP_SIZE
        if cutlass.const_expr(SPLIT_V):
            if warp_idx < P1_QKG_WARPS + P1_V_WARPS:
                # Pair each local V channel with one interleaved token stream.
                split_local_v = v_thread % SPLIT_TILE_V
                token_stream = v_thread // SPLIT_TILE_V
                num_token_streams = P1_V_WARPS * (WARP_SIZE // SPLIT_TILE_V)
                split_v_idx = i_z * SPLIT_TILE_V + split_local_v
                _split_wv = [
                    sConvW[V_WEIGHT_BASE + w * HEAD_DIM + split_v_idx]
                    for w in range(KERNEL_WIDTH)
                ]
                r_split_v_window = cute.make_rmem_tensor(
                    cute.make_layout((KERNEL_WIDTH,), stride=(1,)),
                    cutlass.Float32,
                )
                split_tokens_per_stream = T_LOOP // num_token_streams
                split_t_base = token_stream * split_tokens_per_stream
                for _split_w in cutlass.range_constexpr(KERNEL_WIDTH - 1):
                    _split_history_index = split_t_base + _split_w
                    if _split_history_index < KERNEL_WIDTH - 1:
                        r_split_v_window[_split_w] = cutlass.Float32(
                            cs_v[
                                slot,
                                head_off + split_v_idx,
                                _split_history_index,
                            ]
                        )
                    else:
                        r_split_v_window[_split_w] = cutlass.Float32(
                            x_v[
                                0,
                                bos + _split_history_index - (KERNEL_WIDTH - 1),
                                i_hv,
                                split_v_idx,
                            ]
                        )
                for _split_tg in cutlass.range_constexpr(split_tokens_per_stream):
                    _split_t = split_t_base + _split_tg
                    r_split_v_window[KERNEL_WIDTH - 1] = cutlass.Float32(
                        x_v[0, bos + _split_t, i_hv, split_v_idx]
                    )
                    _split_vconv = r_split_v_window[0] * _split_wv[0]
                    for _split_w in cutlass.range_constexpr(1, KERNEL_WIDTH):
                        _split_vconv += r_split_v_window[_split_w] * _split_wv[_split_w]
                    _split_vconv = _split_vconv * cute.arch.rcp_approx(
                        cutlass.Float32(1.0)
                        + cute.math.exp(-_split_vconv, fastmath=True)
                    )
                    _split_vconv = cutlass.Float32(cutlass.BFloat16(_split_vconv))
                    sVall[_split_t * ACTIVE_V_CHANNELS + split_local_v] = _split_vconv
                    if cutlass.const_expr(CACHE_RING):
                        ring_rawv[slot, i_hv, _split_t, split_v_idx] = cutlass.BFloat16(
                            _split_vconv
                        )
                    for _split_w in cutlass.range_constexpr(KERNEL_WIDTH - 1):
                        intermediate_conv_v[
                            scratch_row,
                            _split_t,
                            head_off + split_v_idx,
                            _split_w,
                        ] = cutlass.BFloat16(r_split_v_window[_split_w + 1])
                    for _split_w in cutlass.range_constexpr(KERNEL_WIDTH - 1):
                        r_split_v_window[_split_w] = r_split_v_window[_split_w + 1]
        else:
            for _c in range(V_CH_PER_THREAD):
                local_v = v_thread + _c * (ACTIVE_V_CHANNELS // V_CH_PER_THREAD)
                v_idx = local_v
                _csv0 = cutlass.Float32(cs_v[slot, head_off + v_idx, 0])
                _csv1 = cutlass.Float32(cs_v[slot, head_off + v_idx, 1])
                _csv2 = cutlass.Float32(cs_v[slot, head_off + v_idx, 2])
                _wv = [
                    sConvW[V_WEIGHT_BASE + w * HEAD_DIM + v_idx]
                    for w in range(KERNEL_WIDTH)
                ]
                # Sliding conv window, oldest -> newest.
                _win = [_csv0, _csv1, _csv2]
                for _t in cutlass.range_constexpr(T_LOOP):
                    _win.append(
                        cutlass.Float32(
                            x_v[
                                0,
                                bos + cutlass.min(_t, n_tok - 1),
                                i_hv,
                                v_idx,
                            ]
                        )
                    )
                for _t in cutlass.range_constexpr(T_LOOP):
                    _vconv = _win[_t] * _wv[0]
                    for _w in cutlass.range_constexpr(1, KERNEL_WIDTH):
                        _vconv += _win[_t + _w] * _wv[_w]
                    _vconv = _vconv * cute.arch.rcp_approx(
                        cutlass.Float32(1.0) + cute.math.exp(-_vconv, fastmath=True)
                    )
                    _vconv = cutlass.Float32(cutlass.BFloat16(_vconv))
                    sVall[_t * ACTIVE_V_CHANNELS + local_v] = _vconv
                    if cutlass.const_expr(CACHE_RING):
                        ring_rawv[slot, i_hv, _t, v_idx] = cutlass.BFloat16(_vconv)
                    for _w in cutlass.range_constexpr(KERNEL_WIDTH - 1):
                        intermediate_conv_v[scratch_row, _t, head_off + v_idx, _w] = (
                            cutlass.BFloat16(_win[_t + 1 + _w])
                        )

    k_grp = in_warp_tid % P2_LANES_K
    row_grp = in_warp_tid // P2_LANES_K

    staged = cutlass.const_expr(STATE_STAGES < ACTIVE_V_TILES)
    if cutlass.const_expr(not staged):
        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()
    for i_v in cutlass.range_constexpr(ACTIVE_V_TILES):
        if cutlass.const_expr(staged):
            cute.arch.cp_async_wait_group(
                min(STATE_STAGES + i_v, ACTIVE_V_TILES) - 1 - i_v
            )
            cute.arch.barrier()
        if cutlass.const_expr(not SPLIT_V) or warp_idx < P2_NUM_WARPS:
            for b in range(P2_BATCHES):
                for j in range(P2_VEC):
                    r_state[(i_v * P2_BATCHES + b) * P2_VEC + j] = cutlass.Float32(
                        sState[
                            warp_idx * NUM_V_ROWS + b * P2_ROWS_LANE + row_grp,
                            j * P2_LANES_K + k_grp,
                            i_v % STATE_STAGES,
                        ]
                    )
        if cutlass.const_expr(staged) and i_v + STATE_STAGES < ACTIVE_V_TILES:
            cute.arch.barrier()
            _issue_state_tile(
                state_g2s_copy,
                thr_state_copy,
                gStateTiles,
                sState,
                i_v + STATE_STAGES,
            )

    # Split norm still needs its invocation-local scratch.
    if cutlass.const_expr(not (APPLY_ONORM and SPLIT_V)):
        cute.arch.griddepcontrol_launch_dependents()

    if cutlass.const_expr(SPLIT_V):
        # One warp per token computes q·k once. A spare warp computes all beta
        # gates in parallel with those reductions, removing their special
        # functions from the phase-1 K-warp critical path. The recurrent row
        # warps then use o = (D·q) + alpha(q·k), avoiding a second post-update
        # contraction.
        if warp_idx < T_LOOP:
            qk_1 = cutlass.Float32(0.0)
            qk_2 = cutlass.Float32(0.0)
            for jp in range(VEC_SIZE // 2):
                j0 = jp * 2
                j1 = jp * 2 + 1
                k_idx0 = j0 * WARP_SIZE + in_warp_tid
                k_idx1 = j1 * WARP_SIZE + in_warp_tid
                qk_1, qk_2 = cute.arch.fma_packed_f32x2(
                    src_a=(sQ[warp_idx, k_idx0], sQ[warp_idx, k_idx1]),
                    src_b=(sK[warp_idx, k_idx0], sK[warp_idx, k_idx1]),
                    src_c=(qk_1, qk_2),
                )
            qk = qk_1 + qk_2
            for offset in [16, 8, 4, 2, 1]:
                qk += cute.arch.shuffle_sync_bfly(
                    qk, offset=offset, mask=-1, mask_and_clamp=31
                )
            if in_warp_tid == 0:
                sConvW[QK_SCRATCH_BASE + warp_idx] = qk
        elif warp_idx == T_LOOP and in_warp_tid < T_LOOP:
            beta_t = in_warp_tid
            beta_tok = bos + cutlass.min(beta_t, n_tok - 1)
            beta_raw = cutlass.Float32(beta[0, beta_tok, i_hv])
            sBeta[beta_t] = cute.math.rcp(
                cutlass.Float32(1.0) + cute.math.exp(-beta_raw, fastmath=False)
            )
            if cutlass.const_expr(CACHE_RING):
                if i_z == 0:
                    ring_beta[slot, i_hv, beta_t] = sBeta[beta_t]
        cute.arch.barrier()

    if cutlass.const_expr(APPLY_ONORM and SPLIT_V):
        # The upper eight warps are free during the two-row recurrence. Use
        # them to move gated-RMSNorm special-function work off the cluster
        # synchronization tail. sConvW is dead after phase 1.
        if warp_idx >= P2_NUM_WARPS:
            norm_thread = tidx - P2_NUM_WARPS * WARP_SIZE
            if norm_thread < T_LOOP * SPLIT_TILE_V:
                norm_t = norm_thread // SPLIT_TILE_V
                norm_local_v = norm_thread % SPLIT_TILE_V
                norm_v_idx = i_z * SPLIT_TILE_V + norm_local_v
                norm_tok = bos + cutlass.min(norm_t, n_tok - 1)
                norm_gate_raw = cutlass.Float32(onorm_g[0, norm_tok, i_hv, norm_v_idx])
                sConvW[norm_thread] = cute.arch.rcp_approx(
                    cutlass.Float32(1.0) + cute.math.exp(-norm_gate_raw, fastmath=True)
                )
                if norm_t == 0:
                    sConvW[T_LOOP * SPLIT_TILE_V + norm_local_v] = cutlass.Float32(
                        onorm_weight[norm_v_idx]
                    )

    if cutlass.const_expr(not SPLIT_V) or warp_idx < P2_NUM_WARPS:
        for i_t in cutlass.range(T_LOOP, unroll=P2_TOKEN_UNROLL):
            r_beta_val = sBeta[i_t]
            for jp in range(P2_VEC // 2):
                j0 = jp * 2
                j1 = jp * 2 + 1
                k_idx0 = j0 * P2_LANES_K + k_grp
                k_idx1 = j1 * P2_LANES_K + k_grp
                r_q[j0] = sQ[i_t, k_idx0]
                r_q[j1] = sQ[i_t, k_idx1]
                _k0 = sK[i_t, k_idx0]
                _k1 = sK[i_t, k_idx1]
                r_decay[j0] = sG[i_t, k_idx0]
                r_decay[j1] = sG[i_t, k_idx1]
                r_bk[j0] = _k0
                r_bk[j1] = _k1
                if cutlass.const_expr(not SPLIT_V):
                    r_k[j0], r_k[j1] = cute.arch.mul_packed_f32x2(
                        (r_decay[j0], r_decay[j1]), (_k0, _k1)
                    )
            for i_v in range(ACTIVE_V_TILES):
                v_base = (
                    i_z * SPLIT_TILE_V if cutlass.const_expr(SPLIT_V) else i_v * TILE_V
                )
                local_v_base = i_v * CTA_V_ROWS
                for b in range(P2_BATCHES):
                    _st = (i_v * P2_BATCHES + b) * P2_VEC
                    v_row = warp_idx * NUM_V_ROWS + b * P2_ROWS_LANE + row_grp
                    # Every lane of a group wants the same v; smem broadcasts it.
                    r_v = sVall[i_t * ACTIVE_V_CHANNELS + local_v_base + v_row]
                    if cutlass.const_expr(SPLIT_V):
                        for jp in range(P2_VEC // 2):
                            _p = jp * 2
                            r_state[_st + _p], r_state[_st + _p + 1] = (
                                cute.arch.mul_packed_f32x2(
                                    (r_decay[_p], r_decay[_p + 1]),
                                    (
                                        r_state[_st + _p],
                                        r_state[_st + _p + 1],
                                    ),
                                )
                            )
                        shk_1 = cutlass.Float32(0.0)
                        shk_2 = cutlass.Float32(0.0)
                        shk_3 = cutlass.Float32(0.0)
                        shk_4 = cutlass.Float32(0.0)
                        shq_1 = cutlass.Float32(0.0)
                        shq_2 = cutlass.Float32(0.0)
                        shq_3 = cutlass.Float32(0.0)
                        shq_4 = cutlass.Float32(0.0)
                        for jp in range(P2_VEC // 4):
                            _p = jp * 4
                            shk_1, shk_2 = cute.arch.fma_packed_f32x2(
                                src_a=(
                                    r_state[_st + _p],
                                    r_state[_st + _p + 1],
                                ),
                                src_b=(r_bk[_p], r_bk[_p + 1]),
                                src_c=(shk_1, shk_2),
                            )
                            shk_3, shk_4 = cute.arch.fma_packed_f32x2(
                                src_a=(
                                    r_state[_st + _p + 2],
                                    r_state[_st + _p + 3],
                                ),
                                src_b=(r_bk[_p + 2], r_bk[_p + 3]),
                                src_c=(shk_3, shk_4),
                            )
                            shq_1, shq_2 = cute.arch.fma_packed_f32x2(
                                src_a=(
                                    r_state[_st + _p],
                                    r_state[_st + _p + 1],
                                ),
                                src_b=(r_q[_p], r_q[_p + 1]),
                                src_c=(shq_1, shq_2),
                            )
                            shq_3, shq_4 = cute.arch.fma_packed_f32x2(
                                src_a=(
                                    r_state[_st + _p + 2],
                                    r_state[_st + _p + 3],
                                ),
                                src_b=(r_q[_p + 2], r_q[_p + 3]),
                                src_c=(shq_3, shq_4),
                            )
                        shk = (shk_1 + shk_3) + (shk_2 + shk_4)
                        shq = (shq_1 + shq_3) + (shq_2 + shq_4)
                    else:
                        shk_1 = cutlass.Float32(0.0)
                        shk_2 = cutlass.Float32(0.0)
                        for jp in range(P2_VEC // 2):
                            _p = jp * 2
                            shk_1, shk_2 = cute.arch.fma_packed_f32x2(
                                src_a=(
                                    r_state[_st + _p],
                                    r_state[_st + _p + 1],
                                ),
                                src_b=(r_k[_p], r_k[_p + 1]),
                                src_c=(shk_1, shk_2),
                            )
                        shk = shk_1 + shk_2
                    for offset in P2_BFLY:
                        shk += cute.arch.shuffle_sync_bfly(
                            shk,
                            offset=offset,
                            mask=-1,
                            mask_and_clamp=P2_SHUFFLE_CLAMP,
                        )
                        if cutlass.const_expr(SPLIT_V):
                            shq += cute.arch.shuffle_sync_bfly(
                                shq,
                                offset=offset,
                                mask=-1,
                                mask_and_clamp=P2_SHUFFLE_CLAMP,
                            )
                    vn = r_v - shk
                    vn_beta = vn * r_beta_val
                    if cutlass.const_expr(not SPLIT_V):
                        shq_1 = cutlass.Float32(0.0)
                        shq_2 = cutlass.Float32(0.0)
                    for jp in range(P2_VEC // 2):
                        _p = jp * 2
                        if cutlass.const_expr(SPLIT_V):
                            r_state[_st + _p], r_state[_st + _p + 1] = (
                                cute.arch.fma_packed_f32x2(
                                    src_a=(vn_beta, vn_beta),
                                    src_b=(r_bk[_p], r_bk[_p + 1]),
                                    src_c=(
                                        r_state[_st + _p],
                                        r_state[_st + _p + 1],
                                    ),
                                )
                            )
                        else:
                            vnbk_0, vnbk_1 = cute.arch.mul_packed_f32x2(
                                (vn_beta, vn_beta),
                                (r_bk[_p], r_bk[_p + 1]),
                            )
                            r_state[_st + _p], r_state[_st + _p + 1] = (
                                cute.arch.fma_packed_f32x2(
                                    src_a=(r_decay[_p], r_decay[_p + 1]),
                                    src_b=(
                                        r_state[_st + _p],
                                        r_state[_st + _p + 1],
                                    ),
                                    src_c=(vnbk_0, vnbk_1),
                                )
                            )
                        if cutlass.const_expr(not SPLIT_V):
                            shq_1, shq_2 = cute.arch.fma_packed_f32x2(
                                src_a=(
                                    r_state[_st + _p],
                                    r_state[_st + _p + 1],
                                ),
                                src_b=(r_q[_p], r_q[_p + 1]),
                                src_c=(shq_1, shq_2),
                            )
                    if cutlass.const_expr(SPLIT_V):
                        shq = cute.fma(
                            vn_beta,
                            sConvW[QK_SCRATCH_BASE + i_t],
                            shq,
                        )
                    else:
                        shq = shq_1 + shq_2
                        for offset in P2_BFLY:
                            shq += cute.arch.shuffle_sync_bfly(
                                shq,
                                offset=offset,
                                mask=-1,
                                mask_and_clamp=P2_SHUFFLE_CLAMP,
                            )
                    if k_grp == 0 and i_t < n_tok:
                        if cutlass.const_expr(APPLY_ONORM):
                            sOall[i_t * ACTIVE_V_CHANNELS + local_v_base + v_row] = shq
                        else:
                            o[0, bos + i_t, i_hv, v_base + v_row] = cutlass.BFloat16(
                                shq
                            )
            if cutlass.const_expr(not CACHE_RING):
                for i_v in range(ACTIVE_V_TILES):
                    v_base = (
                        i_z * SPLIT_TILE_V
                        if cutlass.const_expr(SPLIT_V)
                        else i_v * TILE_V
                    )
                    for b in range(P2_BATCHES):
                        _st = (i_v * P2_BATCHES + b) * P2_VEC
                        v_row = warp_idx * NUM_V_ROWS + b * P2_ROWS_LANE + row_grp
                        for j in range(P2_VEC):
                            ht[
                                scratch_row,
                                i_t,
                                i_hv,
                                v_base + v_row,
                                j * P2_LANES_K + k_grp,
                            ] = r_state[_st + j]

    if cutlass.const_expr(APPLY_ONORM and SPLIT_V):
        # Reduce RMS statistics across the CTA cluster.
        cute.arch.barrier()
        for i_t in cutlass.range(warp_idx, T_LOOP, NUM_WARPS):
            partial_sumsq = cutlass.Float32(0.0)
            if in_warp_tid < SPLIT_TILE_V:
                raw_o = sOall[i_t * ACTIVE_V_CHANNELS + in_warp_tid]
                partial_sumsq = raw_o * raw_o
            for offset in [8, 4, 2, 1]:
                partial_sumsq += cute.arch.shuffle_sync_bfly(
                    partial_sumsq, offset=offset, mask=-1, mask_and_clamp=31
                )
            if in_warp_tid == 0:
                onorm_partials[i_hv, i_z, i_t] = partial_sumsq

        cute.arch.barrier()
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()

        for i_t in cutlass.range(warp_idx, T_LOOP, NUM_WARPS):
            sumsq = cutlass.Float32(0.0)
            for i_peer in range(NUM_SPLIT_V_CTAS):
                sumsq += onorm_partials[i_hv, i_peer, i_t]
            rms = cute.math.rsqrt(
                sumsq / cutlass.Float32(HEAD_DIM) + onorm_eps, fastmath=True
            )
            if in_warp_tid < SPLIT_TILE_V:
                local_v = in_warp_tid
                v_idx = i_z * SPLIT_TILE_V + local_v
                raw_o = sOall[i_t * ACTIVE_V_CHANNELS + local_v]
                gate = sConvW[i_t * SPLIT_TILE_V + local_v]
                norm_weight = sConvW[T_LOOP * SPLIT_TILE_V + local_v]
                if i_t < n_tok:
                    o[0, bos + i_t, i_hv, v_idx] = cutlass.BFloat16(
                        raw_o * rms * norm_weight * gate
                    )
        # Reconverge before the block-level PDL trigger.
        cute.arch.barrier()
        cute.arch.griddepcontrol_launch_dependents()
    elif cutlass.const_expr(APPLY_ONORM):
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
    onorm_partials: cute.Tensor,
    scale: cutlass.Constexpr[float],
    H: cutlass.Constexpr[int],
    N: cutlass.Constexpr[int],
    NUM_SPEC: cutlass.Constexpr[int],
    BLOCK_THREADS: cutlass.Constexpr[int],
    SPLIT_V: cutlass.Constexpr[bool],
    lower_bound: cutlass.Constexpr[float],
    CACHE_RING: cutlass.Constexpr[bool],
    APPLY_ONORM: cutlass.Constexpr[bool],
    onorm_eps: cutlass.Constexpr[float],
    stream: cuda.CUstream,
):
    """Launch the fixed Kimi-K3/DSpARK bonus + NUM_SPEC-draft specialization."""
    smem_qk_layout = cute.make_layout((1 + NUM_SPEC, TILE_K), stride=(TILE_K, 1))
    p2_lanes_k = P2_LANES_K_SPLIT if cutlass.const_expr(SPLIT_V) else P2_LANES_K_SERIAL
    # The split-V topology assigns a full warp to each 128-wide state row, so
    # its naturally aligned stride is already bank-conflict free.  Padding is
    # useful only for the serial eight-lane recurrence topology.
    smem_state_stride = TILE_K if cutlass.const_expr(SPLIT_V) else TILE_K + p2_lanes_k
    state_tile_v = SPLIT_TILE_V if cutlass.const_expr(SPLIT_V) else TILE_V
    serial_state_stages = min(NUM_STATE_STAGES, TILE_K // TILE_V)
    state_stages = 1 if cutlass.const_expr(SPLIT_V) else serial_state_stages
    smem_state_layout = cute.make_layout(
        (state_tile_v, TILE_K, state_stages),
        stride=(smem_state_stride, 1, state_tile_v * smem_state_stride),
    )
    state_copy_atom = cute.make_copy_atom(
        cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
        cutlass.Float32,
        num_bits_per_copy=128,
    )
    state_copy_threads = (
        STATE_COPY_THREADS_SPLIT if cutlass.const_expr(SPLIT_V) else BLOCK_THREADS
    )
    state_g2s_copy = cute.make_tiled_copy_tv(
        state_copy_atom,
        thr_layout=cute.make_layout(
            (state_tile_v, state_copy_threads // state_tile_v),
            stride=(state_copy_threads // state_tile_v, 1),
        ),
        val_layout=cute.make_layout((1, 4)),
    )
    t_loop = 1 + NUM_SPEC
    active_v_channels = state_tile_v if cutlass.const_expr(SPLIT_V) else HEAD_DIM
    smem_bytes = (
        # sQ, sK, sG, sBeta, sVall, and sConvW.
        (3 * t_loop * TILE_K + t_loop + 1 + t_loop * active_v_channels + 8 * TILE_K) * 4
        # State stages and the raw output tile for normalization.
        + state_stages * state_tile_v * smem_state_stride * 4
        + (t_loop * active_v_channels * 4 if cutlass.const_expr(APPLY_ONORM) else 0)
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
        onorm_partials,
        smem_qk_layout,
        smem_state_layout,
        ssm_state_indices,
        cu_seqlens,
        scale,
        NUM_SPEC,
        BLOCK_THREADS,
        SPLIT_V,
        lower_bound,
        CACHE_RING,
        APPLY_ONORM,
        onorm_eps,
    ).launch(
        grid=(H, N, NUM_SPLIT_V_CTAS if cutlass.const_expr(SPLIT_V) else 1),
        block=[BLOCK_THREADS, 1, 1],
        cluster=(
            (1, 1, NUM_SPLIT_V_CTAS)
            if cutlass.const_expr(SPLIT_V and APPLY_ONORM)
            else None
        ),
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


_DSPARK_COMPILED = {}


def _tensor_layout_key(tensor):
    return (tensor.device, tensor.dtype, tuple(tensor.shape), tuple(tensor.stride()))


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


def _cute_tensor(tensor, *, dynamic=True):
    from cutlass.cute.runtime import from_dlpack

    if tensor.requires_grad:
        tensor = tensor.detach()
    value = from_dlpack(
        tensor,
        assumed_align=16,
        use_32bit_stride=_fits_32bit_stride(tensor),
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
    split_v=False,
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

    split_v selects the N=1/T=16 eight-CTA specialization.

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
    if N <= 0 or T % N != 0 or T // N < 2:
        raise ValueError(
            f"DSpARK KDA MTP requires a fixed 1 + num_spec dense tokens per "
            f"request; got T={T}, N={N}"
        )
    num_spec = T // N - 1
    if split_v and (N != 1 or 1 + num_spec != 16):
        raise ValueError("split_v requires exactly N=1 and 16 verify tokens")
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
    block_threads = BLOCK_THREADS_SPLIT if split_v else _block_threads(H=H, N=N)
    out = torch.empty_like(x_v)
    onorm_partials = (
        torch.empty((H, NUM_SPLIT_V_CTAS, T), dtype=torch.float32, device=x_v.device)
        if apply_onorm and split_v
        else A_log.reshape(1, 1, -1)
    )
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
        onorm_partials,
    )
    key = (
        H,
        N,
        num_spec,
        block_threads,
        split_v,
        cache_ring,
        apply_onorm,
        float(onorm_eps) if apply_onorm else 0.0,
        float(scale),
        float(lower_bound),
        *(_tensor_layout_key(tensor) for tensor in args),
    )
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = _DSPARK_COMPILED.get(key)
    if compiled is None:
        cute_args = tuple(_cute_tensor(tensor, dynamic=False) for tensor in args)
        compiled = cute.compile(
            _run_kda_decode_mtp_dspark,
            *cute_args,
            scale=float(scale),
            H=H,
            N=N,
            NUM_SPEC=num_spec,
            BLOCK_THREADS=block_threads,
            SPLIT_V=split_v,
            lower_bound=float(lower_bound),
            CACHE_RING=cache_ring,
            APPLY_ONORM=apply_onorm,
            onorm_eps=float(onorm_eps) if apply_onorm else 0.0,
            stream=stream,
        )
        _DSPARK_COMPILED[key] = compiled
    compiled(
        *(_cute_tensor(tensor) for tensor in args),
        stream,
    )
    return out
