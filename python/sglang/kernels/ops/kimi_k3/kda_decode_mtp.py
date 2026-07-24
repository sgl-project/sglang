"""CuTe DSL device kernel KDA conv-MTP.

conv enabled, no bias, optional fused gated RMSNorm, lower_bound gate, Q/K
L2 norm, beta sigmoid, ILP=2, W=4. TILE_V is 64 with a serial two-tile
loop. Recurrent-state tiles use a two-stage cp.async pipeline.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync

NUM_THREADS = 256
TILE_K = 128
TILE_V = 64
KERNEL_WIDTH = 4
NUM_STATE_STAGES = 2


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
    lower_bound: cutlass.Constexpr[float],
    USE_SETMAXREG: cutlass.Constexpr[bool],
    CACHE_RING: cutlass.Constexpr[bool],
    APPLY_ONORM: cutlass.Constexpr[bool],
    onorm_eps: cutlass.Constexpr[float],
):
    """KDA MTP decode — SMEM pre-compute + register-resident state.

    PDL lets immutable recurrent-state and model-weight preloads overlap the
    predecessor, then orders activation and conv-state reads before releasing
    dependents.

    One block owns all 128 value rows, so APPLY_ONORM can reduce the RMS
    denominator without cross-block synchronization.
    """
    tidx, _, _ = cute.arch.thread_idx()
    in_warp_tid = tidx % 32
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    i_hv, i_n, _ = cute.arch.block_idx()
    K = TILE_K
    V = TILE_K
    hk_off = i_hv * K
    hv_off = i_hv * V
    T_loop = 1 + NUM_SPEC
    vec_size = TILE_K // 32
    num_v_tiles = V // TILE_V
    NUM_V_ROWS = TILE_V // (NUM_THREADS // 32)
    v_weight_elems = KERNEL_WIDTH * V
    v_weight_base = KERNEL_WIDTH * K
    conv_weight_elems = KERNEL_WIDTH * K + v_weight_elems
    smem = cutlass.utils.SmemAllocator()
    sQ = smem.allocate_tensor(cutlass.Float32, smem_qk_layout, 16)
    sK = smem.allocate_tensor(cutlass.Float32, smem_qk_layout, 16)
    sG = smem.allocate_tensor(cutlass.Float32, smem_qk_layout, 16)
    sBeta = smem.allocate_tensor(cutlass.Float32, cute.make_layout((T_loop,)), 16)
    sWarpSum = smem.allocate_tensor(cutlass.Float32, cute.make_layout((8,)), 16)
    sVall = smem.allocate_tensor(cutlass.Float32, cute.make_layout((T_loop * V,)), 16)
    sConvW = smem.allocate_tensor(
        cutlass.Float32,
        cute.make_layout((conv_weight_elems,)),
        16,
    )
    sState = smem.allocate_tensor(cutlass.Float32, smem_state_layout, 16)
    if cutlass.const_expr(APPLY_ONORM):
        sOall = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout((T_loop * V,)), 16
        )
    else:
        # Compile-time-dead placeholder; avoids charging the non-norm path
        # for an output tile it never touches.
        sOall = sVall
    r_q = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32
    )
    r_k = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32
    )
    r_decay = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32
    )
    r_bk = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32
    )
    r_state = cute.make_rmem_tensor(
        cute.make_layout((NUM_V_ROWS * vec_size,), stride=(1,)), cutlass.Float32
    )
    r_wq = cute.make_rmem_tensor(
        cute.make_layout((KERNEL_WIDTH * vec_size,), stride=(1,)), cutlass.Float32
    )

    slot = ssm_state_indices[i_n]
    # CUDA-graph padding rows use slot == -1. Keep the original slot for
    # guarded ReplaySSM writes, but form all unconditional source addresses
    # from a valid discard row. In particular, cp.async does not have Python's
    # negative-index wrapping and would otherwise read before the state pool.
    read_slot = cutlass.max(slot, 0)

    # Persistent recurrent state and its slot mapping are read-only during
    # verify, so start the first 32 KiB tile before the PDL wait and overlap it
    # with the immutable model-weight preload below. The 128-bit copy atom maps
    # each thread to coalesced K vectors.
    gState = h0[(read_slot, i_hv, None, None)]
    gStateTiles = cute.local_tile(gState, (TILE_V, TILE_K), (None, 0))
    thr_state_copy = state_g2s_copy.get_slice(tidx)
    gStateFirst = gStateTiles[(None, None, 0)]
    sStateFirst = sState[(None, None, 0)]
    cute.copy(
        state_g2s_copy,
        thr_state_copy.partition_S(gStateFirst),
        thr_state_copy.partition_D(sStateFirst),
    )
    cute.arch.cp_async_commit_group()

    for w in range(KERNEL_WIDTH):
        if tidx < K:
            sConvW[w * K + tidx] = cutlass.Float32(w_k[hk_off + tidx, w])
    for ld in range(V * KERNEL_WIDTH // NUM_THREADS):
        flat = ld * NUM_THREADS + tidx
        sConvW[v_weight_base + flat] = cutlass.Float32(
            w_v[hv_off + flat % V, flat // V]
        )
    if warp_idx == 0:
        for w in range(KERNEL_WIDTH):
            for i in range(vec_size):
                r_wq[w * vec_size + i] = cutlass.Float32(
                    w_q[hk_off + i * 32 + in_warp_tid, w]
                )
    cute.arch.griddepcontrol_wait()

    bos = cu_seqlens[i_n]
    eos = cu_seqlens[i_n + 1]
    scratch_row = intermediate_state_indices[i_n]
    r_exp_A = cutlass.Float32(0.0)

    for i in range(vec_size):
        k_idx = i * 32 + in_warp_tid
        for w in range(KERNEL_WIDTH - 1):
            r_state[w * vec_size + i] = cutlass.Float32(
                cs_q[read_slot, hk_off + k_idx, w]
            )
            r_state[(KERNEL_WIDTH - 1) * vec_size + w * vec_size + i] = cutlass.Float32(
                cs_k[read_slot, hk_off + k_idx, w]
            )
    cute.arch.barrier()
    if cutlass.const_expr(USE_SETMAXREG):
        cute.arch.warpgroup_reg_dealloc(64)

    if warp_idx < 3:
        if warp_idx == 2:
            r_exp_A = cute.math.exp(cutlass.Float32(A_log[i_hv]), fastmath=True)
        i_t = 0
        while i_t < T_loop:
            token = bos + i_t
            if warp_idx == 0:
                for i_pair in range(vec_size // 2):
                    i0 = i_pair * 2
                    i1 = i_pair * 2 + 1
                    k_idx0 = i0 * 32 + in_warp_tid
                    k_idx1 = i1 * 32 + in_warp_tid
                    r_conv_0 = 0.0
                    r_conv_1 = 0.0
                    for w in range(KERNEL_WIDTH - 1):
                        r_conv_0 += r_state[w * vec_size + i0] * r_wq[w * vec_size + i0]
                        r_conv_1 += r_state[w * vec_size + i1] * r_wq[w * vec_size + i1]
                    r_xq_0 = cutlass.Float32(x_q[0, token, i_hv, k_idx0])
                    r_xq_1 = cutlass.Float32(x_q[0, token, i_hv, k_idx1])
                    _cwq_last_0 = r_wq[(KERNEL_WIDTH - 1) * vec_size + i0]
                    _cwq_last_1 = r_wq[(KERNEL_WIDTH - 1) * vec_size + i1]
                    r_conv_0 += r_xq_0 * _cwq_last_0
                    r_conv_1 += r_xq_1 * _cwq_last_1
                    e0 = cute.math.exp(-r_conv_0, fastmath=True)
                    e1 = cute.math.exp(-r_conv_1, fastmath=True)
                    sig_0 = cute.arch.rcp_approx(cutlass.Float32(1.0) + e0)
                    sig_1 = cute.arch.rcp_approx(cutlass.Float32(1.0) + e1)
                    r_q[i0] = r_conv_0 * sig_0
                    r_q[i1] = r_conv_1 * sig_1
                    r_state[0 * vec_size + i0] = r_state[1 * vec_size + i0]
                    r_state[0 * vec_size + i1] = r_state[1 * vec_size + i1]
                    r_state[1 * vec_size + i0] = r_state[2 * vec_size + i0]
                    r_state[1 * vec_size + i1] = r_state[2 * vec_size + i1]
                    r_state[2 * vec_size + i0] = r_xq_0
                    r_state[2 * vec_size + i1] = r_xq_1
                sum_q = 0.0
                for i in range(vec_size):
                    sum_q += r_q[i] * r_q[i]
                for offset in [16, 8, 4, 2, 1]:
                    sum_q += cute.arch.shuffle_sync_bfly(
                        sum_q, offset=offset, mask=-1, mask_and_clamp=31
                    )
                rnorm_q_scaled = cute.math.rsqrt(sum_q + 1e-06, fastmath=True) * scale
                for i in range(vec_size):
                    r_q[i] = r_q[i] * rnorm_q_scaled
                for i in range(vec_size):
                    k_idx = i * 32 + in_warp_tid
                    sQ[i_t, k_idx] = r_q[i]
            elif warp_idx == 1:
                r_b_raw = cutlass.Float32(0.0)
                if in_warp_tid == 0:
                    r_b_raw = cutlass.Float32(beta[0, token, i_hv])
                for i in range(vec_size):
                    k_idx = i * 32 + in_warp_tid
                    r_conv = (
                        r_state[(KERNEL_WIDTH - 1) * vec_size + 0 * vec_size + i]
                        * sConvW[0 * K + i * 32 + in_warp_tid]
                    )
                    r_conv += (
                        r_state[(KERNEL_WIDTH - 1) * vec_size + 1 * vec_size + i]
                        * sConvW[1 * K + i * 32 + in_warp_tid]
                    )
                    r_conv += (
                        r_state[(KERNEL_WIDTH - 1) * vec_size + 2 * vec_size + i]
                        * sConvW[2 * K + i * 32 + in_warp_tid]
                    )
                    r_xk = cutlass.Float32(x_k[0, token, i_hv, k_idx])
                    r_conv += (
                        r_xk * sConvW[(KERNEL_WIDTH - 1) * K + i * 32 + in_warp_tid]
                    )
                    r_conv = r_conv * cute.arch.rcp_approx(
                        cutlass.Float32(1.0) + cute.math.exp(-r_conv, fastmath=True)
                    )
                    r_k[i] = r_conv
                    if cutlass.const_expr(CACHE_RING):
                        # slot -1 marks cuda-graph padding rows
                        # (same guard as the Triton CACHE_RING).
                        if slot >= 0:
                            ring_rawk[slot, i_hv, i_t, k_idx] = cutlass.BFloat16(r_conv)
                    r_state[(KERNEL_WIDTH - 1) * vec_size + 0 * vec_size + i] = r_state[
                        (KERNEL_WIDTH - 1) * vec_size + 1 * vec_size + i
                    ]
                    r_state[(KERNEL_WIDTH - 1) * vec_size + 1 * vec_size + i] = r_state[
                        (KERNEL_WIDTH - 1) * vec_size + 2 * vec_size + i
                    ]
                    r_state[(KERNEL_WIDTH - 1) * vec_size + 2 * vec_size + i] = r_xk
                sum_k = 0.0
                for i in range(vec_size):
                    sum_k += r_k[i] * r_k[i]
                for offset in [16, 8, 4, 2, 1]:
                    sum_k += cute.arch.shuffle_sync_bfly(
                        sum_k, offset=offset, mask=-1, mask_and_clamp=31
                    )
                rnorm_k = cute.math.rsqrt(sum_k + 1e-06, fastmath=True)
                for i in range(vec_size):
                    r_k[i] = r_k[i] * rnorm_k
                for i in range(vec_size):
                    k_idx = i * 32 + in_warp_tid
                    sK[i_t, k_idx] = r_k[i]
                if in_warp_tid == 0:
                    sBeta[i_t] = cute.arch.rcp_approx(
                        cutlass.Float32(1.0) + cute.math.exp(-r_b_raw, fastmath=True)
                    )
                    if cutlass.const_expr(CACHE_RING):
                        if slot >= 0:
                            ring_beta[slot, i_hv, i_t] = sBeta[i_t]
            else:
                for i in range(vec_size):
                    k_idx = i * 32 + in_warp_tid
                    r_g_raw = cutlass.Float32(g[0, token, i_hv, k_idx])
                    r_g_raw = r_g_raw + cutlass.Float32(dt_bias[i_hv * K + k_idx])
                    exp_A_x = r_exp_A * r_g_raw
                    sigmoid_val = cute.arch.rcp_approx(
                        cutlass.Float32(1.0) + cute.math.exp(-exp_A_x, fastmath=True)
                    )
                    r_gk = lower_bound * sigmoid_val
                    sG[i_t, k_idx] = cute.math.exp(r_gk, fastmath=True)
                    if cutlass.const_expr(CACHE_RING):
                        if slot >= 0:
                            ring_g[slot, i_hv, i_t, k_idx] = r_gk
            if warp_idx == 0:
                for i in range(vec_size):
                    k_idx = i * 32 + in_warp_tid
                    for w in range(KERNEL_WIDTH - 1):
                        intermediate_conv_q[scratch_row, i_t, hk_off + k_idx, w] = (
                            cutlass.BFloat16(r_state[w * vec_size + i])
                        )
            elif warp_idx == 1:
                for i in range(vec_size):
                    k_idx = i * 32 + in_warp_tid
                    for w in range(KERNEL_WIDTH - 1):
                        intermediate_conv_k[scratch_row, i_t, hk_off + k_idx, w] = (
                            cutlass.BFloat16(
                                r_state[
                                    (KERNEL_WIDTH - 1) * vec_size + w * vec_size + i
                                ]
                            )
                        )
            i_t = i_t + 1
    else:
        _v_idx = tidx - 96
        if _v_idx < V:
            _csv0 = cutlass.Float32(cs_v[read_slot, hv_off + _v_idx, 0])
            _csv1 = cutlass.Float32(cs_v[read_slot, hv_off + _v_idx, 1])
            _csv2 = cutlass.Float32(cs_v[read_slot, hv_off + _v_idx, 2])
            _wv = [sConvW[v_weight_base + w * V + _v_idx] for w in range(KERNEL_WIDTH)]
            # Sliding conv window, oldest -> newest.
            _win = [_csv0, _csv1, _csv2]
            for _t in cutlass.range_constexpr(T_loop):
                _win.append(cutlass.Float32(x_v[0, bos + _t, i_hv, _v_idx]))
            for _t in cutlass.range_constexpr(T_loop):
                _vconv = _win[_t] * _wv[0]
                for _w in cutlass.range_constexpr(1, KERNEL_WIDTH):
                    _vconv += _win[_t + _w] * _wv[_w]
                _vconv = _vconv * cute.arch.rcp_approx(
                    cutlass.Float32(1.0) + cute.math.exp(-_vconv, fastmath=True)
                )
                sVall[_t * V + _v_idx] = _vconv
                if cutlass.const_expr(CACHE_RING):
                    if slot >= 0:
                        ring_rawv[slot, i_hv, _t, _v_idx] = cutlass.BFloat16(_vconv)
                for _w in cutlass.range_constexpr(KERNEL_WIDTH - 1):
                    intermediate_conv_v[scratch_row, _t, hv_off + _v_idx, _w] = (
                        cutlass.BFloat16(_win[_t + 1 + _w])
                    )
    # Queue the second serial V tile behind the first. Waiting for one
    # outstanding group below makes tile 0 visible while tile 1 continues
    # loading during tile-0 recurrence.
    gStateNext = gStateTiles[(None, None, 1)]
    sStateNext = sState[(None, None, 1)]
    cute.copy(
        state_g2s_copy,
        thr_state_copy.partition_S(gStateNext),
        thr_state_copy.partition_D(sStateNext),
    )
    cute.arch.cp_async_commit_group()

    # PTX requires warpgroup synchronization between consecutive setmaxnreg calls.
    cute.arch.barrier()
    if cutlass.const_expr(USE_SETMAXREG):
        cute.arch.warpgroup_reg_alloc(72)
    for i_v in range(num_v_tiles):
        v_base = i_v * TILE_V
        state_stage = i_v

        if i_v == 0:
            cute.arch.cp_async_wait_group(1)
        else:
            cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()

        for row in range(NUM_V_ROWS):
            v_row = warp_idx * NUM_V_ROWS + row
            for i in range(vec_size):
                r_state[row * vec_size + i] = cutlass.Float32(
                    sState[v_row, i * 32 + in_warp_tid, state_stage]
                )

        # A dependent launch may update persistent state. Release it only
        # after this block has consumed the last outstanding cp.async read.
        if i_v + 1 == num_v_tiles:
            cute.arch.griddepcontrol_launch_dependents()

        r_v_val = cutlass.Float32(0.0)
        i_t = 0
        while i_t < T_loop:
            if in_warp_tid < NUM_V_ROWS:
                v_idx = v_base + warp_idx * NUM_V_ROWS + in_warp_tid
                r_v_val = sVall[i_t * V + v_idx]
            r_beta_val = sBeta[i_t]
            for i_pair in range(vec_size // 2):
                i0 = i_pair * 2
                i1 = i_pair * 2 + 1
                k_idx0 = i0 * 32 + in_warp_tid
                k_idx1 = i1 * 32 + in_warp_tid
                r_q[i0] = sQ[i_t, k_idx0]
                r_q[i1] = sQ[i_t, k_idx1]
                _k0 = sK[i_t, k_idx0]
                _k1 = sK[i_t, k_idx1]
                r_decay[i0] = sG[i_t, k_idx0]
                r_decay[i1] = sG[i_t, k_idx1]
                r_bk[i0], r_bk[i1] = cute.arch.mul_packed_f32x2(
                    (r_beta_val, r_beta_val), (_k0, _k1)
                )
                r_k[i0], r_k[i1] = cute.arch.mul_packed_f32x2(
                    (r_decay[i0], r_decay[i1]), (_k0, _k1)
                )
            for row_pair in range(NUM_V_ROWS // 2):
                ra = row_pair * 2
                rb = row_pair * 2 + 1
                r_va = cute.arch.shuffle_sync(r_v_val, ra)
                r_vb = cute.arch.shuffle_sync(r_v_val, rb)
                shk_a1 = 0.0
                shk_a2 = 0.0
                shk_b1 = 0.0
                shk_b2 = 0.0
                for _pi in range(vec_size // 2):
                    _p = _pi * 2
                    shk_a1, shk_a2 = cute.arch.fma_packed_f32x2(
                        src_a=(
                            r_state[ra * vec_size + _p],
                            r_state[ra * vec_size + _p + 1],
                        ),
                        src_b=(r_k[_p], r_k[_p + 1]),
                        src_c=(shk_a1, shk_a2),
                    )
                    shk_b1, shk_b2 = cute.arch.fma_packed_f32x2(
                        src_a=(
                            r_state[rb * vec_size + _p],
                            r_state[rb * vec_size + _p + 1],
                        ),
                        src_b=(r_k[_p], r_k[_p + 1]),
                        src_c=(shk_b1, shk_b2),
                    )
                shk_a = shk_a1 + shk_a2
                shk_b = shk_b1 + shk_b2
                for offset in [16, 8, 4, 2, 1]:
                    shk_a += cute.arch.shuffle_sync_bfly(
                        shk_a, offset=offset, mask=-1, mask_and_clamp=31
                    )
                    shk_b += cute.arch.shuffle_sync_bfly(
                        shk_b, offset=offset, mask=-1, mask_and_clamp=31
                    )
                vn_a = r_va - shk_a
                vn_b = r_vb - shk_b
                shq_a1 = 0.0
                shq_a2 = 0.0
                shq_b1 = 0.0
                shq_b2 = 0.0
                for _pi in range(vec_size // 2):
                    _p = _pi * 2
                    vnbk_a0, vnbk_a1 = cute.arch.mul_packed_f32x2(
                        (vn_a, vn_a), (r_bk[_p], r_bk[_p + 1])
                    )
                    vnbk_b0, vnbk_b1 = cute.arch.mul_packed_f32x2(
                        (vn_b, vn_b), (r_bk[_p], r_bk[_p + 1])
                    )
                    r_state[ra * vec_size + _p], r_state[ra * vec_size + _p + 1] = (
                        cute.arch.fma_packed_f32x2(
                            src_a=(r_decay[_p], r_decay[_p + 1]),
                            src_b=(
                                r_state[ra * vec_size + _p],
                                r_state[ra * vec_size + _p + 1],
                            ),
                            src_c=(vnbk_a0, vnbk_a1),
                        )
                    )
                    r_state[rb * vec_size + _p], r_state[rb * vec_size + _p + 1] = (
                        cute.arch.fma_packed_f32x2(
                            src_a=(r_decay[_p], r_decay[_p + 1]),
                            src_b=(
                                r_state[rb * vec_size + _p],
                                r_state[rb * vec_size + _p + 1],
                            ),
                            src_c=(vnbk_b0, vnbk_b1),
                        )
                    )
                    shq_a1, shq_a2 = cute.arch.fma_packed_f32x2(
                        src_a=(
                            r_state[ra * vec_size + _p],
                            r_state[ra * vec_size + _p + 1],
                        ),
                        src_b=(r_q[_p], r_q[_p + 1]),
                        src_c=(shq_a1, shq_a2),
                    )
                    shq_b1, shq_b2 = cute.arch.fma_packed_f32x2(
                        src_a=(
                            r_state[rb * vec_size + _p],
                            r_state[rb * vec_size + _p + 1],
                        ),
                        src_b=(r_q[_p], r_q[_p + 1]),
                        src_c=(shq_b1, shq_b2),
                    )
                shq_a = shq_a1 + shq_a2
                shq_b = shq_b1 + shq_b2
                for offset in [16, 8, 4, 2, 1]:
                    shq_a += cute.arch.shuffle_sync_bfly(
                        shq_a, offset=offset, mask=-1, mask_and_clamp=31
                    )
                    shq_b += cute.arch.shuffle_sync_bfly(
                        shq_b, offset=offset, mask=-1, mask_and_clamp=31
                    )
                if in_warp_tid == 0:
                    v_row_a = warp_idx * NUM_V_ROWS + ra
                    v_row_b = warp_idx * NUM_V_ROWS + rb
                    if cutlass.const_expr(APPLY_ONORM):
                        sOall[i_t * V + v_base + v_row_a] = shq_a
                        sOall[i_t * V + v_base + v_row_b] = shq_b
                    else:
                        o[0, bos + i_t, i_hv, v_base + v_row_a] = cutlass.BFloat16(
                            shq_a
                        )
                        o[0, bos + i_t, i_hv, v_base + v_row_b] = cutlass.BFloat16(
                            shq_b
                        )
            if cutlass.const_expr(not CACHE_RING):
                for row in range(NUM_V_ROWS):
                    v_row = warp_idx * NUM_V_ROWS + row
                    for i in range(vec_size):
                        ht[
                            scratch_row,
                            i_t,
                            i_hv,
                            v_base + v_row,
                            i * 32 + in_warp_tid,
                        ] = r_state[row * vec_size + i]
            i_t = i_t + 1

    if cutlass.const_expr(APPLY_ONORM):
        cute.arch.barrier()
        for i_t in cutlass.range_constexpr(T_loop):
            raw_o = cutlass.Float32(0.0)
            local_sumsq = cutlass.Float32(0.0)
            if tidx < V:
                raw_o = sOall[i_t * V + tidx]
                local_sumsq = raw_o * raw_o
            for offset in [16, 8, 4, 2, 1]:
                local_sumsq += cute.arch.shuffle_sync_bfly(
                    local_sumsq, offset=offset, mask=-1, mask_and_clamp=31
                )
            if in_warp_tid == 0 and warp_idx < V // 32:
                sWarpSum[warp_idx] = local_sumsq
            cute.arch.barrier()

            if warp_idx == 0:
                block_sumsq = cutlass.Float32(0.0)
                if in_warp_tid < V // 32:
                    block_sumsq = sWarpSum[in_warp_tid]
                for offset in [16, 8, 4, 2, 1]:
                    block_sumsq += cute.arch.shuffle_sync_bfly(
                        block_sumsq, offset=offset, mask=-1, mask_and_clamp=31
                    )
                if in_warp_tid == 0:
                    sWarpSum[0] = cute.math.rsqrt(
                        block_sumsq / cutlass.Float32(V) + onorm_eps,
                        fastmath=True,
                    )
            cute.arch.barrier()

            if tidx < V:
                gate_raw = cutlass.Float32(onorm_g[0, bos + i_t, i_hv, tidx])
                gate = cute.arch.rcp_approx(
                    cutlass.Float32(1.0) + cute.math.exp(-gate_raw, fastmath=True)
                )
                o[0, bos + i_t, i_hv, tidx] = cutlass.BFloat16(
                    raw_o * sWarpSum[0] * cutlass.Float32(onorm_weight[tidx]) * gate
                )
            cute.arch.barrier()


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
    lower_bound: cutlass.Constexpr[float],
    USE_SETMAXREG: cutlass.Constexpr[bool],
    CACHE_RING: cutlass.Constexpr[bool],
    APPLY_ONORM: cutlass.Constexpr[bool],
    onorm_eps: cutlass.Constexpr[float],
    stream: cuda.CUstream,
):
    """Launch the fixed Kimi-K3/DSpARK bonus + NUM_SPEC-draft specialization."""
    smem_qk_layout = cute.make_layout((1 + NUM_SPEC, TILE_K), stride=(TILE_K, 1))
    smem_state_layout = cute.make_layout(
        (TILE_V, TILE_K, NUM_STATE_STAGES),
        stride=(TILE_K, 1, TILE_V * TILE_K),
    )
    state_copy_atom = cute.make_copy_atom(
        cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
        cutlass.Float32,
        num_bits_per_copy=128,
    )
    state_g2s_copy = cute.make_tiled_copy_tv(
        state_copy_atom,
        thr_layout=cute.make_layout((TILE_V, 4), stride=(4, 1)),
        val_layout=cute.make_layout((1, 4)),
    )
    t_loop = 1 + NUM_SPEC
    smem_bytes = (
        # sQ, sK, sG, sBeta, sWarpSum, sVall, and sConvW.
        (3 * t_loop * TILE_K + t_loop + 8 + t_loop * TILE_K + 8 * TILE_K) * 4
        # Two 32 KiB state stages and the raw output tile for normalization.
        + NUM_STATE_STAGES * TILE_V * TILE_K * 4
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
        lower_bound,
        USE_SETMAXREG,
        CACHE_RING,
        APPLY_ONORM,
        onorm_eps,
    ).launch(
        grid=(H, N, 1),
        block=[NUM_THREADS, 1, 1],
        smem=smem_bytes,
        stream=stream,
        use_pdl=True,
    )


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
    if N <= 0 or T % N != 0 or T // N < 2:
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
    use_setmaxreg = N in (1, 32, 128, 512) and H in (2, 12, 32)
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
    key = (
        H,
        N,
        num_spec,
        cache_ring,
        apply_onorm,
        float(onorm_eps) if apply_onorm else 0.0,
        float(scale),
        float(lower_bound),
        use_setmaxreg,
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
            lower_bound=float(lower_bound),
            USE_SETMAXREG=use_setmaxreg,
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
