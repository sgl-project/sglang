import builtins
import os
import warnings
import operator
import torch
from typing import Optional

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass._mlir.dialects import nvvm
from cutlass import Int32, Int64, Float16, BFloat16, Float32, const_expr
from functools import partial

import cuda.bindings.driver as cuda

import logging
logger = logging.getLogger(__name__)

fma_packed_f32x2 = partial(cute.arch.fma_packed_f32x2, rnd=nvvm.RoundingModeKind.RN)
mul_packed_f32x2 = partial(cute.arch.mul_packed_f32x2, rnd=nvvm.RoundingModeKind.RN)
add_packed_f32x2 = partial(cute.arch.add_packed_f32x2, rnd=nvvm.RoundingModeKind.RN)
sub_packed_f32x2 = partial(
    cute.arch.calc_packed_f32x2_op,
    src_c=None,
    calc_func=nvvm.sub_packed_f32x2,
    rnd=nvvm.RoundingModeKind.RN,
)

torch2cute_dtype_map = {
    torch.float16: Float16,
    torch.bfloat16: BFloat16,
    torch.float32: Float32,
    torch.int32: Int32,
    torch.int64: Int64,
}

@cute.jit
def reduce_dim0(
    input_r: cute.Tensor,
    output: cute.Tensor,
    DIM0: cutlass.Constexpr[int],
    DIM1: cutlass.Constexpr[int],
):
    '''
    Input:
    ↑     | T0V0 | T0V4 | T0V8 | T0V12 | T32V0 | T32V4 | T32V8 | T32V12 |
    | r   | T0V1 | T0V5 | T0V9 | T0V13 | T32V1 | T32V5 | T32V9 | T32V13 |
    | e   | T0V2 | T0V6 | T0V10 | T0V14 | T32V2 | T32V6 | T32V10 | T32V14 |
    | d   | T0V3 | T0V7 | T0V11 | T0V15 | T32V3 | T32V7 | T32V11 | T32V15 |
    | u   | T1V0 | T1V4 | T1V8 | T1V12 | T32V0 | T32V4 | T32V8 | T32V12 |
    | c   | T1V1 | T1V5 | T1V9 | T1V13 | T32V1 | T32V5 | T32V9 | T32V13 |
    | e   | T1V2 | T1V6 | T1V10 | T1V14 | T32V2 | T32V6 | T32V10 | T32V14 |
    |     | T1V3 | T1V7 | T1V11 | T1V15 | T32V3 | T32V7 | T32V11 | T32V15 |
    |     | .... |
    
    Output:
    | T0V0 | T0V4 | T0V8 | T0V12 | T32V0 | T32V4 | T32V8 | T32V12 |
    '''
    # reduce input_r along dim 0 and output to output
    assert output.shape[0] == DIM1
    assert input_r.shape[0][0] == DIM0
    assert input_r.shape[0][1] == DIM1

    input_r_ = cute.make_rmem_tensor_like(output, input_r.element_type)

    for reg_H_idx_y in cutlass.range_constexpr(0, DIM1, 2, unroll=(DIM1 // 2)):
        input_r_[reg_H_idx_y], input_r_[reg_H_idx_y + 1] = add_packed_f32x2(
            (input_r[reg_H_idx_y * DIM0], input_r[(reg_H_idx_y + 1) * DIM0]),
            (input_r[reg_H_idx_y * DIM0 + 1], input_r[(reg_H_idx_y + 1) * DIM0 + 1]),
        )

        for reg_H_idx_x in cutlass.range_constexpr(2, DIM0, 1, unroll=(DIM0 - 2)):
            input_r_[reg_H_idx_y], input_r_[reg_H_idx_y + 1] = add_packed_f32x2(
                (input_r_[reg_H_idx_y], input_r_[reg_H_idx_y + 1]),
                (input_r[reg_H_idx_y * DIM0 + reg_H_idx_x], input_r[(reg_H_idx_y + 1) * DIM0 + reg_H_idx_x]),
            ) 

    for reg_H_idx_y in cutlass.range_constexpr(0, DIM1, 1, unroll=DIM1):
        input_r_[reg_H_idx_y] = cute.arch.warp_reduction(
            input_r_[reg_H_idx_y],
            operator.add,
            threads_in_group=cute.arch.WARP_SIZE
        )

    for reg_H_idx_y in cutlass.range_constexpr(0, DIM1, 1, unroll=DIM1):
        output[reg_H_idx_y] = input_r_[reg_H_idx_y].to(output.element_type)

@cute.jit
def L2Norm(
    X: cute.Tensor,
    elem_per_thread: cutlass.Constexpr[int],
):
    '''
    X_norm = X / tl.sqrt(tl.sum(X * X) + 1e-6)
    '''
    thrX_r = X.load().to(cute.Float32)
    thrX_norm = cute.make_rmem_tensor_like(X, cute.Float32)
    thrX_sum = 0.0
    for reg_X_idx in cutlass.range_constexpr(0, elem_per_thread, 2, unroll=2):
        thrX_norm[reg_X_idx], thrX_norm[reg_X_idx + 1] = mul_packed_f32x2(
            (thrX_r[reg_X_idx], thrX_r[reg_X_idx + 1]),
            (thrX_r[reg_X_idx], thrX_r[reg_X_idx + 1]),
        )
        thrX_sum += thrX_norm[reg_X_idx]
        thrX_sum += thrX_norm[reg_X_idx + 1]

    thrX_sum = cute.arch.warp_reduction(
        thrX_sum,
        operator.add,
        threads_in_group=cute.arch.WARP_SIZE
    )

    thrX_rsqrt = cute.rsqrt(thrX_sum + 1e-6)
    for reg_X_idx in cutlass.range_constexpr(0, elem_per_thread, 2, unroll=2):
        thrX_norm[reg_X_idx], thrX_norm[reg_X_idx + 1] = mul_packed_f32x2(
            (thrX_r[reg_X_idx], thrX_r[reg_X_idx + 1]),
            (thrX_rsqrt, thrX_rsqrt),
        )
    return thrX_norm

@cute.kernel
def fused_recurrent_sigmoid_update_kernel_128x32_col(
    gA: cute.Tensor,
    ga: cute.Tensor,
    gdt_bias: cute.Tensor,
    softplus_beta: float,
    softplus_threshold: float,
    gQ: cute.Tensor, 
    gK: cute.Tensor, 
    gV: cute.Tensor, 
    gH: cute.Tensor, 
    gO: cute.Tensor,
    gB: cute.Tensor,
    gIndices: cute.Tensor,
    cu_seqlens: cute.Tensor,
    scale: float,
    tv_layout_k: cute.Layout,
    tv_layout_v: cute.Layout,
    tv_layout_h: cute.Layout, 
    T: cutlass.Constexpr[int],
    HK: cutlass.Constexpr[int],
    HV: cutlass.Constexpr[int],
    x_threads: int,
    y_threads: int,
    ELEM_H_X: cutlass.Constexpr[int],
    ELEM_H_Y: cutlass.Constexpr[int],
    USE_QK_L2NORM_IN_KERNEL: cutlass.Constexpr[bool],
):
    batch_idx, head_idx, bidx = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()

    if const_expr(cu_seqlens is not None):
        bos, eos = cu_seqlens[batch_idx], cu_seqlens[batch_idx + 1]
        T = eos - bos
        state_idx = gIndices[batch_idx]
        batch_idx = 0
    else:
        bos = 0
        state_idx = gIndices[batch_idx]

    blk_coord_H = ((None, None, None, None), (state_idx, head_idx, None, bidx))
    blkH = gH[blk_coord_H]
    
    tidfrgH = cute.composition(blkH, tv_layout_h)
    
    tArA = gA[head_idx].to(cute.Float32)
    tDrD = gdt_bias[head_idx].to(cute.Float32)

    thrQ_coord = (tidx % x_threads, None)
    thrK_coord = (tidx % x_threads, None)
    thrV_coord = (bidx * y_threads + tidx // x_threads, None)
    thrH_coord = (tidx, None)
    thrO_coord = (bidx * y_threads + tidx // x_threads, None)

    tHgH = tidfrgH[thrH_coord]

    tHrH_i = tHgH.load().to(cute.Float32)
    tHrH_g = cute.make_rmem_tensor_like(tHgH, cute.Float32)
    tHrHk = cute.make_rmem_tensor_like(tHgH, cute.Float32)

    for t_idx in cutlass.range(0, 1, 1, unroll=1):
        blk_coord_a = ((None, None, None), (batch_idx, bos + t_idx, head_idx))
        blk_coord_B = ((None, None, None), (batch_idx, bos + t_idx, head_idx))
        blk_coord_Q = ((None, None, None, None), (batch_idx, bos + t_idx, head_idx // (HV // HK), None))
        blk_coord_K = ((None, None, None, None), (batch_idx, bos + t_idx, head_idx // (HV // HK), None))
        blk_coord_V = ((None, None, None, None), (batch_idx, bos + t_idx, head_idx, None))
        blk_coord_O = ((None, None, None, None), (batch_idx, bos + t_idx, head_idx, None))

        blka = ga[blk_coord_a]
        blkQ = gQ[blk_coord_Q]  
        blkK = gK[blk_coord_K] 
        blkV = gV[blk_coord_V]
        blkO = gO[blk_coord_O]
        blkB = gB[blk_coord_B]

        tidfrgQ = cute.composition(blkQ, tv_layout_k)
        tidfrgK = cute.composition(blkK, tv_layout_k)
        tidfrgV = cute.composition(blkV, tv_layout_v)
        tidfrgO = cute.composition(blkO, tv_layout_v)

        tQgQ = tidfrgQ[thrQ_coord]
        tKgK = tidfrgK[thrK_coord]
        tVgV = tidfrgV[thrV_coord]
        tOgO = tidfrgO[thrO_coord]

        tBrBeta = blkB.load()[0].to(cute.Float32)
        tMrMa = blka.load().to(cute.Float32)
        tVrU = cute.make_rmem_tensor_like(tVgV, cute.Float32)
        tVrV = tVgV.load().to(cute.Float32)

        # sigmoid
        x = tMrMa + tDrD
        beta_x = softplus_beta * x
        softplux_x = cute.where(
            beta_x <= softplus_threshold,
            (1.0 / softplus_beta) * cute.math.log(1.0 + cute.math.exp(beta_x)),
            x,
        )

        tGrG = -cute.math.exp(tArA) * softplux_x
        tBrB = 1.0 / (1.0 + cute.math.exp(-tBrBeta))
        

        if const_expr(USE_QK_L2NORM_IN_KERNEL):
            tQrQ = L2Norm(tQgQ, ELEM_H_X)
            tKrK = L2Norm(tKgK, ELEM_H_X)
        else:
            tQrQ = tQgQ.load().to(cute.Float32)
            tKrK = tKgK.load().to(cute.Float32)

        for reg_Q_idx in cutlass.range_constexpr(0, ELEM_H_X, 2, unroll=2):
            tQrQ[reg_Q_idx], \
                tQrQ[reg_Q_idx + 1] = mul_packed_f32x2(
                (
                    tQrQ[reg_Q_idx], 
                    tQrQ[reg_Q_idx + 1]
                ),
                (
                    scale, 
                    scale
                ),
            )

        tGrGexp = cute.math.exp(tGrG, fastmath=True)[0]
        for reg_H_idx in cutlass.range_constexpr(0, ELEM_H_X * ELEM_H_Y, 2, unroll=2):
            tHrH_g[reg_H_idx], tHrH_g[reg_H_idx + 1] = mul_packed_f32x2(
                (
                    tHrH_i[reg_H_idx],
                    tHrH_i[reg_H_idx + 1]
                ),
                (
                    tGrGexp,
                    tGrGexp
                ),
            )

        # b_v = b_beta * (b_v - tl.sum(b_h * b_k[:, None], 0))
        # b_h * b_k[:, None]
        for reg_H_idx in cutlass.range_constexpr(0, ELEM_H_X * ELEM_H_Y, 2, unroll=2):
            tHrHk[reg_H_idx], tHrHk[reg_H_idx + 1] = mul_packed_f32x2(
                (
                    tHrH_g[reg_H_idx],
                    tHrH_g[reg_H_idx + 1]
                ),
                (
                    tKrK[reg_H_idx % ELEM_H_X], 
                    tKrK[(reg_H_idx + 1) % ELEM_H_X]
                ),
            )

        reduce_dim0(tHrHk, tVrU, ELEM_H_X, ELEM_H_Y)

        # b_v - sum(b_h*b_k) 
        for reg_V_idx in cutlass.range_constexpr(0, ELEM_H_X, 2, unroll=2):
            tVrU[reg_V_idx], tVrU[reg_V_idx + 1] = sub_packed_f32x2(
                (
                    tVrV[reg_V_idx], 
                    tVrV[reg_V_idx + 1]
                ),
                (
                    tVrU[reg_V_idx], 
                    tVrU[reg_V_idx + 1]
                ),
            )

            tVrU[reg_V_idx], \
                tVrU[reg_V_idx + 1] = mul_packed_f32x2(
                (
                    tVrU[reg_V_idx], 
                    tVrU[reg_V_idx + 1]
                ),
                (
                    tBrB, 
                    tBrB
                ),
            )

        # b_h = b_k[:, None] * b_v
        for reg_K_idx in cutlass.range_constexpr(0, ELEM_H_X, 2, unroll=2):
            for reg_V_idx in cutlass.range_constexpr(0, ELEM_H_Y, 2, unroll=2):
                tHrHk[reg_V_idx * ELEM_H_X + reg_K_idx], \
                    tHrHk[(reg_V_idx + 1) * ELEM_H_X + reg_K_idx + 1] = fma_packed_f32x2(
                    (
                        tKrK[reg_K_idx], 
                        tKrK[reg_K_idx+1]
                    ),
                    (
                        tVrU[reg_V_idx], 
                        tVrU[reg_V_idx+1]
                    ),
                    (
                        tHrH_g[reg_V_idx * ELEM_H_X + reg_K_idx], 
                        tHrH_g[(reg_V_idx + 1) * ELEM_H_X + reg_K_idx + 1]
                    )
                )

                tHrHk[(reg_V_idx + 1) * ELEM_H_X + reg_K_idx], \
                    tHrHk[reg_V_idx * ELEM_H_X + reg_K_idx + 1] = fma_packed_f32x2(
                    (
                        tKrK[reg_K_idx], 
                        tKrK[reg_K_idx+1]
                    ),
                    (
                        tVrU[reg_V_idx + 1], 
                        tVrU[reg_V_idx]
                    ),
                    (
                        tHrH_g[(reg_V_idx + 1) * ELEM_H_X + reg_K_idx], 
                        tHrH_g[reg_V_idx * ELEM_H_X + reg_K_idx + 1]
                    )
                )
                
        for reg_H_idx in cutlass.range_constexpr(0, ELEM_H_X * ELEM_H_Y, 2, unroll=2):
            tHrH_g[reg_H_idx], \
                tHrH_g[reg_H_idx + 1] = mul_packed_f32x2(
                (
                    tHrHk[reg_H_idx], 
                    tHrHk[reg_H_idx + 1]
                ),
                (
                    tQrQ[reg_H_idx % ELEM_H_X], 
                    tQrQ[(reg_H_idx + 1) % ELEM_H_X]
                ),
            )

        reduce_dim0(tHrH_g, tOgO, ELEM_H_X, ELEM_H_Y)

    for reg_H_idx in cutlass.range_constexpr(0, ELEM_H_X * ELEM_H_Y, 1, unroll=4):
        tHgH[reg_H_idx] = tHrHk[reg_H_idx].to(tHgH.element_type)

@cute.jit
def fused_recurrent_sigmoid_update_128x32_col(
    mA: cute.Tensor,
    ma: cute.Tensor,
    mdt_bias: cute.Tensor,
    softplus_beta: cutlass.Constexpr[float],
    softplus_threshold: cutlass.Constexpr[float],
    mQ: cute.Tensor,
    mK: cute.Tensor,
    mV: cute.Tensor,
    mH: cute.Tensor,
    mO: cute.Tensor,
    mB: cute.Tensor,
    mIndices: cute.Tensor,
    cu_seqlens: cute.Tensor,
    BK: cutlass.Constexpr[int],
    BV: cutlass.Constexpr[int],
    DIM: cutlass.Constexpr[int],
    scale: cutlass.Constexpr[float],
    USE_QK_L2NORM_IN_KERNEL: cutlass.Constexpr[bool],
    stream: cuda.CUstream = None,
):
    h_dtype = mH.element_type
    k_dtype = mK.element_type
    x_threads = 32
    y_threads = 8
    if const_expr(DIM == 256):
        x_threads = 32
        y_threads = 16

    elem_per_thread_k = BK // x_threads
    k_thr_layout = cute.make_ordered_layout((1, x_threads), order=(1, 0))
    k_val_layout = cute.make_ordered_layout((1, elem_per_thread_k * k_dtype.width // 8), order=(1, 0))
    k_val_layout_recast = cute.recast_layout(k_dtype.width, 8, k_val_layout)
    k_tiler_mn, tv_layout_k = cute.make_layout_tv(k_thr_layout, k_val_layout_recast)

    elem_per_thread_v = BV // y_threads
    v_thr_layout = cute.make_ordered_layout((1, y_threads), order=(1, 0))
    v_val_layout = cute.make_ordered_layout((1, elem_per_thread_v * k_dtype.width // 8), order=(1, 0))
    v_val_layout_recast = cute.recast_layout(k_dtype.width, 8, v_val_layout)
    v_tiler_mn, tv_layout_v = cute.make_layout_tv(v_thr_layout, v_val_layout_recast)

    coalesced_bytesl_h_x = BK * h_dtype.width // 8 // x_threads
    elem_per_thread_h_y = BV // y_threads
    thr_h_layout = cute.make_ordered_layout((x_threads, y_threads), order=(0, 1))
    h_val_layout = cute.make_ordered_layout((coalesced_bytesl_h_x, elem_per_thread_h_y), order=(0, 1))
    h_val_layout = cute.recast_layout(h_dtype.width, 8, h_val_layout)
    tiler_mn_h, tv_layout_h = cute.make_layout_tv(thr_h_layout, h_val_layout)
    elem_h_x, elem_h_y = h_val_layout.shape[0], h_val_layout.shape[1]

    # (B, T, H, N)
    gQ = cute.zipped_divide(mQ, (1, 1, k_tiler_mn[0], k_tiler_mn[1]))  
    gK = cute.zipped_divide(mK, (1, 1, k_tiler_mn[0], k_tiler_mn[1]))  
    gV = cute.zipped_divide(mV, (1, 1, v_tiler_mn[0], v_tiler_mn[1]))  
    gO = cute.zipped_divide(mO, (1, 1, v_tiler_mn[0], v_tiler_mn[1]))  
    gH = cute.zipped_divide(mH, (1, 1, tiler_mn_h[0], tiler_mn_h[1]))  
    gB = cute.zipped_divide(mB, (1, 1, 1)) 
    gA = mA
    ga = cute.zipped_divide(ma, (1, 1, 1))
    gdt_bias = mdt_bias

    B = mQ.shape[0] if cu_seqlens is None else cu_seqlens.shape[0] - 1
    T = mK.shape[1]
    HK = mK.shape[2]
    HV = mV.shape[2]
    blocks_per_head = mK.shape[-1] // BV

    fused_recurrent_sigmoid_update_kernel_128x32_col(
        gA, 
        ga, 
        gdt_bias,
        softplus_beta,
        softplus_threshold,
        gQ,
        gK,
        gV,
        gH,
        gO,
        gB,
        mIndices,
        cu_seqlens,
        scale,
        tv_layout_k,
        tv_layout_v,
        tv_layout_h,
        T,
        HK,
        HV,
        x_threads,
        y_threads,
        elem_h_x,
        elem_h_y,
        USE_QK_L2NORM_IN_KERNEL
    ).launch(
        grid=[B, HV, blocks_per_head],
        block=[cute.size(tv_layout_h, mode=[0]), 1, 1],
        stream=stream,
    )

def cutedsl_fused_recurrent_sigmoid_gated_delta_rule_update(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    softplus_beta: float,
    softplus_threshold: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    gv: torch.Tensor | None = None,
    b: torch.Tensor | None = None,
    scale: float = None,
    initial_state_source: torch.Tensor = None,
    initial_state_indices: torch.Tensor = None,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    stream: cuda.CUstream = None,
):
    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[-2]

    if initial_state_source is not None and initial_state_source.stride()[-2] != 1:
        warnings.warn("K dim should be contiguous, or performance will be degraded", RuntimeWarning)
    assert K == V and (K == 128 or K == 256), "Current cutedsl decode only support K and V dim to be 128 or 256"


    if scale is None:
        scale = k.shape[-1] ** -0.5
    else:
        assert scale > 0, "scale must be positive"

    if b is None:
        b = torch.ones_like(q[..., 0])

    if b.dim() == 2:
        b = b.unsqueeze(0)
    if a.dim() == 2:
        a = a.unsqueeze(0)

    q_ = from_dlpack(q.detach(), assumed_align=16)
    k_ = from_dlpack(k.detach(), assumed_align=16)
    v_ = from_dlpack(v.detach(), assumed_align=16)
    h_ = from_dlpack(initial_state_source.detach(), assumed_align=16)
    b_ = from_dlpack(b.detach(), assumed_align=16)
    ind_ = from_dlpack(initial_state_indices.detach(), assumed_align=16)
    if cu_seqlens is not None:
        cu_seqlens_ = from_dlpack(cu_seqlens.detach(), assumed_align=16)
    else:
        cu_seqlens_ = None
    A_log_ = from_dlpack(A_log.detach(), assumed_align=16)
    a_ = from_dlpack(a.detach(), assumed_align=16)
    dt_bias_ = from_dlpack(dt_bias.detach(), assumed_align=16)

    o = torch.empty_like(v)
    o_ = from_dlpack(o.detach(), assumed_align=16)

    BK = K
    BV = V // 4

    dtype = torch2cute_dtype_map[initial_state_source.dtype]

    compile_key = (dtype, B, T, H, HV, BV, use_qk_l2norm_in_kernel)
        
    if stream is None:
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    if compile_key not in cutedsl_fused_recurrent_sigmoid_gated_delta_rule_update.compile_cache:
        logger.info(f"\nCompiling fused_recurrent_sigmoid_gated_delta_rule_update_fwd kernel with state_dtype={dtype}, B={B}, T={T}, H={H}, HV={HV}, K={K}, V={V}, BV={BV}, use_qk_l2norm_in_kernel={use_qk_l2norm_in_kernel}")
        cutedsl_fused_recurrent_sigmoid_gated_delta_rule_update.compile_cache[compile_key] = \
            cute.compile(fused_recurrent_sigmoid_update_128x32_col, 
                            A_log_, 
                            a_, 
                            dt_bias_, 
                            softplus_beta, 
                            softplus_threshold, 
                            q_, 
                            k_,
                            v_,
                            h_,
                            o_,
                            b_,
                            ind_,
                            cu_seqlens_, 
                            BK, 
                            BV, 
                            DIM=K, 
                            scale=scale, 
                            USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel, 
                            stream=stream,
                            # options="--enable-tvm-ffi"
                        )

    cutedsl_fused_recurrent_sigmoid_gated_delta_rule_update.compile_cache[compile_key](
        A_log_,
        a_,
        dt_bias_,
        q_, 
        k_, 
        v_, 
        h_, 
        o_, 
        b_, 
        ind_, 
        cu_seqlens_, 
        stream=stream,
    )
    o = o.squeeze(0)
    return o

cutedsl_fused_recurrent_sigmoid_gated_delta_rule_update.compile_cache = {}


@cute.kernel
def fused_recurrent_update_kernel_128x32_col(
    gQ: cute.Tensor, 
    gK: cute.Tensor, 
    gV: cute.Tensor, 
    gH: cute.Tensor, 
    gO: cute.Tensor,
    gB: cute.Tensor,
    gG: cute.Tensor,
    gIndices: cute.Tensor,
    cu_seqlens: cute.Tensor,
    gInterState: cute.Tensor,
    gInterIndices: cute.Tensor,
    scale: float,
    tv_layout_k: cute.Layout,
    tv_layout_v: cute.Layout,
    tv_layout_h: cute.Layout, 
    T: cutlass.Constexpr[int],
    HK: cutlass.Constexpr[int],
    HV: cutlass.Constexpr[int],
    x_threads: int,
    y_threads: int,
    ELEM_H_X: cutlass.Constexpr[int],
    ELEM_H_Y: cutlass.Constexpr[int],
    USE_QK_L2NORM_IN_KERNEL: cutlass.Constexpr[bool],
    DISABLE_STATE_UPDATE: cutlass.Constexpr[bool],
    DISABLE_OUTPUT_CALCULATION: cutlass.Constexpr[bool],
    CACHE_INTERMEDIATE_STATES: cutlass.Constexpr[bool],
    CACHE_STEPS: cutlass.Constexpr[int],
    # RETRIEVE_PARENT_TOKEN: cutlass.Constexpr[bool],
):
    batch_idx, head_idx, bidx = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()

    if const_expr(cu_seqlens is not None):
        bos, eos = cu_seqlens[batch_idx], cu_seqlens[batch_idx + 1]
        T = eos - bos
    else:
        bos = 0

    state_idx = gIndices[batch_idx]

    cache_idx = -1
    if const_expr(CACHE_INTERMEDIATE_STATES):
        cache_idx = gInterIndices[batch_idx]
    
    if const_expr(cu_seqlens is not None):
        batch_idx = 0

    blk_coord_H = ((None, None, None, None), (state_idx, head_idx, None, bidx))
    
    blkH = gH[blk_coord_H] 

    tidfrgH = cute.composition(blkH, tv_layout_h)

    thrQ_coord = (tidx % x_threads, None)
    thrK_coord = (tidx % x_threads, None)
    thrV_coord = (bidx * y_threads + tidx // x_threads, None)
    thrH_coord = (tidx, None)
    thrI_coord = (tidx, None)
    thrO_coord = (bidx * y_threads + tidx // x_threads, None)

    tHgH = tidfrgH[thrH_coord]
    tHrH_i = tHgH.load().to(cute.Float32)
    tHrH_g = cute.make_rmem_tensor_like(tHgH, cute.Float32)
    tHrHk = cute.make_rmem_tensor_like(tHgH, cute.Float32)

    # Initialize tHrHk with tHrH_i to avoid dominance issues
    for reg_H_idx in cutlass.range_constexpr(0, ELEM_H_X * ELEM_H_Y, 1, unroll=ELEM_H_X * ELEM_H_Y):
        tHrHk[reg_H_idx] = tHrH_i[reg_H_idx].to(tHrHk.element_type)

    for t_idx in cutlass.range(0, T, 1, unroll=1):
        blk_coord_B = ((None, None, None), (batch_idx, bos + t_idx, head_idx))
        blk_coord_G = ((None, None, None), (batch_idx, bos + t_idx, head_idx))
        blk_coord_Q = ((None, None, None, None), (batch_idx, bos + t_idx, head_idx // (HV // HK), None))
        blk_coord_K = ((None, None, None, None), (batch_idx, bos + t_idx, head_idx // (HV // HK), None))
        blk_coord_V = ((None, None, None, None), (batch_idx, bos + t_idx, head_idx, None)) 
        blk_coord_O = ((None, None, None, None), (batch_idx, bos + t_idx, head_idx, None))

        blkQ = gQ[blk_coord_Q]
        blkK = gK[blk_coord_K]
        blkV = gV[blk_coord_V]
        blkO = gO[blk_coord_O]
        blkB = gB[blk_coord_B]
        blkG = gG[blk_coord_G]

        tIgI = 0.0
        if const_expr(CACHE_INTERMEDIATE_STATES):
            blk_coord_I = ((None, None, None, None, None), (cache_idx, t_idx, head_idx, None, bidx))
            blkI = gInterState[blk_coord_I]
            tidfrgI = cute.composition(blkI, tv_layout_h)
            tIgI = tidfrgI[thrI_coord]

        tidfrgQ = cute.composition(blkQ, tv_layout_k)
        tidfrgK = cute.composition(blkK, tv_layout_k)
        tidfrgV = cute.composition(blkV, tv_layout_v)
        tidfrgO = cute.composition(blkO, tv_layout_v)

        tQgQ = tidfrgQ[thrQ_coord]
        tKgK = tidfrgK[thrK_coord]
        tVgV = tidfrgV[thrV_coord]
        tOgO = tidfrgO[thrO_coord]
        tBrB = blkB.load()[0].to(cute.Float32)
        tGrG = blkG.load()[0].to(cute.Float32)

        tVrU = cute.make_rmem_tensor_like(tVgV, cute.Float32)
        tVrV = tVgV.load().to(cute.Float32)

        if const_expr(USE_QK_L2NORM_IN_KERNEL):
            tQrQ = L2Norm(tQgQ, ELEM_H_X)
            tKrK = L2Norm(tKgK, ELEM_H_X)
        else:
            tQrQ = tQgQ.load().to(cute.Float32)
            tKrK = tKgK.load().to(cute.Float32)

        for reg_Q_idx in cutlass.range_constexpr(0, ELEM_H_X, 2, unroll=2):
            tQrQ[reg_Q_idx], \
                tQrQ[reg_Q_idx + 1] = mul_packed_f32x2(
                (
                    tQrQ[reg_Q_idx], 
                    tQrQ[reg_Q_idx + 1]
                ),
                (
                    scale, 
                    scale
                ),
            )

        if const_expr(gG is not None):
            tGrGexp = cute.math.exp(tGrG, fastmath=True)
            for reg_H_idx in cutlass.range_constexpr(0, ELEM_H_X * ELEM_H_Y, 1, unroll=1):
                tHrH_g[reg_H_idx] = tHrHk[reg_H_idx] * tGrGexp
        else:
            for reg_H_idx in cutlass.range_constexpr(0, ELEM_H_X * ELEM_H_Y, 1, unroll=1):
                tHrH_g[reg_H_idx] = tHrHk[reg_H_idx]

        for reg_H_idx in cutlass.range_constexpr(0, ELEM_H_X * ELEM_H_Y, 2, unroll=2):
            tHrHk[reg_H_idx], tHrHk[reg_H_idx + 1] = mul_packed_f32x2(
                (
                    tHrH_g[reg_H_idx],
                    tHrH_g[reg_H_idx + 1]
                ),
                (
                    tKrK[reg_H_idx % ELEM_H_X], 
                    tKrK[(reg_H_idx + 1) % ELEM_H_X]
                ),
            )

        reduce_dim0(tHrHk, tVrU, ELEM_H_X, ELEM_H_Y)

        for reg_V_idx in cutlass.range_constexpr(0, ELEM_H_X, 2, unroll=2):
            tVrU[reg_V_idx], tVrU[reg_V_idx + 1] = sub_packed_f32x2(
                (
                    tVrV[reg_V_idx], 
                    tVrV[reg_V_idx + 1]
                ),
                (
                    tVrU[reg_V_idx], 
                    tVrU[reg_V_idx + 1]
                ),
            )

            tVrU[reg_V_idx], \
                tVrU[reg_V_idx + 1] = mul_packed_f32x2(
                (
                    tVrU[reg_V_idx], 
                    tVrU[reg_V_idx + 1]
                ),
                (
                    tBrB, 
                    tBrB
                ),
            )

        for reg_K_idx in cutlass.range_constexpr(0, ELEM_H_X, 2, unroll=2):
            for reg_V_idx in cutlass.range_constexpr(0, ELEM_H_Y, 2, unroll=2):
                tHrHk[reg_V_idx * ELEM_H_X + reg_K_idx], \
                    tHrHk[(reg_V_idx + 1) * ELEM_H_X + reg_K_idx + 1] = fma_packed_f32x2(
                    (
                        tKrK[reg_K_idx], 
                        tKrK[reg_K_idx+1]
                    ),
                    (
                        tVrU[reg_V_idx], 
                        tVrU[reg_V_idx+1]
                    ),
                    (
                        tHrH_g[reg_V_idx * ELEM_H_X + reg_K_idx], 
                        tHrH_g[(reg_V_idx + 1) * ELEM_H_X + reg_K_idx + 1]
                    )
                )

                tHrHk[(reg_V_idx + 1) * ELEM_H_X + reg_K_idx], \
                    tHrHk[reg_V_idx * ELEM_H_X + reg_K_idx + 1] = fma_packed_f32x2(
                    (
                        tKrK[reg_K_idx], 
                        tKrK[reg_K_idx+1]
                    ),
                    (
                        tVrU[reg_V_idx + 1], 
                        tVrU[reg_V_idx]
                    ),
                    (
                        tHrH_g[(reg_V_idx + 1) * ELEM_H_X + reg_K_idx], 
                        tHrH_g[reg_V_idx * ELEM_H_X + reg_K_idx + 1]
                    )
                )
                
        if const_expr(not DISABLE_OUTPUT_CALCULATION):
            for reg_H_idx in cutlass.range_constexpr(0, ELEM_H_X * ELEM_H_Y, 2, unroll=2):
                tHrH_g[reg_H_idx], \
                    tHrH_g[reg_H_idx + 1] = mul_packed_f32x2(
                    (
                        tHrHk[reg_H_idx], 
                        tHrHk[reg_H_idx + 1]
                    ),
                    (
                        tQrQ[reg_H_idx % ELEM_H_X], 
                        tQrQ[(reg_H_idx + 1) % ELEM_H_X]
                    ),
                )

            reduce_dim0(tHrH_g, tOgO, ELEM_H_X, ELEM_H_Y)

        if const_expr(CACHE_INTERMEDIATE_STATES):
            if cache_idx >= 0:
                for reg_I_idx in cutlass.range_constexpr(0, ELEM_H_X * ELEM_H_Y, 1, unroll=4):
                    tIgI[reg_I_idx] = tHrHk[reg_I_idx].to(tIgI.element_type)
                

    if const_expr(not DISABLE_STATE_UPDATE):
        for reg_H_idx in cutlass.range_constexpr(0, ELEM_H_X * ELEM_H_Y, 1, unroll=4):
            tHgH[reg_H_idx] = tHrHk[reg_H_idx].to(tHgH.element_type)

@cute.jit
def fused_recurrent_update_128x32_col(
    mQ: cute.Tensor,
    mK: cute.Tensor,
    mV: cute.Tensor,
    mH: cute.Tensor,
    mO: cute.Tensor,
    mG: cute.Tensor,
    mB: cute.Tensor,
    mIndices: cute.Tensor,
    cu_seqlens: cute.Tensor,
    mInter: cute.Tensor | None,
    mInterIndices: cute.Tensor | None,
    BK: cutlass.Constexpr[int],
    BV: cutlass.Constexpr[int],
    DIM: cutlass.Constexpr[int],
    scale: cutlass.Constexpr[float],
    USE_QK_L2NORM_IN_KERNEL: cutlass.Constexpr[bool],
    DISABLE_STATE_UPDATE: cutlass.Constexpr[bool] = False,
    DISABLE_OUTPUT_CALCULATION: cutlass.Constexpr[bool] = False,
    CACHE_STEPS: cutlass.Constexpr[int] = None,
    CACHE_INTERMEDIATE_STATES: cutlass.Constexpr[bool] = False,
    RETRIEVE_PARENT_TOKEN: cute.Tensor | None = None,
    stream: cuda.CUstream = None
):
    assert all(t.element_type == mQ.element_type for t in [mQ, mK, mV])

    h_dtype = mH.element_type
    k_dtype = mK.element_type
    x_threads = 32
    y_threads = 8
    if const_expr(DIM == 256):
        x_threads = 32
        y_threads = 16

    elem_per_thread_k = BK // x_threads
    k_thr_layout = cute.make_ordered_layout((1, x_threads), order=(1, 0))
    k_val_layout = cute.make_ordered_layout((1, elem_per_thread_k * k_dtype.width // 8), order=(1, 0))
    k_val_layout_recast = cute.recast_layout(k_dtype.width, 8, k_val_layout)
    k_tiler_mn, tv_layout_k = cute.make_layout_tv(k_thr_layout, k_val_layout_recast)

    elem_per_thread_v = BV // y_threads
    v_thr_layout = cute.make_ordered_layout((1, y_threads), order=(1, 0))
    v_val_layout = cute.make_ordered_layout((1, elem_per_thread_v * k_dtype.width // 8), order=(1, 0))
    v_val_layout_recast = cute.recast_layout(k_dtype.width, 8, v_val_layout)
    v_tiler_mn, tv_layout_v = cute.make_layout_tv(v_thr_layout, v_val_layout_recast)

    coalesced_bytesl_h_x = BK * h_dtype.width // 8 // x_threads
    elem_per_thread_h_y = BV // y_threads
    thr_h_layout = cute.make_ordered_layout((x_threads, y_threads), order=(0, 1))
    h_val_layout = cute.make_ordered_layout((coalesced_bytesl_h_x, elem_per_thread_h_y), order=(0, 1))
    h_val_layout = cute.recast_layout(h_dtype.width, 8, h_val_layout)
    tiler_mn_h, tv_layout_h = cute.make_layout_tv(thr_h_layout, h_val_layout)
    elem_h_x, elem_h_y = h_val_layout.shape[0], h_val_layout.shape[1]

    # (B, T, H, N)
    gQ = cute.zipped_divide(mQ, (1, 1, k_tiler_mn[0], k_tiler_mn[1]))  
    gK = cute.zipped_divide(mK, (1, 1, k_tiler_mn[0], k_tiler_mn[1]))  
    gV = cute.zipped_divide(mV, (1, 1, v_tiler_mn[0], v_tiler_mn[1]))  
    gO = cute.zipped_divide(mO, (1, 1, v_tiler_mn[0], v_tiler_mn[1]))  
    gH = cute.zipped_divide(mH, (1, 1, tiler_mn_h[0], tiler_mn_h[1]))  
    gB = cute.zipped_divide(mB, (1, 1, 1)) 
    gG = cute.zipped_divide(mG, (1, 1, 1))  

    gInterState = None
    if const_expr(CACHE_INTERMEDIATE_STATES):
        gInterState = cute.zipped_divide(mInter, (1, 1, 1, tiler_mn_h[0], tiler_mn_h[1]))  

    B = mQ.shape[0] if cu_seqlens is None else cu_seqlens.shape[0] - 1
    T = mK.shape[1]
    HK = mK.shape[2]
    HV = mV.shape[2]
    blocks_per_head = mK.shape[-1] // BV
    
    assert T // CACHE_STEPS == B, "batch * CACHE_STEPS must be equal to T"
    fused_recurrent_update_kernel_128x32_col(
        gQ,
        gK,
        gV,
        gH,
        gO,
        gB,
        gG,
        mIndices,
        cu_seqlens,
        gInterState,
        mInterIndices,
        scale,
        tv_layout_k,
        tv_layout_v,
        tv_layout_h,
        T,
        HK,
        HV,
        x_threads,
        y_threads,
        elem_h_x,
        elem_h_y,  
        USE_QK_L2NORM_IN_KERNEL=USE_QK_L2NORM_IN_KERNEL, 
        DISABLE_STATE_UPDATE=DISABLE_STATE_UPDATE, 
        DISABLE_OUTPUT_CALCULATION=DISABLE_OUTPUT_CALCULATION,   
        CACHE_INTERMEDIATE_STATES=CACHE_INTERMEDIATE_STATES,
        CACHE_STEPS=CACHE_STEPS
    ).launch(
        grid=[B, HV, blocks_per_head],
        block=[cute.size(tv_layout_h, mode=[0]), 1, 1],
        stream=stream,
    )

def cutedsl_fused_recurrent_gated_delta_rule_update(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    gv: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    scale: float = None,
    initial_state_source: torch.Tensor = None,
    initial_state_indices: torch.Tensor = None,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    disable_state_update: bool = False,
    disable_output_calculation: bool = False,
    intermediate_states_buffer: Optional[torch.Tensor] = None,
    intermediate_state_indices: Optional[torch.Tensor] = None,
    cache_steps: Optional[int] = None,
    retrieve_parent_token: Optional[torch.Tensor] = None,
    stream: cuda.CUstream = None,
):
    B, T, H, K, V = *k.shape, v.shape[-1]
    HV =  v.shape[-2]

    if initial_state_source is not None and initial_state_source.stride()[-2] != 1:
        warnings.warn("K dim should be contiguous, or performance will be degraded", RuntimeWarning)
    assert K == V and (K == 128 or K == 256), "Current cutedsl decode only support K and V dim to be 128 or 256"

    if scale is None:
        scale = k.shape[-1] ** -0.5
    else:
        assert scale > 0, "scale must be positive"

    if beta is None:
        beta = torch.ones_like(q[..., 0])

    if beta.dim() == 2:
        beta = beta.unsqueeze(0)

    q_ = from_dlpack(q.detach(), assumed_align=16)
    k_ = from_dlpack(k.detach(), assumed_align=16)
    v_ = from_dlpack(v.detach(), assumed_align=16)
    h_ = from_dlpack(initial_state_source.detach(), assumed_align=16)
    beta_ = from_dlpack(beta.detach(), assumed_align=16)
    g_ = from_dlpack(g.detach(), assumed_align=16)
    ind_ = from_dlpack(initial_state_indices.detach(), assumed_align=16)
    if cu_seqlens != None:
        cu_seqlens_ = from_dlpack(cu_seqlens.detach(), assumed_align=16)
    else:
        cu_seqlens_ = None

    if cache_steps is None:
        cache_steps = T

    CACHE_INTERMEDIATE_STATES = False
    if intermediate_states_buffer != None:
        intermediate_states_buffer_ = from_dlpack(intermediate_states_buffer.detach(), assumed_align=16)
        intermediate_state_indices_ = from_dlpack(intermediate_state_indices.detach(), assumed_align=16)
        CACHE_INTERMEDIATE_STATES = True
    else:
        intermediate_states_buffer_ = None
        intermediate_state_indices_ = None

    o = torch.empty_like(v)
    o_ = from_dlpack(o.detach(), assumed_align=16)

    BK = K
    BV = K // 4

    dtype = torch2cute_dtype_map[initial_state_source.dtype]

    compile_key = (dtype, B, T, H, HV, BV, use_qk_l2norm_in_kernel, cache_steps)
        
    if stream is None:
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    
    if compile_key not in cutedsl_fused_recurrent_gated_delta_rule_update.compile_cache:
        logger.info(f"\nCompiling cutedsl_fused_recurrent_gated_delta_rule_update kernel with state_dtype={dtype}, B={B}, T={T}, K={K}, V={V}, BV={BV}, use_qk_l2norm_in_kernel={use_qk_l2norm_in_kernel}, cache_steps={cache_steps}")
        cutedsl_fused_recurrent_gated_delta_rule_update.compile_cache[compile_key] = \
            cute.compile(fused_recurrent_update_128x32_col, 
                         q_, 
                         k_,
                         v_, 
                         h_, 
                         o_, 
                         g_, 
                         beta_, 
                         ind_, 
                         cu_seqlens_, 
                         intermediate_states_buffer_, 
                         intermediate_state_indices_, 
                         BK, 
                         BV, 
                         DIM=K,
                         scale=scale,
                         USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel, 
                         DISABLE_STATE_UPDATE=disable_state_update, 
                         DISABLE_OUTPUT_CALCULATION=disable_output_calculation, 
                         CACHE_INTERMEDIATE_STATES=CACHE_INTERMEDIATE_STATES,
                         CACHE_STEPS=cache_steps,
                         stream=stream,
                        #  options="--enable-tvm-ffi"
                    )
    
    cutedsl_fused_recurrent_gated_delta_rule_update.compile_cache[compile_key](
        q_, 
        k_, 
        v_, 
        h_, 
        o_, 
        g_, 
        beta_,
        ind_,
        cu_seqlens_, 
        intermediate_states_buffer_,
        intermediate_state_indices_,
        stream=stream,
    )
    o = o.squeeze(0)
    return o

cutedsl_fused_recurrent_gated_delta_rule_update.compile_cache = {}