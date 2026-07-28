# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# =============================================================================
# chunk_kda_impl.py — Kimi Delta Attention (chunked gated delta-rule linear
# attention, forward only). A single @pypto.frontend.jit TND-varlen kernel plus a
# host wrapper; fixed-length (BSND) input folds into the varlen path via synthesized
# cu_seqlens. See Chunk_KDA.md for the math, public ABI, and constraints.
# =============================================================================

import pypto
import torch
import torch_npu  # noqa: F401  required for NPU device init

_BT = 128   # chunk length (static)
_MIN = 16   # block-inverse leaf size: BT/8 = 16
_SCALE = 128 ** -0.5    # K=128 query scale, folded in fp32 in-kernel
_L2_EPS = 1e-6          # q/k L2-norm eps
_BC = 64                # intra-chunk decay sub-block (one local pivot per block)
_NC = _BT // _BC        # sub-blocks per 128-row chunk
_DECAY_CAP = 80.0       # clamp col-factor exponent below fp32 exp overflow (~88.7)


# =============================================================================
# 8x16 block matrix inverse. Inverts (I + A) where A is strict-lower: feed neg-A so
# the unit-lower forward-substitution returns (I-(-A))^-1 = (I+A)^-1. Hierarchical
# merge: 8 leaf 16x16 -> 32 -> 64 -> 128.
# =============================================================================
def _inv_min_length(attn_dim1, eye, row_num, col_num):
    # 8 stacked diag 16x16 blocks [16,128] -> their inverses [16,128] via fwd-subst.
    size = col_num // row_num                                   # 8
    pypto.set_vec_tile_shapes(128, 128)
    cur = attn_dim1[:2, :]                                      # [2,128] rows 0,1
    for i in range(2, row_num, 1):                             # 2..15
        cur = cur + 0.0                                        # land in UB
        row = attn_dim1.view([1, col_num], [i, 0])            # [1,128] row i across blocks
        row_exp = row.reshape([size, row_num]).view([size, i], [0, 0]).transpose(1, 0).reshape([size * i, 1])
        cur_r = cur.reshape([size * i, row_num])              # [8i,16]
        prod_mul = (row_exp * cur_r).reshape([i, col_num])    # [i,128]
        prod = pypto.sum(prod_mul, dim=0, keepdim=True)       # [1,128]
        cur = pypto.concat([cur, row + prod], dim=0)         # grow [i+1,128]
    return cur + eye                                          # [16,128] add diag I


def _inv_merge(attn, inv11, inv22, x_ofs, y_ofs, m_len, zero_t):
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
    a21 = attn.view([m_len, m_len], [x_ofs + m_len, y_ofs])   # [m,m] lower-left
    a21_inv = pypto.matmul(pypto.matmul(inv22, a21, pypto.DT_FP32), inv11, pypto.DT_FP32)  # D_inv C A_inv
    # assemble the [2m,2m] block matrix [[inv11, 0], [a21_inv, inv22]] via concat;
    # zero_t supplies the [m,m] upper-right zero block
    pypto.set_vec_tile_shapes(128, 128)
    top = pypto.concat([inv11, zero_t], dim=1)                # [m,2m] upper: inv11 | 0
    bot = pypto.concat([a21_inv, inv22], dim=1)               # [m,2m] lower: a21_inv | inv22
    return pypto.concat([top, bot], dim=0)                    # [2m,2m]


def _inverse8(neg_a, eye_stack, z8, z16, z32):
    pypto.set_vec_tile_shapes(128, 128)
    diag = [neg_a.view([_MIN, _MIN], [_MIN * i, _MIN * i]) + 0.0 for i in range(8)]  # 8x[16,16]
    diag_inv = _inv_min_length(pypto.concat(diag, dim=1), eye_stack, _MIN, _BT)      # [16,128]
    li = [diag_inv[:, _MIN * i:_MIN * (i + 1)] + 0.0 for i in range(8)]              # 8x[16,16]
    l4 = [_inv_merge(neg_a, li[2 * i], li[2 * i + 1], _MIN * 2 * i, _MIN * 2 * i, _MIN, z8) for i in range(4)]
    l2 = [_inv_merge(neg_a, l4[2 * i], l4[2 * i + 1], _MIN * 4 * i, _MIN * 4 * i, _MIN * 2, z16) for i in range(2)]
    return _inv_merge(neg_a, l2[0], l2[1], 0, 0, _MIN * 4, z32)                      # [128,128]


# =============================================================================
# Intra-chunk decay A (k-side) / A_qk (q-side). Splits the [128,128] causal matrix
# into _NC row-blocks of _BC rows, each with a LOCAL pivot gp = gcum at the block-start
# row: the row factor exp(gcum_row-gp) has exponent <=0 and the col factor
# exp(gp-gcum_col) is clamped by _DECAY_CAP, so the reconstructed decay stays finite.
# Returns both UNMASKED; _chunk_compute applies the tril masks.
# =============================================================================
def _intra_chunk_a(q_s, k_s, gcum):
    pypto.set_vec_tile_shapes(_BC, 128)                       # row [BC,K] & col [128,K] elementwise
    pypto.set_cube_tile_shapes([_BC, _BC], [128, 128], [_BT, _BT])  # [BC,K]@[K,128]->[BC,128] band matmul
    af_bands = []
    a2_bands = []
    for a in range(_NC):                                      # _NC row-blocks of _BC rows (static loop)
        r0 = a * _BC
        gp = gcum[r0:r0 + 1, :]                               # [1,K] pivot = gcum at block-start row
        g_band = gcum[r0:r0 + _BC, :]                         # [BC,K]
        row_dec = pypto.exp(g_band - gp)                      # [BC,K] exponent <=0 (rows >= pivot)
        # A / A_qk matmuls stay fp32 (extreme decay dynamic range would round in bf16)
        kg_a = k_s[r0:r0 + _BC, :] * row_dec                  # [BC,K] k-side row factor (fp32)
        qg_a = q_s[r0:r0 + _BC, :] * row_dec                  # [BC,K] q-side row factor (fp32)
        col_dec = pypto.exp(pypto.minimum(gp - gcum, _DECAY_CAP))  # [128,K] clamp -> no +inf in future region
        ki_a = k_s * col_dec                                  # [128,K] shared col factor (fp32, feeds both matmuls)
        af_bands.append(pypto.matmul(kg_a, ki_a, pypto.DT_FP32, b_trans=True))  # [BC,128] fp32 (extreme decay range)
        a2_bands.append(pypto.matmul(qg_a, ki_a, pypto.DT_FP32, b_trans=True))  # [BC,128] fp32 (extreme decay range)
    a_full = pypto.concat(af_bands, dim=0)                    # [128,128] (k-side, unmasked)
    a2 = pypto.concat(a2_bands, dim=0)                        # [128,128] (q-side, unmasked)
    return a_full, a2


# =============================================================================
# Per-chunk M1+M2+M3 compute. Plain helper (no pypto.is_loop_end); takes the
# non-tensor bool use_qk_l2norm_in_kernel so it is NOT @pypto.frontend.function-
# decorated. Called from both is_loop_end branches. Sets its own cube+vec tiles.
# Returns (oc, s_new); caller writes s_carry[:] = s_new.
# =============================================================================
def _chunk_compute(qc, kc, vc, gc, bc, s_carry, tril_incl, tril_strict, eye_stack,
                   use_qk_l2norm_in_kernel, scale, z8, z16, z32):
    K = qc.shape[1]                                       # 128 (static); for glast row-(_BT-1) view
    pypto.set_cube_tile_shapes([64, 128], [128, 128], [128, 128])   # M1/M2 cube (self-contained)
    pypto.set_vec_tile_shapes(128, 128)                            # M1 vec
    q_f = pypto.cast(qc, pypto.DT_FP32)                   # [128,K] fp32, pre-scale
    k_f = pypto.cast(kc, pypto.DT_FP32)                   # [128,K] fp32
    # --- in-kernel q/k L2-norm over K, applied BEFORE scale (trace-time `if`) ---
    if use_qk_l2norm_in_kernel:
        q_f = q_f * pypto.rsqrt(pypto.sum(q_f * q_f, dim=1, keepdim=True) + _L2_EPS)  # [128,K] per-token over K
        k_f = k_f * pypto.rsqrt(pypto.sum(k_f * k_f, dim=1, keepdim=True) + _L2_EPS)  # [128,K] per-token over K
    q_s = q_f * scale                                     # [128,K] q*scale (fp32, after l2norm; scale is a trace-time const)
    k_s = k_f                                             # [128,K] k (unscaled, l2normed)
    # M1: per-channel gated cumsum
    gcum = pypto.matmul(tril_incl, gc, pypto.DT_FP32)     # [128,K]
    eg = pypto.exp(gcum)                                  # [128,K] exp(gcum) shared by kg (M2 w/u) and qg (M3 o)
    kg = k_s * eg                                         # [128,K] (cross-chunk state term w/u; NOT intra A)
    # M2: intra-chunk A (k-side) and A_qk (q-side), returned unmasked; masks applied here
    a_full, a2_raw = _intra_chunk_a(q_s, k_s, gcum)       # [128,128], [128,128] (stable, unmasked)
    pypto.set_vec_tile_shapes(128, 128)                  # restore caller vec tile (_intra_chunk_a set its own)
    b_row = bc.transpose(0, 1)                            # [128,1] (bc already fp32)
    a = a_full * b_row * tril_strict                      # [128,128]
    neg_a = pypto.neg(a)
    a_inv = _inverse8(neg_a, eye_stack, z8, z16, z32)     # [128,128]
    a_inv = a_inv * bc
    pypto.set_vec_tile_shapes(128, 128)
    pypto.set_cube_tile_shapes([64, 128], [128, 128], [128, 128])
    # --- bf16 matmul inputs (Cube fast path), fp32 accumulation ---
    a_inv_bf = pypto.cast(a_inv, pypto.DT_BF16)          # [128,128] feeds w & u matmuls
    kg_bf = pypto.cast(kg, pypto.DT_BF16)                # [128,K] kg=k_s*exp(gcum): fp32 then bf16
    w = pypto.matmul(a_inv_bf, kg_bf, pypto.DT_FP32)     # [128,K] bf16 in, fp32 accum
    u = pypto.matmul(a_inv_bf, vc, pypto.DT_FP32)        # [128,V] bf16 in (vc already bf16), fp32 accum
    # M3: cross-chunk recurrence (state S carried over the chunk loop)
    qg = q_s * eg                                         # [128,K] reuse exp(gcum) from M1 (fp32; cast bf16 below)
    a2 = a2_raw * tril_incl                               # [128,128] (stable q-side A, inclusive-lower)
    s_carry_bf = pypto.cast(s_carry, pypto.DT_BF16)      # [K,V] feeds w@s_carry & qg@s_carry; fp32 s_carry kept for s_new
    w_bf = pypto.cast(w, pypto.DT_BF16)                  # [128,K] w=a_inv@kg: fp32 then bf16
    vi = u - pypto.matmul(w_bf, s_carry_bf, pypto.DT_FP32)   # [128,V] fp32 (u & matmul both fp32)
    qg_bf = pypto.cast(qg, pypto.DT_BF16)                # [128,K] qg=q_s*exp(gcum): fp32 then bf16
    a2_bf = pypto.cast(a2, pypto.DT_BF16)                # [128,128] fp32 -> bf16
    vi_bf = pypto.cast(vi, pypto.DT_BF16)               # [128,V] feeds a2@vi & kdec@vi -> cast once
    oc = pypto.matmul(qg_bf, s_carry_bf, pypto.DT_FP32) + pypto.matmul(a2_bf, vi_bf, pypto.DT_FP32)  # [128,V] fp32 accum
    glast = pypto.view(gcum, [1, K], [_BT - 1, 0])       # [1,K] row _BT-1 == gcum[actual_l-1] (g pad=0 -> flat)
    kdec = k_s * pypto.exp(glast - gcum)                  # [128,K] (fp32; cast to bf16 below)
    kdec_bf = pypto.cast(kdec, pypto.DT_BF16)            # [128,K] kdec=k_s*exp(glast-gcum): fp32 then bf16
    s_new = s_carry * pypto.exp(glast).transpose(0, 1) + pypto.matmul(kdec_bf, vi_bf, pypto.DT_FP32, a_trans=True)  # [K,V] fp32
    return oc, s_new


# =============================================================================
# JIT entry — the single module-level @pypto.frontend.jit (TND varlen; fixed-len
# BSND folds in via cu synthesis in the host). K==V==128 baked. Tensors first, then
# the trace-time bool. The whole loop nest is inlined here so pypto.is_loop_end lives
# directly in the @jit body.
# =============================================================================
@pypto.frontend.jit(
        runtime_options={"run_mode": pypto.RunMode.NPU, "launch_sched_aicpu_num": 3},
        pass_options={
            "vec_nbuffer_setting": {-2: 1, -1: 64},
            "cube_l1_reuse_setting": {-1: 4},
            "cube_nbuffer_setting": {-1: 4}
            },
        debug_options={
            "runtime_debug_mode": 0
        })
def chunk_kda_varlen_kernel(
    q:      pypto.Tensor([1, pypto.DYNAMIC, pypto.DYNAMIC, 128], pypto.DT_BF16),  # [1,T,H,K] packed  # SNAP:SIG_JIT
    k:      pypto.Tensor([1, pypto.DYNAMIC, pypto.DYNAMIC, 128], pypto.DT_BF16),  # [1,T,H,K]
    v:      pypto.Tensor([1, pypto.DYNAMIC, pypto.DYNAMIC, 128], pypto.DT_BF16),  # [1,T,H,V]
    g:      pypto.Tensor([1, pypto.DYNAMIC, pypto.DYNAMIC, 128], pypto.DT_FP32),  # [1,T,H,K] fp32
    beta:   pypto.Tensor([1, pypto.DYNAMIC, pypto.DYNAMIC],      pypto.DT_FP32),  # [1,T,H] fp32
    cu:     pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT32),  # [N+1] segment boundaries (indexed in-kernel)
    tril:   pypto.Tensor([128, 128], pypto.DT_FP32),  # cumsum incl
    trils:  pypto.Tensor([128, 128], pypto.DT_FP32),  # strict-lower mask
    eyestk: pypto.Tensor([16, 128],  pypto.DT_FP32),  # block-inv diag I
    state0: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, 128, 128], pypto.DT_FP32),  # [N,H,V,K] upstream ABI (transpose in-kernel)
    o_out:  pypto.Tensor([1, pypto.DYNAMIC, pypto.DYNAMIC, 128], pypto.DT_BF16),  # [1,T,H,V]
    s_out:  pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, 128, 128], pypto.DT_FP32),  # [N,H,V,K] upstream ABI (transpose in-kernel)
    use_qk_l2norm_in_kernel: bool = False,  # non-tensor param (after all tensors); trace-time switch
    scale: float = _SCALE,  # non-tensor param; trace-time query-scale constant (folded in fp32 in-kernel)
):
    # H/N are derived in-kernel from tensor dims (not int params) so all (H,N) shapes
    # reuse one compile; use_qk_l2norm_in_kernel and scale are the only trace-time
    # (non-tensor) params -> they key the compile cache (scale is realistically constant).
    pypto.experimental.set_operation_options(combine_axis=True)  # inline brcb for tail-1 beta broadcasts
    K = q.shape[3]                                        # K == 128
    V = v.shape[3]                                        # V == 128
    H = q.shape[2]                                        # head count, derived from tensor dim (q [1,T,H,K])
    N = state0.shape[0]                                   # seq count,  derived from tensor dim (state0 [N,H,V,K])
    # z8/z16/z32: the _inv_merge upper-right zero quadrants ([16,16]/[32,32]/[64,64]),
    # loop-invariant so built once per launch and threaded into _chunk_compute/_inverse8.
    pypto.set_vec_tile_shapes(16, 64)
    z8 = pypto.full(size=[_MIN, _MIN], fill_value=0.0, dtype=pypto.DT_FP32)
    z16 = pypto.full(size=[_MIN * 2, _MIN * 2], fill_value=0.0, dtype=pypto.DT_FP32)
    z32 = pypto.full(size=[_MIN * 4, _MIN * 4], fill_value=0.0, dtype=pypto.DT_FP32)
    # SNAP:before_nt_loop
    for h in pypto.loop(0, H, 1, name="LH"):        # head loop
        for n in pypto.loop(0, N, 1, name="LN"):    # sequence loop
            bos = cu[n]                                   # seq start token (indexed from device cu tensor)
            eos = cu[n + 1]                               # seq end token (exclusive)
            L = eos - bos                                 # seq length (SymbolicScalar; may be %128 != 0)
            pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
            pypto.set_vec_tile_shapes(1, 1, 128, 128)       # rank-4 for state0[N,H,V,K] view (upstream ABI)
            s_seed_vk = pypto.reshape(pypto.view(state0, [1, 1, V, K], [n, h, 0, 0]), [V, K])  # [V,K] upstream ABI slice
            pypto.set_vec_tile_shapes(128, 128)             # rank-2 [V,K]->[K,V] transpose (V==K==128)
            s_carry = s_seed_vk.transpose(0, 1) + 0.0       # [V,K] -> [K,V] internal seed (transpose in-kernel)
            pypto.set_vec_tile_shapes(128, 128)             # back to rank-2
            # dynamic-bound static-step chunk loop (step _BT); last iteration is the
            # possibly-partial tail
            for s_idx in pypto.loop(0, L, _BT, name="LNT", submit_before_loop=False, unroll_list=[4]):
                # SNAP:inside_nt_loop
                off = bos + s_idx                          # absolute token offset of this chunk
                actual_l = (L - s_idx).min(_BT)            # valid rows in THIS chunk (<= _BT); tail may be < _BT
                # ---- rank-4 strided head-slice views; valid_shape marks the actual rows.
                #      Cube tile for the M1/M2 matmuls is set inside _chunk_compute. ----
                pypto.set_vec_tile_shapes(1, 128, 1, 128)     # rank-4 q/k/v/g chunk views
                q4 = pypto.view(q, [1, _BT, 1, K], [0, off, h, 0], valid_shape=[1, actual_l, 1, K])
                k4 = pypto.view(k, [1, _BT, 1, K], [0, off, h, 0], valid_shape=[1, actual_l, 1, K])
                v4 = pypto.view(v, [1, _BT, 1, V], [0, off, h, 0], valid_shape=[1, actual_l, 1, V])
                g4 = pypto.view(g, [1, _BT, 1, K], [0, off, h, 0], valid_shape=[1, actual_l, 1, K])
                q2 = pypto.reshape(q4, [_BT, K], valid_shape=[actual_l, K])   # [128,K], rows>=actual_l are pad
                k2 = pypto.reshape(k4, [_BT, K], valid_shape=[actual_l, K])
                v2 = pypto.reshape(v4, [_BT, V], valid_shape=[actual_l, V])
                g2 = pypto.reshape(g4, [_BT, K], valid_shape=[actual_l, K])
                pypto.set_vec_tile_shapes(1, 128)             # rank-2 beta view/reshape
                b3 = pypto.view(beta, [1, _BT, 1], [0, off, h], valid_shape=[1, actual_l, 1])
                b2 = pypto.reshape(b3, [1, _BT], valid_shape=[1, actual_l])   # [1,128], cols>=actual_l are pad
                # ---- tail branch: fillpad zeros the pad region of the last, possibly-
                #      partial chunk; full chunks (actual_l==_BT) skip fillpad. ----
                if pypto.is_loop_end(s_idx):
                    pypto.set_vec_tile_shapes(128, 128)                      # q2/k2/v2/g2 are [128,128]
                    qc = pypto.fillpad(q2, "constant", 0.0)                  # rows [actual_l:_BT] -> 0.0
                    kc = pypto.fillpad(k2, "constant", 0.0)
                    vc = pypto.fillpad(v2, "constant", 0.0)
                    gc = pypto.fillpad(g2, "constant", 0.0)                  # KEY: g pad=0 -> gcum flat (glast valid)
                    pypto.set_vec_tile_shapes(1, 128)                       # b2 is [1,128]
                    bc = pypto.fillpad(b2, "constant", 0.0)                  # cols [actual_l:_BT] -> 0.0
                    oc, s_new = _chunk_compute(qc, kc, vc, gc, bc, s_carry, tril, trils, eyestk,
                                               use_qk_l2norm_in_kernel, scale, z8, z16, z32)
                    s_carry[:] = s_new                                       # carry state across chunks
                    pypto.set_vec_tile_shapes(1, 128, 1, 128)               # rank-4 o reshape+assemble
                    o_chunk = pypto.reshape(pypto.cast(oc, pypto.DT_BF16), [1, _BT, 1, V],
                                            valid_shape=[1, actual_l, 1, V])  # only actual_l rows valid
                    pypto.assemble(o_chunk, [0, off, h, 0], o_out)          # writes o_out[0, off:off+actual_l, h, :]
                else:
                    oc, s_new = _chunk_compute(q2, k2, v2, g2, b2, s_carry, tril, trils, eyestk,
                                               use_qk_l2norm_in_kernel, scale, z8, z16, z32)
                    s_carry[:] = s_new
                    pypto.set_vec_tile_shapes(1, 128, 1, 128)               # rank-4 o reshape+assemble
                    o_chunk = pypto.reshape(pypto.cast(oc, pypto.DT_BF16), [1, _BT, 1, V])  # full 128 rows
                    pypto.assemble(o_chunk, [0, off, h, 0], o_out)          # non-tail chunk is always full
                pypto.set_vec_tile_shapes(128, 128)
            pypto.set_vec_tile_shapes(128, 128)                             # rank-2 [K,V]->[V,K] transpose
            s_carry_vk = s_carry.transpose(0, 1)                            # [K,V] -> [V,K] upstream ABI (transpose IN-KERNEL)
            pypto.set_vec_tile_shapes(1, 1, 128, 128)                       # rank-4 reshape+assemble
            s_chunk = pypto.reshape(s_carry_vk, [1, 1, V, K])               # [N,H,V,K] upstream ABI
            pypto.assemble(s_chunk, [n, h, 0, 0], s_out)
    # SNAP:after_nt_loop


# =============================================================================
# Host wrapper — layout/alloc only, one jit call. cu_seqlens=None -> synthesize N==B
# segments of length T and pack [B,T,H,*]->[1,B*T,H,*] (state [B,H,V,K]); cu_seqlens
# given -> TND pass-through, cu kept on-device and indexed in-kernel (state [N,H,V,K]).
# No host padding: the partial last chunk is zero-filled in-kernel by fillpad.
# initial_state is passed directly in upstream ABI [N,H,V,K] (the [V,K]<->[K,V]
# transpose is done in-kernel). _validate_inputs does introspection-only validation.
# =============================================================================
def _validate_inputs(q, k, v, g, beta, scale, initial_state, output_final_state,
                     use_qk_l2norm_in_kernel, cu_seqlens):
    """Introspection-only host validation. Rejects malformed inputs early; derives dims
    internally for shape checks. Returns nothing (wrapper re-derives dims)."""
    # Introspection only (.dim()/.shape/.dtype/.device/.numel()); no tensor arithmetic
    # and no cu_seqlens.cpu()/.tolist() (cu is a device tensor indexed in-kernel, so its
    # values are not validated here).
    assert q.dim() == 4, f"q must be 4-D [B,T,H,K]; got dim {q.dim()}"
    B, T, H, K = q.shape
    V = v.shape[-1]
    dev = q.device
    assert k.dim() == 4 and g.dim() == 4, \
        f"k/g must be 4-D [B,T,H,K]; got dims {k.dim()}/{g.dim()}"
    assert v.dim() == 4, f"v must be 4-D [B,T,H,V]; got dim {v.dim()}"
    assert beta.dim() == 3, f"beta must be 3-D [B,T,H]; got dim {beta.dim()}"
    assert tuple(k.shape) == tuple(q.shape) and tuple(g.shape) == tuple(q.shape), \
        f"k/g shape must equal q[B,T,H,K]; got k{tuple(k.shape)} g{tuple(g.shape)} vs q{tuple(q.shape)}"
    assert tuple(v.shape[:3]) == (B, T, H), \
        f"v.shape[:3] must be (B,T,H)=({B},{T},{H}); got {tuple(v.shape[:3])}"
    assert tuple(beta.shape) == (B, T, H), \
        f"beta.shape must be (B,T,H)=({B},{T},{H}); got {tuple(beta.shape)}"
    # Exact kernel-ABI dtypes (wrapper no longer casts — caller guarantees types).
    assert q.dtype == torch.bfloat16 and k.dtype == torch.bfloat16 and v.dtype == torch.bfloat16, \
        f"q/k/v must be bfloat16 (kernel ABI); got {q.dtype}/{k.dtype}/{v.dtype}"
    assert g.dtype == torch.float32, f"g must be float32 (kernel ABI); got {g.dtype}"
    assert beta.dtype == torch.float32, f"beta must be float32 (kernel ABI); got {beta.dtype}"
    assert k.device == dev and v.device == dev and g.device == dev and beta.device == dev, \
        f"q/k/v/g/beta must share one device; got q{dev} k{k.device} v{v.device} g{g.device} beta{beta.device}"
    assert scale is None or (isinstance(scale, (int, float)) and not isinstance(scale, bool) and scale > 0), \
        f"scale must be a positive number or None; got {scale!r}"
    assert isinstance(output_final_state, (bool, int)), \
        f"output_final_state must be bool; got {type(output_final_state).__name__}"
    if initial_state is not None:
        assert initial_state.dim() == 4, \
            f"initial_state must be 4-D [N,H,V,K]; got dim {initial_state.dim()}"
        _N0 = B if cu_seqlens is None else (cu_seqlens.numel() - 1)
        assert tuple(initial_state.shape) == (_N0, H, V, K), \
            f"initial_state must be (N,H,V,K)=({_N0},{H},{V},{K}); got {tuple(initial_state.shape)}"
        assert initial_state.dtype == torch.float32, \
            f"initial_state must be float32 (kernel ABI); got {initial_state.dtype}"
    if cu_seqlens is not None:
        assert cu_seqlens.dim() == 1, f"cu_seqlens must be 1-D [N+1]; got dim {cu_seqlens.dim()}"
        assert cu_seqlens.dtype == torch.int32, \
            f"cu_seqlens must be int32 (kernel ABI); got {cu_seqlens.dtype}"
        assert cu_seqlens.numel() >= 2, f"cu_seqlens must have >=2 entries; got {cu_seqlens.numel()}"
    assert K == 128 and V == 128, "chunk_kda P0: only K==V==128 supported"


def chunk_kda_wrapper(q, k, v, g, beta, scale=None, initial_state=None,
                      output_final_state=False, use_qk_l2norm_in_kernel=False,
                      cu_seqlens=None, **kwargs):
    # Explicit forward ABI (upstream chunk_kda parity); chunk_size is fixed at _BT=128,
    # so a stray prebuilt_meta / chunk_size in **kwargs is silently ignored.
    use_qk_l2norm_in_kernel = bool(use_qk_l2norm_in_kernel)
    _validate_inputs(q, k, v, g, beta, scale, initial_state, output_final_state,
                     use_qk_l2norm_in_kernel, cu_seqlens)
    B, T, H, K = q.shape
    V = v.shape[-1]
    dev = q.device
    if scale is None:
        scale = K ** -0.5
    scale = float(scale)   # threaded into the JIT as a trace-time constant (folded in fp32 in-kernel)
    Ttot = B * T
    # ---- segment boundaries: cu stays a device int32 tensor, indexed in-kernel.
    #      Fixed-len -> synthesize N==B segments of length T; varlen -> pass through. ----
    if cu_seqlens is None:
        assert T % _BT == 0, "T must be a multiple of chunk_size when cu_seqlens=None"
        N = B
        cu = torch.arange(0, (B + 1) * T, T, dtype=torch.int32, device=dev)   # [0,T,2T,...,B*T]
    else:
        assert B == 1, "varlen (cu_seqlens) requires packed batch size 1"
        cu = cu_seqlens                                                       # PASS-THROUGH device int32 tensor (validated, no cast)
        N = cu_seqlens.numel() - 1                                            # python int (introspection)
    # pack [B,T,H,*] -> [1,Ttot,H,*], reshape only (no dtype cast, no host padding)
    qk = q.reshape(1, Ttot, H, K)                                      # [1,Ttot,H,K] bf16
    kk = k.reshape(1, Ttot, H, K)
    vk = v.reshape(1, Ttot, H, V)
    gk = g.reshape(1, Ttot, H, K)                                      # [1,Ttot,H,K] fp32
    bk = beta.reshape(1, Ttot, H)                                      # [1,Ttot,H] fp32
    # in-kernel constants (arange masks + [16,128] block-diag I)
    idx = torch.arange(_BT, device=dev)
    tril = (idx.reshape(_BT, 1) >= idx.reshape(1, _BT)).float()        # inclusive lower-tri (cumsum)
    trils = (idx.reshape(_BT, 1) > idx.reshape(1, _BT)).float()        # strict lower-tri (A mask)
    eyestk = torch.eye(_MIN, device=dev, dtype=torch.float32).repeat(1, 8)  # [16,128] block-diag I
    # pass initial_state directly in upstream ABI [N,H,V,K] (the [V,K]->[K,V] transpose
    # is done in-kernel); None -> a zeros seed in [N,H,V,K].
    if initial_state is not None:
        state_in = initial_state.contiguous()                          # [N,H,V,K] fp32 (validated; contiguity only, no cast)
    else:
        state_in = torch.zeros(N, H, V, K, dtype=torch.float32, device=dev)  # [N,H,V,K] zero seed
    # HOST_WRAPPER_INSPECT_ALLOC
    o_out = torch.zeros(1, Ttot, H, V, dtype=torch.bfloat16, device=dev)   # [1,Ttot,H,V] (no pad; Tp==Ttot)
    s_out = torch.zeros(N, H, V, K, dtype=torch.float32, device=dev)       # [N,H,V,K] upstream ABI (transpose in-kernel)
    # HOST_WRAPPER_INSPECT_PASS — single statically-reachable module-level JIT
    chunk_kda_varlen_kernel(qk, kk, vk, gk, bk, cu, tril, trils, eyestk, state_in, o_out, s_out,
                            use_qk_l2norm_in_kernel, scale)
    o = o_out.reshape(B, T, H, V)          # [1,Ttot,H,V] -> [B,T,H,V] (host view)
    # final_state already in upstream ABI [N,H,V,K] (internal [K,V]->[V,K] transpose done in-kernel)
    S = s_out if output_final_state else None
    return o, S
