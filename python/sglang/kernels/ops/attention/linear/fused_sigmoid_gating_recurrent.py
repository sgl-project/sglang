from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.kernels.jit.utils import is_arch_support_pdl
from sglang.srt.utils import (
    get_device_sm,
    is_cuda,
    is_gfx95_supported,
    is_hip,
    is_sm90_supported,
)

_is_hip = is_hip()
_is_gfx95 = is_gfx95_supported()
_is_sm90 = is_sm90_supported()
_is_sm103 = is_cuda() and get_device_sm() == 103


def _select_recurrent_launch_config(
    n: int,
    h: int,
    hv: int,
    k: int,
    v: int,
    is_kda: bool,
    target_verify: bool = False,
    *,
    cache_steps: int = 0,
) -> tuple[int, int]:
    """Select the value tile and warp count for recurrent GDN/KDA."""
    if (
        _is_hip
        and _is_gfx95
        and not is_kda
        and 0 < n <= 32
        and h == 4
        and hv == 16
        and k == 128
        and v == 128
    ):
        return (8, 4) if n == 1 else (16, 2)
    if (
        target_verify
        and not _is_hip
        and not is_kda
        and 0 < n <= 64
        and k == 128
        and v == 128
        and _is_sm90
    ):
        # BV=4 and n <= 64 measured on H100/H200. SM90 only: Blackwell is faster
        # with narrow tiles but not bit-identical to BV=32. Only the dense
        # intermediate-state verify sets target_verify; cache_ring keeps BV=32.
        return 4, 1
    if (
        _is_sm103
        and is_kda
        and target_verify
        and (n, h, hv, k, v, cache_steps) == (1, 8, 8, 128, 128, 8)
    ):
        # GLM TP8 DFlash verification: 256 CTAs instead of 32.
        return 4, 1
    return min(triton.next_power_of_2(v), 32), 1


_EX2_APPROX_FTZ_F32 = tl.constexpr("ex2.approx.ftz.f32 $0, $1;")
_LG2_APPROX_FTZ_F32 = tl.constexpr("lg2.approx.ftz.f32 $0, $1;")
_UNARY_F32_REGISTERS = tl.constexpr("=f,f")


@triton.jit
def _ex2_approx_ftz_f32(x):
    return tl.inline_asm_elementwise(
        asm=_EX2_APPROX_FTZ_F32,
        constraints=_UNARY_F32_REGISTERS,
        args=(x,),
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _lg2_approx_ftz_f32(x):
    return tl.inline_asm_elementwise(
        asm=_LG2_APPROX_FTZ_F32,
        constraints=_UNARY_F32_REGISTERS,
        args=(x,),
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _cutedsl_exp(x):
    return _ex2_approx_ftz_f32(x * 1.4426950408889634)


@triton.jit
def _cutedsl_qk_sum_squares(x, BK: tl.constexpr):
    """Lane-striped Q/K sum of squares, as FlashInfer's ``recurrent_kda`` CuTe
    DSL decode kernel reduces it: lane ``l`` owns the elements ``l + i * 32``
    and a 32-lane butterfly combines the lanes. Keeping both reduction
    dimensions explicit stops Triton from flattening K into a different
    reduction tree.
    """
    NS: tl.constexpr = BK // 32
    tl.static_assert(
        NS >= 1 and NS <= 4,
        "match_cutedsl_decode needs 32 <= BK <= 128 (head_k_dim <= 128)",
    )
    # The decode kernel accumulates its per-lane squares with packed f32x2 FMA,
    # so the even and odd i streams stay separate until both are complete. A
    # plain tl.sum(x * x) is mathematically equivalent, but can move a BF16
    # recurrent state across a rounding boundary after several verify steps.
    lane_values = tl.reshape(x, (NS, 32), can_reorder=False)
    row = tl.arange(0, NS)[:, None]
    even0 = tl.sum(tl.where(row == 0, lane_values, 0.0), axis=0)
    odd0 = tl.sum(tl.where(row == 1, lane_values, 0.0), axis=0)
    zeros = tl.zeros((32,), tl.float32)
    even_sum = tl.fma(even0, even0, zeros)
    odd_sum = tl.fma(odd0, odd0, zeros)
    if NS == 4:
        even1 = tl.sum(tl.where(row == 2, lane_values, 0.0), axis=0)
        odd1 = tl.sum(tl.where(row == 3, lane_values, 0.0), axis=0)
        even_sum = tl.fma(even1, even1, even_sum)
        odd_sum = tl.fma(odd1, odd1, odd_sum)
    lane_sum = even_sum + odd_sum
    return tl.sum(lane_sum, axis=0)


@triton.jit(do_not_specialize=["T"])
def fused_sigmoid_gating_delta_rule_update_kernel(
    A_log,
    a,
    dt_bias,
    softplus_beta,
    softplus_threshold,
    lower_bound,
    q,
    k,
    v,
    b,
    o,
    h0_source,
    h0_indices,
    stride_h0_source,
    cu_seqlens,
    # Parameters for target_verify support (unused for decode)
    intermediate_states_buffer,
    intermediate_state_indices,
    cache_steps,
    retrieve_parent_token_ptr,
    stride_retrieve_parent_token_seq: tl.constexpr,
    stride_retrieve_parent_token_token: tl.constexpr,
    # ================================================
    scale,
    T,
    stride_a,
    stride_q,
    stride_k,
    stride_v,
    stride_b,
    NP2_T: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_KDA: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
    # Optional flags for target_verify support (default False for decode)
    DISABLE_STATE_UPDATE: tl.constexpr = False,
    CACHE_INTERMEDIATE_STATES: tl.constexpr = False,
    HAS_EAGLE_TREE_CUSTOM_ATTN_MASK: tl.constexpr = False,
    # ReplaySSM fused ring-write. Pointers stay None and CACHE_RING False for
    # decode / flag-off -> byte-identical. The gate ring layout follows IS_KDA
    # (see the store below).
    replayssm_rawv=None,
    replayssm_rawk=None,
    replayssm_g=None,
    replayssm_beta=None,
    stride_rawv_slot: tl.constexpr = 0,
    stride_rawk_slot: tl.constexpr = 0,
    stride_g_slot: tl.constexpr = 0,
    stride_beta_slot: tl.constexpr = 0,
    MAX_CACHE_LEN: tl.constexpr = 0,
    CACHE_RING: tl.constexpr = False,
    SPLIT_N_HV_GRID: tl.constexpr = False,
    USE_GDC: tl.constexpr = False,
    MATCH_CUTEDSL_DECODE: tl.constexpr = False,
    ROUND_STATE_TO_BF16: tl.constexpr = False,
):
    """
    Fused kernel that combines sigmoid gating computation with recurrent delta rule update.
    """
    if SPLIT_N_HV_GRID:
        i_v, i_n, i_hv = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        # The GPU wrapper asserts NK == 1. Keep N and HV on independent grid
        # axes so large GLM5 decode batches do not exceed CUDA's grid limit.
        i_k = 0
    else:
        i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        i_n, i_hv = i_nh // HV, i_nh % HV
    # PDL: overlap this kernel's prologue with the producer (the KDA/GDN
    # conv1d_update). All global loads below happen after the wait, so
    # numerics are unchanged. The immediate trigger releases the LAUNCH of
    # the next PDL kernel so its prologue overlaps this whole body;
    # consumers' own gdc_wait still fences on full completion.
    if USE_GDC:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()

    i_h = i_hv // (HV // H)

    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        all = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all = B * T

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    p_q = q + bos * stride_q + i_h * K + o_k
    p_k = k + bos * stride_k + i_h * K + o_k
    p_v = v + bos * stride_v + i_hv * V + o_v
    p_b = b + bos * stride_b + i_hv
    p_o = o + ((i_k * all + bos) * HV + i_hv) * V + o_v

    # Gating computation pointers
    p_A_log = A_log + i_hv
    if IS_KDA:
        p_a = a + bos * stride_a + i_hv * K + o_k
        p_dt_bias = dt_bias + i_hv * K + o_k
    else:
        p_a = a + bos * stride_a + i_hv
        p_dt_bias = dt_bias + i_hv

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        # Slot stride comes from the caller (h0_source.stride(0)): the state pool
        # may be an envelope-strided view (page-major / unified memory), where the
        # per-slot pitch spans ALL layers' state, not HV*K*V. int64: envelope
        # pitches overflow an int32 index product.
        idx = tl.load(h0_indices + i_n).to(tl.int64)
        if idx >= 0:
            p_h0 = (
                h0_source
                + idx * stride_h0_source
                + i_hv * K * V
                + o_v[None, :] * K
                + o_k[:, None]
            )
            b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    # Preload tree attention data if needed
    if HAS_EAGLE_TREE_CUSTOM_ATTN_MASK:
        token_indices = tl.arange(0, NP2_T)
        mask_retrieve = token_indices < T
        retrieve_parent_token_base = (
            retrieve_parent_token_ptr
            + (i_n * stride_retrieve_parent_token_seq)
            + token_indices * stride_retrieve_parent_token_token
        )
        parent_idx_tokens = tl.load(
            retrieve_parent_token_base, mask=mask_retrieve, other=0
        )

    # Prepare intermediate state cache index if enabled. int64: the buffer is
    # contiguous but `cache_idx * cache_steps * HV * K * V` can exceed int32 for
    # large slot counts.
    cache_idx = -1
    if CACHE_INTERMEDIATE_STATES:
        cache_idx = tl.load(intermediate_state_indices + i_n).to(tl.int64)

    step_idx = 0
    for _ in range(0, T):
        # Tree attention: load parent's cached state
        if HAS_EAGLE_TREE_CUSTOM_ATTN_MASK:
            # step_idx == 0 uses b_h from USE_INITIAL_STATE
            if step_idx != 0 and cache_idx >= 0:
                parent_step_idx = tl.sum(
                    tl.where(token_indices == step_idx, parent_idx_tokens, 0)
                )
                step_offset = parent_step_idx * HV * K * V
                cache_ptr = (
                    intermediate_states_buffer
                    + cache_idx * cache_steps * HV * K * V
                    + step_offset
                    + i_hv * K * V
                    + o_v[None, :] * K
                    + o_k[:, None]
                )
                b_h = tl.load(cache_ptr, mask=mask_h, other=0).to(tl.float32)

        # Load inputs
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_b = tl.load(p_b).to(tl.float32)

        # Compute sigmoid gating
        # Load gating parameters
        b_A_log = tl.load(p_A_log).to(tl.float32)
        if IS_KDA:
            b_a = tl.load(p_a, mask=mask_k, other=0).to(tl.float32)
            b_dt_bias = tl.load(p_dt_bias, mask=mask_k, other=0).to(tl.float32)
        else:
            b_a = tl.load(p_a).to(tl.float32)
            b_dt_bias = tl.load(p_dt_bias).to(tl.float32)

        x = b_a + b_dt_bias
        if USE_LOWER_BOUND:
            # KDA safe gate: lower_bound * sigmoid(exp(A_log) * (a + dt_bias))
            b_g = lower_bound * tl.sigmoid(tl.exp(b_A_log) * x)
        else:
            if MATCH_CUTEDSL_DECODE and IS_KDA:
                # The CuTe decode kernel forms the multiplicative gate directly
                # in base 2. Algebraically this is exp(-A * softplus(x)), but
                # the instruction sequence decides on which side of a BF16
                # state-rounding boundary the result lands.
                b_gate_multiplier = _ex2_approx_ftz_f32(
                    (-_cutedsl_exp(b_A_log))
                    * _lg2_approx_ftz_f32(1.0 + _cutedsl_exp(x))
                )
                b_g = tl.log(b_gate_multiplier)
            else:
                # Compute g = -exp(A_log) * softplus(a + dt_bias)
                beta_x = softplus_beta * x
                # Apply softplus with numerical stability
                softplus_x = tl.where(
                    beta_x <= softplus_threshold,
                    (1.0 / softplus_beta) * tl.log(1.0 + tl.exp(beta_x)),
                    x,
                )
                b_g = -tl.exp(b_A_log) * softplus_x

        # Compute beta = sigmoid(b)
        if MATCH_CUTEDSL_DECODE:
            b_beta = b_b
        else:
            b_beta = 1.0 / (1.0 + tl.exp(-b_b))

        # fused ring-write: stash this step's raw inputs + in-kernel gate/beta
        # into the per-slot ring for the commit fold to replay. Must sit here --
        # b_k is still pre-l2norm, b_v still pre-delta, b_g/b_beta are formed,
        # so the fold's replay is bit-identical to the update below. rawk uses
        # the k-head i_h (shared across a GQA group); rawv/g/beta use the v-head
        # i_hv. step_idx < MAX_CACHE_LEN: absorb-inflated rows can exceed the
        # ring; the overflow steps are past the committable prefix, so drop them
        # (writing them would smash the next slot's ring).
        if CACHE_RING:
            ring_slot = tl.load(h0_indices + i_n).to(tl.int64)
            if ring_slot >= 0 and step_idx < MAX_CACHE_LEN:
                tl.store(
                    replayssm_rawv
                    + ring_slot * stride_rawv_slot
                    + i_hv * MAX_CACHE_LEN * V
                    + step_idx * V
                    + o_v,
                    b_v.to(replayssm_rawv.dtype.element_ty),
                    mask=mask_v,
                )
                if i_v == 0:
                    tl.store(
                        replayssm_rawk
                        + ring_slot * stride_rawk_slot
                        + i_h * MAX_CACHE_LEN * K
                        + step_idx * K
                        + o_k,
                        b_k.to(replayssm_rawk.dtype.element_ty),
                        mask=mask_k,
                    )
                    # b_g follows IS_KDA: KDA loads a/dt_bias with mask_k, so the
                    # gate is a per-K vector and the ring row is K wide; GDN's is
                    # a scalar per (head, step). The two layouts are not
                    # interchangeable -- storing one into the other's stride is a
                    # shape error, not a slow path -- and memory_pool.py sizes
                    # replayssm_g off the same is_kda test.
                    if IS_KDA:
                        tl.store(
                            replayssm_g
                            + ring_slot * stride_g_slot
                            + i_hv * MAX_CACHE_LEN * K
                            + step_idx * K
                            + o_k,
                            b_g,
                            mask=mask_k,
                        )
                    else:
                        tl.store(
                            replayssm_g
                            + ring_slot * stride_g_slot
                            + i_hv * MAX_CACHE_LEN
                            + step_idx,
                            b_g,
                        )
                    if i_k == 0:
                        tl.store(
                            replayssm_beta
                            + ring_slot * stride_beta_slot
                            + i_hv * MAX_CACHE_LEN
                            + step_idx,
                            b_beta,
                        )

        # Apply L2 normalization if enabled
        if USE_QK_L2NORM_IN_KERNEL:
            if MATCH_CUTEDSL_DECODE:
                q_sum_sq = _cutedsl_qk_sum_squares(b_q, BK)
                k_sum_sq = _cutedsl_qk_sum_squares(b_k, BK)
                q_rsqrt = tl.rsqrt(q_sum_sq + 1e-6)
                k_rsqrt = tl.rsqrt(k_sum_sq + 1e-6)
            else:
                q_sum_sq = tl.sum(b_q * b_q)
                k_sum_sq = tl.sum(b_k * b_k)
                q_rsqrt = 1.0 / tl.sqrt(q_sum_sq + 1e-6)
                k_rsqrt = 1.0 / tl.sqrt(k_sum_sq + 1e-6)
            b_q *= q_rsqrt
            b_k *= k_rsqrt

        b_q = b_q * scale

        # Apply gating to hidden state: h *= exp(g)
        if IS_KDA:
            if MATCH_CUTEDSL_DECODE:
                # The CuTe decode kernel forms the decayed state with packed
                # FMA and a zero addend. Keep that rounding instead of letting
                # the compiler pick a plain multiply.
                b_h = tl.fma(
                    b_h,
                    b_gate_multiplier[:, None],
                    tl.zeros((BK, BV), tl.float32),
                )
            else:
                b_h *= tl.exp(b_g[:, None])
        else:
            b_h *= tl.exp(b_g)

        # Delta rule: v -= sum(h * k, dim=0)
        b_v -= tl.sum(b_h * b_k[:, None], 0)

        # Apply beta gating: v *= beta
        b_v *= b_beta

        # Update hidden state: h += k[:, None] * v[None, :]
        if MATCH_CUTEDSL_DECODE:
            b_h = tl.fma(b_k[:, None], b_v[None, :], b_h)
        elif IS_KDA and V == 128 and BV == 4:
            # The narrow KDA tile keeps the separate FP32 multiply/add rounding.
            b_h = tl.inline_asm_elementwise(
                "add.rn.f32 $0, $1, $2;",
                constraints="=f,f,f",
                args=[b_h, b_k[:, None] * b_v[None, :]],
                dtype=tl.float32,
                is_pure=True,
                pack=1,
            )
        else:
            b_h += b_k[:, None] * b_v[None, :]

        # Compute output: o = sum(h * q, dim=0)
        b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        # Cache intermediate states if enabled
        if CACHE_INTERMEDIATE_STATES:
            if cache_idx >= 0:
                step_offset = step_idx * HV * K * V
                cache_ptr = (
                    intermediate_states_buffer
                    + cache_idx * cache_steps * HV * K * V
                    + step_offset
                    + i_hv * K * V
                    + o_v[None, :] * K
                    + o_k[:, None]
                )
                tl.store(cache_ptr, b_h.to(cache_ptr.dtype.element_ty), mask=mask_h)

        # A BF16 recurrent-state decode persists its state after every token.
        # Multi-token verification must make that precision boundary visible to
        # the next recurrence step instead of retaining the FP32 accumulator for
        # the whole candidate chain. FP32 state pools must retain the accumulator.
        if ROUND_STATE_TO_BF16:
            b_h = b_h.to(tl.bfloat16).to(tl.float32)

        step_idx += 1

        # Update pointers for next timestep
        p_q += stride_q
        p_k += stride_k
        p_v += stride_v
        p_b += stride_b
        p_o += HV * V
        p_a += stride_a

    # Store final state back to h0_source with bounds checking
    if not DISABLE_STATE_UPDATE:
        if USE_INITIAL_STATE:
            idx = tl.load(h0_indices + i_n).to(tl.int64)
            if idx >= 0:
                p_h0 = (
                    h0_source
                    + idx * stride_h0_source
                    + i_hv * K * V
                    + o_v[None, :] * K
                    + o_k[:, None]
                )
                tl.store(p_h0, b_h.to(p_h0.dtype.element_ty), mask=mask_h)


@triton.jit
def fused_sigmoid_gating_delta_rule_tree_factor_kernel(
    A_log,
    a,
    dt_bias,
    softplus_beta,
    softplus_threshold,
    lower_bound,
    q,
    k,
    v,
    b,
    cu_seqlens,
    retrieve_parent_token_ptr,
    tree_depths_ptr,
    processed_k,
    processed_decay,
    processed_beta,
    tree_query_factors,
    tree_value_factors,
    factor_cache_indices,
    tree_rawv,
    tree_rawk,
    tree_g,
    tree_beta,
    stride_rawv_slot: tl.constexpr,
    stride_rawk_slot: tl.constexpr,
    stride_g_slot: tl.constexpr,
    stride_beta_slot: tl.constexpr,
    stride_retrieve_parent_token_seq: tl.constexpr,
    stride_retrieve_parent_token_token: tl.constexpr,
    stride_tree_depth_seq: tl.constexpr,
    stride_tree_depth_token: tl.constexpr,
    scale,
    stride_a,
    stride_q,
    stride_k,
    stride_v,
    stride_b,
    TREE_WIDTH: tl.constexpr,
    CACHE_TREE_WIDTH: tl.constexpr,
    MAX_DEPTH: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BT: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
    CACHE_TREE_FACTORS: tl.constexpr,
    USE_DYNAMIC_DEPTH: tl.constexpr,
    USE_PRECOMPUTED_TRANSFORMS: tl.constexpr,
    USE_GDC: tl.constexpr = False,
):
    """Build value-independent factors for all root-to-node KDA paths.

    For one update, S' = A S + u v^T where
      A = (I - beta k k^T) diag(exp(g)), u = beta k.
    Walking a node's path backwards transforms its output query with A^T and
    emits one scalar coefficient for each ancestor value.  No [K, V] state is
    materialized for proposal nodes.
    """
    i_nt, i_hv = tl.program_id(0), tl.program_id(1)
    if USE_GDC:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()

    i_n = i_nt // TREE_WIDTH
    node_idx = i_nt % TREE_WIDTH
    i_h = i_hv // (HV // H)
    bos = tl.load(cu_seqlens + i_n).to(tl.int64)
    eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
    node_active = node_idx < eos - bos

    o_k = tl.arange(0, BK)
    o_t = tl.arange(0, BT)
    mask_k = o_k < K
    mask_t = o_t < TREE_WIDTH
    factor_row = ((i_n * HV + i_hv) * TREE_WIDTH + node_idx) * TREE_WIDTH
    b_value_factors = tl.zeros([BT], dtype=tl.float32)

    token_idx = bos + node_idx
    b_q = tl.load(
        q + token_idx * stride_q + i_h * K + o_k,
        mask=node_active & mask_k,
        other=0.0,
    ).to(tl.float32)
    if USE_QK_L2NORM_IN_KERNEL:
        b_q /= tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
    b_x = b_q * scale

    b_A_log = tl.load(A_log + i_hv).to(tl.float32)
    b_dt_bias = tl.load(dt_bias + i_hv * K + o_k, mask=mask_k, other=0.0).to(tl.float32)
    current_idx = node_idx.to(tl.int64)
    path_active = node_active
    if USE_DYNAMIC_DEPTH:
        # tree_depth is zero-based, while the factor walk includes the node itself.
        # A runtime loop avoids executing masked exp/log/sqrt work for the
        # MAX_DEPTH - actual_path_length tail of shallow nodes.
        path_length = (
            tl.load(
                tree_depths_ptr
                + i_n * stride_tree_depth_seq
                + node_idx * stride_tree_depth_token,
                mask=node_active,
                other=-1,
            ).to(tl.int32)
            + 1
        )
    else:
        path_length = MAX_DEPTH
    path_step = 0
    while (path_step < path_length) & path_active:
        path_active = path_active & (current_idx >= 0) & (current_idx < TREE_WIDTH)
        safe_idx = tl.maximum(current_idx, 0)
        current_token_idx = bos + safe_idx
        if USE_PRECOMPUTED_TRANSFORMS:
            transform_base = (i_n * HV * TREE_WIDTH + i_hv * TREE_WIDTH + safe_idx) * K
            b_k = tl.load(
                processed_k + transform_base + o_k,
                mask=path_active & mask_k,
                other=0.0,
            ).to(tl.float32)
            b_decay = tl.load(
                processed_decay + transform_base + o_k,
                mask=path_active & mask_k,
                other=0.0,
            ).to(tl.float32)
            b_beta = tl.load(
                processed_beta + (i_n * HV + i_hv) * TREE_WIDTH + safe_idx,
                mask=path_active,
                other=0.0,
            ).to(tl.float32)
        else:
            b_k = tl.load(
                k + current_token_idx * stride_k + i_h * K + o_k,
                mask=path_active & mask_k,
                other=0.0,
            ).to(tl.float32)
            b_rawk = b_k
            if USE_QK_L2NORM_IN_KERNEL:
                b_k /= tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
            b_a = tl.load(
                a + current_token_idx * stride_a + i_hv * K + o_k,
                mask=path_active & mask_k,
                other=0.0,
            ).to(tl.float32)
            b_b = tl.load(
                b + current_token_idx * stride_b + i_hv,
                mask=path_active,
                other=0.0,
            ).to(tl.float32)
            x = b_a + b_dt_bias
            if USE_LOWER_BOUND:
                b_g = lower_bound * tl.sigmoid(tl.exp(b_A_log) * x)
            else:
                beta_x = softplus_beta * x
                softplus_x = tl.where(
                    beta_x <= softplus_threshold,
                    (1.0 / softplus_beta) * tl.log(1.0 + tl.exp(beta_x)),
                    x,
                )
                b_g = -tl.exp(b_A_log) * softplus_x
            b_decay = tl.exp(b_g)
            b_beta = 1.0 / (1.0 + tl.exp(-b_b))
        if CACHE_TREE_FACTORS and not USE_PRECOMPUTED_TRANSFORMS:
            if path_step == 0:
                factor_slot = tl.load(factor_cache_indices + i_n).to(tl.int64)
                if factor_slot >= 0:
                    b_rawv = tl.load(
                        v + current_token_idx * stride_v + i_hv * V + o_k,
                        mask=mask_k,
                        other=0.0,
                    )
                    tl.store(
                        tree_rawv
                        + factor_slot * stride_rawv_slot
                        + (i_hv * CACHE_TREE_WIDTH + node_idx) * V
                        + o_k,
                        b_rawv,
                        mask=mask_k,
                    )
                    tl.store(
                        tree_g
                        + factor_slot * stride_g_slot
                        + (i_hv * CACHE_TREE_WIDTH + node_idx) * K
                        + o_k,
                        b_g,
                        mask=mask_k,
                    )
                    if i_hv % (HV // H) == 0:
                        tl.store(
                            tree_rawk
                            + factor_slot * stride_rawk_slot
                            + (i_h * CACHE_TREE_WIDTH + node_idx) * K
                            + o_k,
                            b_rawk,
                            mask=mask_k,
                        )
                    tl.store(
                        tree_beta
                        + factor_slot * stride_beta_slot
                        + i_hv * CACHE_TREE_WIDTH
                        + node_idx,
                        b_beta,
                    )
        b_coefficient = b_beta * tl.sum(b_x * b_k)
        b_value_factors += tl.where(path_active & (o_t == safe_idx), b_coefficient, 0.0)
        # A^T x = diag(exp(g)) (x - k * (beta k^T x)).
        b_next_x = b_decay * (b_x - b_k * b_coefficient)
        b_x = tl.where(path_active, b_next_x, b_x)
        current_idx = tl.load(
            retrieve_parent_token_ptr
            + i_n * stride_retrieve_parent_token_seq
            + safe_idx * stride_retrieve_parent_token_token,
            mask=path_active,
            other=-1,
        ).to(tl.int64)
        path_step += 1

    tl.store(tree_value_factors + factor_row + o_t, b_value_factors, mask=mask_t)
    query_factor_ptr = (
        tree_query_factors + ((i_n * HV + i_hv) * TREE_WIDTH + node_idx) * K + o_k
    )
    tl.store(query_factor_ptr, b_x, mask=node_active & mask_k)


@triton.jit
def fused_sigmoid_gating_delta_rule_tree_factor_verify_kernel(
    v,
    o,
    h0_source,
    h0_indices,
    stride_h0_source,
    cu_seqlens,
    tree_query_factors,
    tree_value_factors,
    stride_v,
    TREE_WIDTH: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BT: tl.constexpr,
    BM: tl.constexpr,
    NM: tl.constexpr,
    FACTORS_LOW_PRECISION: tl.constexpr,
    USE_GDC: tl.constexpr = False,
):
    """Recover every tree-node output from compact query/value factors."""
    i_v, i_nm, i_hv = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n = i_nm // NM
    i_m = i_nm % NM
    if USE_GDC:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()

    bos = tl.load(cu_seqlens + i_n).to(tl.int64)
    eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
    tree_size = eos - bos
    state_idx = tl.load(h0_indices + i_n).to(tl.int64)
    o_m = i_m * BM + tl.arange(0, BM)
    o_t = tl.arange(0, BT)
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_m = o_m < tree_size
    mask_t = o_t < tree_size
    mask_k = o_k < K
    mask_v = o_v < V

    query_factor_ptr = (
        tree_query_factors
        + ((i_n * HV + i_hv) * TREE_WIDTH + o_m[:, None]) * K
        + o_k[None, :]
    )
    b_query_factors = tl.load(
        query_factor_ptr,
        mask=mask_m[:, None] & mask_k[None, :],
        other=0.0,
    )
    if not FACTORS_LOW_PRECISION:
        b_query_factors = b_query_factors.to(tl.float32)
    state_ptr = (
        h0_source
        + state_idx * stride_h0_source
        + i_hv * K * V
        + o_v[None, :] * K
        + o_k[:, None]
    )
    b_state = tl.load(
        state_ptr,
        mask=(state_idx >= 0) & mask_k[:, None] & mask_v[None, :],
        other=0.0,
    ).to(tl.float32)
    if FACTORS_LOW_PRECISION:
        b_state = b_state.to(tl.bfloat16)
    b_o = tl.dot(b_query_factors, b_state, input_precision="ieee")

    value_factor_ptr = (
        tree_value_factors
        + ((i_n * HV + i_hv) * TREE_WIDTH + o_m[:, None]) * TREE_WIDTH
        + o_t[None, :]
    )
    b_value_factors = tl.load(
        value_factor_ptr,
        mask=mask_m[:, None] & mask_t[None, :],
        other=0.0,
    )
    if not FACTORS_LOW_PRECISION:
        b_value_factors = b_value_factors.to(tl.float32)
    value_ptr = v + (bos + o_t[:, None]) * stride_v + i_hv * V + o_v[None, :]
    b_values = tl.load(
        value_ptr,
        mask=mask_t[:, None] & mask_v[None, :],
        other=0.0,
    ).to(tl.float32)
    if FACTORS_LOW_PRECISION:
        b_values = b_values.to(tl.bfloat16)
    b_o += tl.dot(b_value_factors, b_values, input_precision="ieee")

    output_ptr = o + ((bos + o_m[:, None]) * HV + i_hv) * V + o_v[None, :]
    tl.store(
        output_ptr,
        b_o.to(output_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_v[None, :],
    )


def fused_sigmoid_gating_delta_rule_update(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    softplus_beta: float,
    softplus_threshold: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    scale: Optional[float] = None,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    is_kda: bool = False,
    lower_bound: Optional[float] = None,
    # Optional parameters for target_verify support
    disable_state_update: bool = False,
    intermediate_states_buffer: Optional[torch.Tensor] = None,
    intermediate_state_indices: Optional[torch.Tensor] = None,
    cache_steps: Optional[int] = None,
    retrieve_parent_token: Optional[torch.Tensor] = None,
    # fused ReplaySSM ring-write (spec verify). When cache_ring, each draft step
    # stores pre-norm k / raw v / gate / beta into these per-slot rings,
    # replacing the eager ring-write. Off by default -> decode unchanged.
    cache_ring: bool = False,
    replayssm_rawv: Optional[torch.Tensor] = None,
    replayssm_rawk: Optional[torch.Tensor] = None,
    replayssm_g: Optional[torch.Tensor] = None,
    replayssm_beta: Optional[torch.Tensor] = None,
    # Reproduce FlashInfer's `recurrent_kda` CuTe DSL decode kernel step for
    # step: `b` is then the already-sigmoided FP32 beta, the gate is formed in
    # base 2 with ex2/lg2.approx, Q/K norms reduce lane-striped, the state
    # decays and updates through FMA, and a BF16 state pool is rounded after
    # every token.
    match_cutedsl_decode: bool = False,
    *,
    tree_depths: Optional[torch.Tensor] = None,
    tree_max_depth: int = 0,
    tree_query_factors: Optional[torch.Tensor] = None,
    tree_value_factors: Optional[torch.Tensor] = None,
    force_factorized_tree: bool = False,
    factor_cache_indices: Optional[torch.Tensor] = None,
    factor_num_warps: Optional[int] = None,
    factor_num_stages: int = 2,
    factor_verify_bv: Optional[int] = None,
    factor_verify_bm: Optional[int] = None,
    factor_verify_num_warps: Optional[int] = None,
    factor_verify_num_stages: Optional[int] = None,
    tree_processed_k: Optional[torch.Tensor] = None,
    tree_processed_decay: Optional[torch.Tensor] = None,
    tree_processed_beta: Optional[torch.Tensor] = None,
    force_precomputed_tree_transforms: bool = False,
):
    """
    Fused triton implementation of sigmoid gating delta rule update.
    This function uses a single fused kernel that combines both sigmoid gating computation
    and the recurrent delta rule update for better performance.

    Supports both decode and target_verify modes:
    - decode: standard single-step update with state write-back
    - target_verify: multi-step with intermediate state caching, optional tree attention,
                     and optional state update disable
    """
    if match_cutedsl_decode and (not is_kda or lower_bound is not None):
        raise ValueError(
            "match_cutedsl_decode is defined for the KDA softplus gate only "
            "(is_kda=True, lower_bound=None)."
        )
    if match_cutedsl_decode and _is_hip:
        # The parity branch emits PTX (ex2/lg2.approx) to reproduce FlashInfer's
        # CuTe DSL decode kernel. AMDGCN rejects it at compile time but the launch
        # still runs, so without this check ROCm silently returns wrong values.
        raise NotImplementedError(
            "match_cutedsl_decode reproduces an NVIDIA-only kernel and is not "
            "available on ROCm."
        )
    B, T, H, K, V = *k.shape, v.shape[-1]
    stride_q = q.stride()[1]
    stride_k = k.stride()[1]
    stride_v = v.stride()[1]
    stride_b = b.stride()[-2]
    # Both paths (KDA/GDN) advance p_a once per token, so use the token-axis stride.
    # For 2D a ([T, ...]) this is stride(0); for 3D a ([B, T, ...]) this is stride(1).
    # Using stride()[-2] covers GDN [T, HV] and KDA layouts ([T, HV*K] / [B, T, HV*K]).
    # KDA decode also passes 4-D [B, T, H, K], where [-2] is the head stride, not the
    # token stride; take dim 1 explicitly for that layout.
    stride_a = a.stride()[1] if a.ndim == 4 else a.stride()[-2]
    HV = v.shape[2]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    use_pdl = is_arch_support_pdl()
    target_verify = (
        disable_state_update and not match_cutedsl_decode
        if is_kda
        else intermediate_states_buffer is not None
    )
    BV, num_warps = _select_recurrent_launch_config(
        N,
        H,
        HV,
        K,
        V,
        is_kda,
        target_verify=target_verify,
        cache_steps=cache_steps,
    )
    BK = triton.next_power_of_2(K)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, "NK > 1 is not supported yet"
    num_stages = 3

    if scale is None:
        scale = k.shape[-1] ** -0.5
    else:
        assert scale > 0, "scale must be positive"

    o = q.new_empty(NK, *v.shape)

    # Prepare retrieve_parent_token strides
    if retrieve_parent_token is not None:
        stride_retrieve_parent_token_seq = retrieve_parent_token.stride(0)
        stride_retrieve_parent_token_token = retrieve_parent_token.stride(1)
    else:
        stride_retrieve_parent_token_seq = 0
        stride_retrieve_parent_token_token = 0

    NP2_T = triton.next_power_of_2(T)

    split_n_hv_grid = q.device.type == "cuda"
    grid = (NV, N, HV) if split_n_hv_grid else (NK, NV, N * HV)

    # Adaptive spec changes the runtime draft count without changing the
    # allocated per-request pitch, which is preserved in stride(0).
    if intermediate_states_buffer is not None:
        cache_stride_steps = intermediate_states_buffer.stride(0) // (HV * K * V)
    elif cache_steps is not None and cache_steps > 0:
        cache_stride_steps = cache_steps
    else:
        cache_stride_steps = 0

    # ring strides (per-slot rings are contiguous [num_slots, heads, L, dim];
    # the kernel offsets within a slot with MAX_CACHE_LEN and the dim extents).
    if cache_ring:
        # stride(0) is used as the slot pitch, so a tensor still carrying the
        # layer dim would scribble outside its slot. The gate ring is the one
        # whose rank depends on the model: per-K vector for KDA, per-head scalar
        # for GDN, matching g_shape in memory_pool.py and the IS_KDA branch in
        # the store above.
        assert (
            replayssm_rawv.dim() == 4
            and replayssm_rawk.dim() == 4
            and replayssm_g.dim() == (4 if is_kda else 3)
            and replayssm_beta.dim() == 3
        ), "cache_ring expects per-layer ring views"
        max_cache_len = replayssm_rawv.shape[-2]
        stride_rawv_slot = replayssm_rawv.stride(0)
        stride_rawk_slot = replayssm_rawk.stride(0)
        stride_g_slot = replayssm_g.stride(0)
        stride_beta_slot = replayssm_beta.stride(0)
    else:
        max_cache_len = 0
        stride_rawv_slot = stride_rawk_slot = stride_g_slot = stride_beta_slot = 0

    # PDL (sm90+): chain this kernel behind its producer conv1d_update, which
    # already launches dependents. Bit-exact (scheduling only) — benefits both
    # KDA and GDN recurrent paths.
    pdl_kwargs = {"USE_GDC": True, "launch_pdl": True} if use_pdl else {}

    # CuTeDSL specializes on the persisted state dtype. Its BF16 decode path
    # writes BF16 after every token and reloads FP32 for the next recurrence;
    # its FP32 path has no such rounding boundary. Keep CuTe math parity and
    # state-persistence precision as separate compile-time choices.
    round_state_to_bf16 = match_cutedsl_decode and (
        initial_state_source is None or initial_state_source.dtype == torch.bfloat16
    )

    use_factorized_tree = (
        force_factorized_tree
        and is_kda
        and disable_state_update
        and cu_seqlens is not None
        and retrieve_parent_token is not None
        and tree_max_depth > 0
        and tree_query_factors is not None
        and tree_value_factors is not None
        and initial_state_source is not None
        and initial_state_indices is not None
    )
    if use_factorized_tree:
        # Factorized verify for the fixed-width UBTree path. It computes every
        # node output without materializing per-node [V, K] states; the
        # post-sampling tree commit folds only the accept path.
        tree_width = retrieve_parent_token.shape[1]
        assert N == 1, f"factorized tree verify supports only BS=1, got {N}"
        assert tree_width in (32, 49), (
            f"factorized tree verify supports only widths 32 and 49, got {tree_width}"
        )
        assert retrieve_parent_token.shape == (N, tree_width)
        assert tree_query_factors.shape == (N, HV, tree_width, K)
        assert tree_value_factors.shape == (N, HV, tree_width, tree_width)
        assert tree_query_factors.dtype == tree_value_factors.dtype
        assert tree_query_factors.dtype in (torch.float32, torch.bfloat16)
        assert tree_query_factors.is_contiguous()
        assert tree_value_factors.is_contiguous()
        assert tree_max_depth <= 16
        use_dynamic_depth = tree_depths is not None
        if use_dynamic_depth:
            assert tree_depths.shape == (N, tree_width)
        use_precomputed_transforms = force_precomputed_tree_transforms
        # Precompute outputs are consumed immediately by factor generation, so
        # keep normal stream ordering for this pair instead of early PDL overlap.
        factor_pdl_kwargs = (
            {"USE_GDC": True, "launch_pdl": True}
            if is_arch_support_pdl() and not use_precomputed_transforms
            else {}
        )
        if use_precomputed_transforms:
            assert tree_processed_k.shape == (N, HV, tree_width, K)
            assert tree_processed_decay.shape == (N, HV, tree_width, K)
            assert tree_processed_beta.shape == (N, HV, tree_width)
        if factor_num_warps is None:
            factor_num_warps = 2
        if factor_verify_bv is None:
            # M=32 on Ling3-Flash TP4 is occupancy-limited at BV=16.  Splitting
            # V into BV=8 tiles lowers the verify kernel register footprint and
            # wins despite doubling the V-grid.  Other widths keep their
            # previously validated tile.
            factor_verify_bv = 8 if tree_width == 32 else 32
        assert factor_verify_bv in (8, 16, 32)
        cache_tree_factors = cache_ring
        if cache_tree_factors:
            assert V == K, "factorized tree raw cache currently requires V == K"
            assert factor_cache_indices is not None
            assert max_cache_len >= tree_width
        factor_grid = (N * tree_width, HV)
        fused_sigmoid_gating_delta_rule_tree_factor_kernel[factor_grid](
            A_log=A_log,
            a=a,
            dt_bias=dt_bias,
            softplus_beta=softplus_beta,
            softplus_threshold=softplus_threshold,
            lower_bound=lower_bound if lower_bound is not None else 0.0,
            q=q,
            k=k,
            v=v,
            b=b,
            cu_seqlens=cu_seqlens,
            retrieve_parent_token_ptr=retrieve_parent_token,
            tree_depths_ptr=tree_depths,
            processed_k=tree_processed_k,
            processed_decay=tree_processed_decay,
            processed_beta=tree_processed_beta,
            tree_query_factors=tree_query_factors,
            tree_value_factors=tree_value_factors,
            factor_cache_indices=factor_cache_indices,
            tree_rawv=replayssm_rawv,
            tree_rawk=replayssm_rawk,
            tree_g=replayssm_g,
            tree_beta=replayssm_beta,
            stride_rawv_slot=stride_rawv_slot,
            stride_rawk_slot=stride_rawk_slot,
            stride_g_slot=stride_g_slot,
            stride_beta_slot=stride_beta_slot,
            stride_retrieve_parent_token_seq=retrieve_parent_token.stride(0),
            stride_retrieve_parent_token_token=retrieve_parent_token.stride(1),
            stride_tree_depth_seq=tree_depths.stride(0) if use_dynamic_depth else 0,
            stride_tree_depth_token=tree_depths.stride(1) if use_dynamic_depth else 0,
            scale=scale,
            stride_a=stride_a,
            stride_q=stride_q,
            stride_k=stride_k,
            stride_v=stride_v,
            stride_b=stride_b,
            TREE_WIDTH=tree_width,
            CACHE_TREE_WIDTH=max_cache_len,
            MAX_DEPTH=tree_max_depth,
            H=H,
            HV=HV,
            K=K,
            V=V,
            BK=BK,
            BT=triton.next_power_of_2(tree_width),
            USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
            USE_LOWER_BOUND=lower_bound is not None,
            CACHE_TREE_FACTORS=cache_tree_factors,
            USE_DYNAMIC_DEPTH=use_dynamic_depth,
            USE_PRECOMPUTED_TRANSFORMS=use_precomputed_transforms,
            num_warps=factor_num_warps,
            num_stages=factor_num_stages,
            **factor_pdl_kwargs,
        )
        factor_verify_nv = triton.cdiv(V, factor_verify_bv)
        # Use the validated fixed-width schedules: one 32-row CTA for width 32
        # and one 64-row CTA for width 49.
        if factor_verify_bm is None:
            factor_verify_bm = 32 if tree_width == 32 else 64
        assert factor_verify_bm in (16, 32, 64)
        if factor_verify_num_warps is None:
            factor_verify_num_warps = 2 if tree_width == 32 else 8
        if factor_verify_num_stages is None:
            factor_verify_num_stages = 1 if tree_width == 32 else 3
        factor_verify_nm = triton.cdiv(tree_width, factor_verify_bm)
        factor_verify_grid = (factor_verify_nv, N * factor_verify_nm, HV)
        fused_sigmoid_gating_delta_rule_tree_factor_verify_kernel[factor_verify_grid](
            v=v,
            o=o,
            h0_source=initial_state_source,
            h0_indices=initial_state_indices,
            stride_h0_source=initial_state_source.stride(0),
            cu_seqlens=cu_seqlens,
            tree_query_factors=tree_query_factors,
            tree_value_factors=tree_value_factors,
            stride_v=stride_v,
            TREE_WIDTH=tree_width,
            H=H,
            HV=HV,
            K=K,
            V=V,
            BK=BK,
            BV=factor_verify_bv,
            BT=triton.next_power_of_2(tree_width),
            BM=factor_verify_bm,
            NM=factor_verify_nm,
            FACTORS_LOW_PRECISION=tree_query_factors.dtype != torch.float32,
            num_warps=factor_verify_num_warps,
            num_stages=factor_verify_num_stages,
            **factor_pdl_kwargs,
        )
        return o.squeeze(0)

    fused_sigmoid_gating_delta_rule_update_kernel[grid](
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
        lower_bound=lower_bound if lower_bound is not None else 0.0,
        q=q,
        k=k,
        v=v,
        b=b,
        o=o,
        h0_source=initial_state_source,
        h0_indices=initial_state_indices,
        # Envelope-strided state pools (page-major / unified memory) have a
        # per-slot pitch != HV*K*V; contiguous pools pass exactly HV*K*V.
        stride_h0_source=(
            initial_state_source.stride(0) if initial_state_source is not None else 0
        ),
        cu_seqlens=cu_seqlens,
        intermediate_states_buffer=intermediate_states_buffer,
        intermediate_state_indices=intermediate_state_indices,
        cache_steps=cache_stride_steps,
        retrieve_parent_token_ptr=retrieve_parent_token,
        stride_retrieve_parent_token_seq=stride_retrieve_parent_token_seq,
        stride_retrieve_parent_token_token=stride_retrieve_parent_token_token,
        scale=scale,
        T=T,
        stride_a=stride_a,
        stride_q=stride_q,
        stride_k=stride_k,
        stride_v=stride_v,
        stride_b=stride_b,
        NP2_T=NP2_T,
        B=B,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        USE_INITIAL_STATE=initial_state_source is not None,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        IS_VARLEN=cu_seqlens is not None,
        IS_KDA=is_kda,
        USE_LOWER_BOUND=lower_bound is not None,
        DISABLE_STATE_UPDATE=disable_state_update,
        CACHE_INTERMEDIATE_STATES=intermediate_states_buffer is not None,
        HAS_EAGLE_TREE_CUSTOM_ATTN_MASK=retrieve_parent_token is not None,
        replayssm_rawv=replayssm_rawv,
        replayssm_rawk=replayssm_rawk,
        replayssm_g=replayssm_g,
        replayssm_beta=replayssm_beta,
        stride_rawv_slot=stride_rawv_slot,
        stride_rawk_slot=stride_rawk_slot,
        stride_g_slot=stride_g_slot,
        stride_beta_slot=stride_beta_slot,
        MAX_CACHE_LEN=max_cache_len,
        CACHE_RING=cache_ring,
        SPLIT_N_HV_GRID=split_n_hv_grid,
        MATCH_CUTEDSL_DECODE=match_cutedsl_decode,
        ROUND_STATE_TO_BF16=round_state_to_bf16,
        num_warps=num_warps,
        num_stages=num_stages,
        **pdl_kwargs,
    )
    o = o.squeeze(0)
    return o
