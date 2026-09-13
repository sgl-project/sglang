from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.kernels.jit.utils import is_arch_support_pdl


@triton.jit(do_not_specialize=["T"])
def _gate_and_ring_write_kernel(
    A_log,
    a,
    dt_bias,
    b,
    g_buffer,
    beta_buffer,
    softplus_beta,
    softplus_threshold,
    lower_bound,
    cu_seqlens,
    stride_a,
    stride_b,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_KDA: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
):
    i_nh = tl.program_id(0)
    i_n, i_hv = i_nh // HV, i_nh % HV

    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        T_seq = eos - bos
    else:
        bos = i_n * T
        T_seq = T

    o_k = tl.arange(0, BK)
    mask_k = o_k < K

    p_b = b + bos * stride_b + i_hv
    p_A_log = A_log + i_hv
    if IS_KDA:
        p_a = a + bos * stride_a + i_hv * K + o_k
        p_dt_bias = dt_bias + i_hv * K + o_k
    else:
        p_a = a + bos * stride_a + i_hv
        p_dt_bias = dt_bias + i_hv

    for step_idx in range(0, T_seq):
        b_A_log = tl.load(p_A_log).to(tl.float32)
        if IS_KDA:
            b_a = tl.load(p_a, mask=mask_k, other=0).to(tl.float32)
            b_dt_bias = tl.load(p_dt_bias, mask=mask_k, other=0).to(tl.float32)
        else:
            b_a = tl.load(p_a).to(tl.float32)
            b_dt_bias = tl.load(p_dt_bias).to(tl.float32)

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

        b_b = tl.load(p_b).to(tl.float32)
        b_beta = 1.0 / (1.0 + tl.exp(-b_b))

        if IS_KDA:
            tl.store(
                g_buffer + (bos + step_idx) * HV * K + i_hv * K + o_k,
                b_g,
                mask=mask_k,
            )
        else:
            tl.store(g_buffer + (bos + step_idx) * HV + i_hv, b_g)
        tl.store(beta_buffer + (bos + step_idx) * HV + i_hv, b_beta)

        p_b += stride_b
        p_a += stride_a


@triton.jit(do_not_specialize=["T"])
def _recurrent_delta_rule_preactivated_kernel(
    q,
    k,
    v,
    o,
    g_preact,
    beta_preact,
    h0_source,
    h0_indices,
    stride_h0_source,
    cu_seqlens,
    scale,
    T,
    stride_q,
    stride_k,
    stride_v,
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
):
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        all_tokens = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all_tokens = B * T

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    p_q = q + bos * stride_q + i_h * K + o_k
    p_k = k + bos * stride_k + i_h * K + o_k
    p_v = v + bos * stride_v + i_hv * V + o_v
    p_o = o + ((i_k * all_tokens + bos) * HV + i_hv) * V + o_v

    if IS_KDA:
        p_g = g_preact + bos * HV * K + i_hv * K + o_k
    else:
        p_g = g_preact + bos * HV + i_hv
    p_beta = beta_preact + bos * HV + i_hv

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]

    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    if USE_INITIAL_STATE:
        idx = tl.load(h0_indices + i_n).to(tl.int64)
        if idx >= 0:
            p_h0 = (
                h0_source
                + idx * stride_h0_source
                + i_hv * K * V
                + o_v[:, None] * K
                + o_k[None, :]
            )
            b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for _ in range(0, T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)

        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / (tl.sqrt(tl.sum(b_q * b_q) + 1e-6))
            b_k = b_k / (tl.sqrt(tl.sum(b_k * b_k) + 1e-6))
        b_q = b_q * scale

        if IS_KDA:
            b_g = tl.load(p_g, mask=mask_k, other=0).to(tl.float32)
            b_h *= tl.exp(b_g[None, :])
        else:
            b_g = tl.load(p_g).to(tl.float32)
            b_h *= tl.exp(b_g)

        b_v -= tl.sum(b_h * b_k[None, :], 1)
        b_beta = tl.load(p_beta).to(tl.float32)
        b_v *= b_beta
        b_h += b_v[:, None] * b_k[None, :]
        b_o = tl.sum(b_h * b_q[None, :], 1)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        p_q += stride_q
        p_k += stride_k
        p_v += stride_v
        p_o += HV * V
        if IS_KDA:
            p_g += HV * K
        else:
            p_g += HV
        p_beta += HV


@triton.jit
def _replayssm_verify_recurrent_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    g_preact_ptr,
    beta_preact_ptr,
    initial_state_ptr,
    initial_indices_ptr,
    replayssm_rawv,
    replayssm_rawk,
    replayssm_g,
    replayssm_beta,
    stride_initial_0,
    scale,
    stride_q,
    stride_k,
    stride_v,
    stride_rawv_slot: tl.constexpr,
    stride_rawk_slot: tl.constexpr,
    stride_g_slot: tl.constexpr,
    stride_beta_slot: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    STEPS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    MAX_CACHE_LEN: tl.constexpr,
    CACHE_RING: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_KDA: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_hv = tl.program_id(1)
    pid_v = tl.program_id(2)

    offset_k = tl.arange(0, BK)
    offset_v = pid_v * BV + tl.arange(0, BV)
    mask_k = offset_k < K
    mask_v = offset_v < V
    mask_state = mask_v[:, None] & mask_k[None, :]

    k_head = pid_hv // (HV // H)

    state = tl.zeros([BV, BK], dtype=tl.float32)
    if USE_INITIAL_STATE:
        initial_idx = tl.load(initial_indices_ptr + pid_batch).to(tl.int64)
        if initial_idx >= 0:
            p_h0 = (
                initial_state_ptr
                + initial_idx * stride_initial_0
                + pid_hv * V * K
                + offset_v[:, None] * K
                + offset_k[None, :]
            )
            state = tl.load(p_h0, mask=mask_state, other=0.0).to(tl.float32)

    for step in tl.static_range(0, STEPS):
        token = pid_batch * STEPS + step

        k_raw = tl.load(
            k_ptr + token * stride_k + k_head * K + offset_k,
            mask=mask_k,
            other=0.0,
        )
        v_raw = tl.load(
            v_ptr + token * stride_v + pid_hv * V + offset_v,
            mask=mask_v,
            other=0.0,
        )

        if CACHE_RING:
            if step < MAX_CACHE_LEN:
                tl.store(
                    replayssm_rawv
                    + pid_batch * stride_rawv_slot
                    + pid_hv * MAX_CACHE_LEN * V
                    + step * V
                    + offset_v,
                    v_raw.to(replayssm_rawv.dtype.element_ty),
                    mask=mask_v,
                )
                if pid_v == 0:
                    tl.store(
                        replayssm_rawk
                        + pid_batch * stride_rawk_slot
                        + k_head * MAX_CACHE_LEN * K
                        + step * K
                        + offset_k,
                        k_raw.to(replayssm_rawk.dtype.element_ty),
                        mask=mask_k,
                    )
                    if IS_KDA:
                        tl.store(
                            replayssm_g
                            + pid_batch * stride_g_slot
                            + pid_hv * MAX_CACHE_LEN * K
                            + step * K
                            + offset_k,
                            tl.load(
                                g_preact_ptr
                                + (token * HV + pid_hv) * K
                                + offset_k,
                                mask=mask_k,
                                other=0.0,
                            ),
                            mask=mask_k,
                        )
                    else:
                        tl.store(
                            replayssm_g
                            + pid_batch * stride_g_slot
                            + pid_hv * MAX_CACHE_LEN
                            + step,
                            tl.load(
                                g_preact_ptr + token * HV + pid_hv
                            ),
                        )
                    tl.store(
                        replayssm_beta
                        + pid_batch * stride_beta_slot
                        + pid_hv * MAX_CACHE_LEN
                        + step,
                        tl.load(
                            beta_preact_ptr + token * HV + pid_hv
                        ),
                    )

        q = tl.load(
            q_ptr + token * stride_q + k_head * K + offset_k,
            mask=mask_k,
            other=0.0,
        ).to(tl.float32)
        k = k_raw.to(tl.float32)
        value = v_raw.to(tl.float32)

        if USE_QK_L2NORM_IN_KERNEL:
            q = q / (tl.sqrt(tl.sum(q * q, axis=0)) + 1e-6)
            k = k / (tl.sqrt(tl.sum(k * k, axis=0)) + 1e-6)
        q *= scale

        if IS_KDA:
            gate = tl.load(
                g_preact_ptr + (token * HV + pid_hv) * K + offset_k,
                mask=mask_k,
                other=0.0,
            ).to(tl.float32)
            state *= tl.exp(gate[None, :])
        else:
            gate = tl.load(
                g_preact_ptr + token * HV + pid_hv
            ).to(tl.float32)
            state *= tl.exp(gate)

        value -= tl.sum(state * k[None, :], axis=1)
        beta = tl.load(beta_preact_ptr + token * HV + pid_hv).to(tl.float32)
        value *= beta
        state += value[:, None] * k[None, :]
        output = tl.sum(state * q[None, :], axis=1)

        tl.store(
            out_ptr + (token * HV + pid_hv) * V + offset_v,
            output,
            mask=mask_v,
        )


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
    USE_GDC: tl.constexpr = False,
):
    """
    Fused kernel that combines sigmoid gating computation with recurrent delta rule update.
    """
    # PDL: overlap this kernel's prologue with the producer (the KDA/GDN
    # conv1d_update). All global loads below happen after the wait, so
    # numerics are unchanged. The immediate trigger releases the LAUNCH of
    # the next PDL kernel so its prologue overlaps this whole body;
    # consumers' own gdc_wait still fences on full completion.
    if USE_GDC:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()

    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
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
            ring_slot = i_n
            if step_idx < MAX_CACHE_LEN:
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
            b_q = b_q / (tl.sqrt(tl.sum(b_q * b_q) + 1e-6))
            b_k = b_k / (tl.sqrt(tl.sum(b_k * b_k) + 1e-6))

        b_q = b_q * scale

        # Apply gating to hidden state: h *= exp(g)
        if IS_KDA:
            b_h *= tl.exp(b_g[:, None])
        else:
            b_h *= tl.exp(b_g)

        # Delta rule: v -= sum(h * k, dim=0)
        b_v -= tl.sum(b_h * b_k[:, None], 0)

        # Apply beta gating: v *= beta
        b_v *= b_beta

        # Update hidden state: h += k[:, None] * v[None, :]
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
    cache_steps: Optional[
        int
    ] = None,  # kept for API compat; stride is derived from ``intermediate_states_buffer.shape[1]``
    retrieve_parent_token: Optional[torch.Tensor] = None,
    # fused ReplaySSM ring-write (spec verify). When cache_ring, each draft step
    # stores pre-norm k / raw v / gate / beta into these per-slot rings,
    # replacing the eager ring-write. Off by default -> decode unchanged.
    cache_ring: bool = False,
    replayssm_rawv: Optional[torch.Tensor] = None,
    replayssm_rawk: Optional[torch.Tensor] = None,
    replayssm_g: Optional[torch.Tensor] = None,
    replayssm_beta: Optional[torch.Tensor] = None,
    num_warps: int = 1,
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
    BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 32)
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

    grid = (NK, NV, N * HV)

    # Per-req stride must match the buffer's allocated dim, not runtime steps
    # (they can differ under --speculative-adaptive).
    cache_stride_steps = (
        intermediate_states_buffer.shape[1]
        if intermediate_states_buffer is not None
        else 0
    )

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

    if cache_ring:
        # Split path (NPU-optimized): gate computation is fully parallel
        # (no ring writes — just g/beta → temp buffers); ring writes are
        # folded into the recurrent kernel's compute-heavy loop where they
        # are hidden by the VEC pipeline.
        total_tokens = B * T

        if is_kda:
            g_buffer = torch.empty(
                total_tokens, HV * K, dtype=torch.float32, device=q.device
            )
        else:
            g_buffer = torch.empty(
                total_tokens, HV, dtype=torch.float32, device=q.device
            )
        beta_buffer = torch.empty(
            total_tokens, HV, dtype=torch.float32, device=q.device
        )

        grid_gate = (N * HV,)
        _gate_and_ring_write_kernel[grid_gate](
            A_log=A_log,
            a=a,
            dt_bias=dt_bias,
            b=b,
            g_buffer=g_buffer,
            beta_buffer=beta_buffer,
            softplus_beta=softplus_beta,
            softplus_threshold=softplus_threshold,
            lower_bound=lower_bound if lower_bound is not None else 0.0,
            cu_seqlens=cu_seqlens,
            stride_a=stride_a,
            stride_b=stride_b,
            T=T,
            B=B,
            H=H,
            HV=HV,
            K=K,
            BK=BK,
            IS_VARLEN=cu_seqlens is not None,
            IS_KDA=is_kda,
            USE_LOWER_BOUND=lower_bound is not None,
            num_warps=num_warps,
            num_stages=num_stages,
        )

        STEPS = T // N
        BV_RECURRENT = min(triton.next_power_of_2(V), 64)
        grid_recurrent = (N, HV, triton.cdiv(V, BV_RECURRENT))
        _replayssm_verify_recurrent_kernel[grid_recurrent](
            q,
            k,
            v,
            o,
            g_buffer,
            beta_buffer,
            initial_state_source if initial_state_source is not None else q,
            initial_state_indices if initial_state_indices is not None else q,
            replayssm_rawv if replayssm_rawv is not None else q,
            replayssm_rawk if replayssm_rawk is not None else q,
            replayssm_g if replayssm_g is not None else q,
            replayssm_beta if replayssm_beta is not None else q,
            initial_state_source.stride(0) if initial_state_source is not None else 0,
            scale,
            stride_q,
            stride_k,
            stride_v,
            stride_rawv_slot,
            stride_rawk_slot,
            stride_g_slot,
            stride_beta_slot,
            H=H,
            HV=HV,
            K=K,
            V=V,
            STEPS=STEPS,
            BK=BK,
            BV=BV_RECURRENT,
            MAX_CACHE_LEN=max_cache_len,
            CACHE_RING=True,
            USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
            IS_KDA=is_kda,
            USE_INITIAL_STATE=initial_state_source is not None,
            num_warps=1,
            num_stages=3,
        )
    else:
        # PDL (sm90+): chain this kernel behind its producer conv1d_update,
        # which already launches dependents. Bit-exact (scheduling only) —
        # benefits both KDA and GDN recurrent paths.
        pdl_kwargs = (
            {"USE_GDC": True, "launch_pdl": True} if is_arch_support_pdl() else {}
        )

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
            stride_h0_source=(
                initial_state_source.stride(0)
                if initial_state_source is not None
                else 0
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
            num_warps=num_warps,
            num_stages=num_stages,
            **pdl_kwargs,
        )
    o = o.squeeze(0)
    return o
