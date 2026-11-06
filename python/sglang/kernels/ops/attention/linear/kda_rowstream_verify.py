"""Row-streamed KDA chain verify for ReplaySSM ring mode.

In ReplaySSM ring mode (``--enable-linear-replayssm-spec``) the dense chain
verify produces exactly two things the engine consumes: the per-token outputs
and the ring entries (raw k / raw v / gate / beta). It never needs the per-step
state snapshots nor the final state -- the commit fold replays the ring. The
generic recurrent kernel (``fused_sigmoid_gating_delta_rule_update``) still
materialises ``h_t`` in registers and walks the draft window one token at a
time: every step does a full ``[K, BV]`` decay, a cross-lane k-reduction, a
rank-1 update and a cross-lane q-reduction, i.e. ``4 * T`` serial full-tile
passes with two block reductions each.

This kernel streams each row's state tile ``h0`` once and applies all ``T``
steps in chunked (WY) form against that single tile::

    G_t   = sum_{s<=t} g_s                           (per-K, log space)
    W     = (exp(G) * k_hat) @ h0                    [T, BV]
    Qh    = (exp(G) * q_hat) @ h0                    [T, BV]
    A_ts  = k_hat_t . (exp(G_t - G_s) * k_hat_s)     s <  t   (state-free)
    M_ts  = q_hat_t . (exp(G_t - G_s) * k_hat_s)     s <= t   (state-free)
    u_t   = beta_t * (v_t - W_t - sum_{s<t} A_ts u_s)
    o     = Qh + M @ u

which is the closed form of the sequential recurrence

    h_t = exp(g_t) h_{t-1} + k_t u_t^T
    u_t = beta_t (v_t - (exp(g_t) h_{t-1})^T k_t)
    o_t = h_t^T q_t

that ``fused_sigmoid_gating_delta_rule_update`` evaluates step by step. All four
contractions become ``tl.dot``s, and the only serial chain left is the
``T x T`` forward substitution, which never touches the state tile.

The ring entries are formed from the same expressions, at the same point in the
kernel (pre-l2norm k, pre-delta v, in-kernel gate/beta), so the commit-time
replay sees bit-identical inputs. The outputs are not bit-identical to the
recurrent kernel -- the chunked form reassociates the same sums, exactly as the
chunked prefill scan does relative to the recurrent one. Measured on MI355X the
deviation stays within one BF16 ulp of the recurrent output.
"""

from typing import Optional

import torch
import triton
import triton.language as tl

# tl.dot requires M >= 16; draft windows are shorter than that.
_MIN_DOT_M = 16
# Decay exponents are split into exp(G_t - G_ref) * exp(G_ref - G_s) so the pair
# matrices become one dot. Each half is clamped into the FP32 exponent range,
# exactly as the chunked KDA prefill scan clamps its own split (chunk_intra.py);
# a draft window is at most a few tokens, so the clamp is far looser here.
_MAX_DECAY_EXP = tl.constexpr(80.0)
# ROCm lowers an FP32 ``tl.dot`` to the narrow v_mfma_f32_16x16x4_f32 pipe;
# the 3-way BF16 split keeps ~FP32 accuracy on the wide BF16 MFMA pipe instead.
# (The chunked KDA prefill scan truncates the same products to plain BF16.)
_DOT_PRECISION = "bf16x3" if torch.version.hip is not None else "ieee"


@triton.jit
def _kda_rowstream_verify_ring_kernel(
    A_log,
    a,
    dt_bias,
    lower_bound,
    softplus_beta,
    softplus_threshold,
    q,
    k,
    v,
    b,
    o,
    h0_source,
    h0_indices,
    stride_h0_source,
    cu_seqlens,
    ring_indices,
    replayssm_rawv,
    replayssm_rawk,
    replayssm_g,
    replayssm_beta,
    stride_rawv_slot,
    stride_rawk_slot,
    stride_g_slot,
    stride_beta_slot,
    scale,
    stride_a,
    stride_q,
    stride_k,
    stride_v,
    stride_b,
    T: tl.constexpr,
    BT: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
    MAX_CACHE_LEN: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
    SEPARATE_RING_INDEX: tl.constexpr,
    PREC: tl.constexpr,
):
    i_v, i_n, i_hv = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h = i_hv // (HV // H)

    # Graph-padded rows carry no tokens; their output slots do not exist.
    bos = tl.load(cu_seqlens + i_n).to(tl.int64)
    eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
    row_len = eos - bos

    o_t = tl.arange(0, BT)
    o_k = tl.arange(0, K)
    o_v = i_v * BV + tl.arange(0, BV)
    m_t = o_t < row_len
    m_k = o_k < K
    m_v = o_v < V
    m_tk = m_t[:, None] & m_k[None, :]
    m_tv = m_t[:, None] & m_v[None, :]

    # Stream the committed state tile once. Issued first so the load latency
    # overlaps the gate prologue below.
    idx = tl.load(h0_indices + i_n).to(tl.int64)
    p_h0 = (
        h0_source
        + tl.maximum(idx, 0) * stride_h0_source
        + i_hv * K * V
        + o_v[None, :] * K
        + o_k[:, None]
    )
    b_h = tl.load(p_h0, mask=m_k[:, None] & m_v[None, :], other=0.0).to(tl.float32)
    b_h = tl.where(idx >= 0, b_h, 0.0)

    # Whole draft window at once: [BT, K] / [BT, BV].
    tok = bos + o_t
    b_q = tl.load(
        q + tok[:, None] * stride_q + i_h * K + o_k[None, :], mask=m_tk, other=0.0
    ).to(tl.float32)
    b_k = tl.load(
        k + tok[:, None] * stride_k + i_h * K + o_k[None, :], mask=m_tk, other=0.0
    ).to(tl.float32)
    b_v = tl.load(
        v + tok[:, None] * stride_v + i_hv * V + o_v[None, :], mask=m_tv, other=0.0
    ).to(tl.float32)
    b_b = tl.load(b + tok * stride_b + i_hv, mask=m_t, other=0.0).to(tl.float32)
    b_a = tl.load(
        a + tok[:, None] * stride_a + i_hv * K + o_k[None, :], mask=m_tk, other=0.0
    ).to(tl.float32)
    b_A_log = tl.load(A_log + i_hv).to(tl.float32)
    b_dt_bias = tl.load(dt_bias + i_hv * K + o_k, mask=m_k, other=0.0).to(tl.float32)

    # Gate / beta, formed exactly as fused_sigmoid_gating_delta_rule_update does.
    x = b_a + b_dt_bias[None, :]
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
    # Padded steps must not decay the real ones through the cumsum below.
    b_g = tl.where(m_tk, b_g, 0.0)
    b_beta = 1.0 / (1.0 + tl.exp(-b_b))

    # Ring write: raw k (pre-l2norm), raw v (pre-delta), in-kernel gate/beta.
    # Steps past the ring length are past the committable prefix; dropping them
    # keeps the write inside this slot (see the recurrent kernel's CACHE_RING).
    if SEPARATE_RING_INDEX:
        ring_slot = tl.load(ring_indices + i_n).to(tl.int64)
    else:
        ring_slot = idx
    if ring_slot >= 0:
        m_ring = m_t & (o_t < MAX_CACHE_LEN)
        tl.store(
            replayssm_rawv
            + ring_slot * stride_rawv_slot
            + i_hv * MAX_CACHE_LEN * V
            + o_t[:, None] * V
            + o_v[None, :],
            b_v.to(replayssm_rawv.dtype.element_ty),
            mask=m_ring[:, None] & m_v[None, :],
        )
        if i_v == 0:
            m_ring_k = m_ring[:, None] & m_k[None, :]
            # rawk is indexed by the k-head, shared across a GQA group: only the
            # first v-head of each group writes it.
            if i_hv % (HV // H) == 0:
                tl.store(
                    replayssm_rawk
                    + ring_slot * stride_rawk_slot
                    + i_h * MAX_CACHE_LEN * K
                    + o_t[:, None] * K
                    + o_k[None, :],
                    b_k.to(replayssm_rawk.dtype.element_ty),
                    mask=m_ring_k,
                )
            tl.store(
                replayssm_g
                + ring_slot * stride_g_slot
                + i_hv * MAX_CACHE_LEN * K
                + o_t[:, None] * K
                + o_k[None, :],
                b_g,
                mask=m_ring_k,
            )
            tl.store(
                replayssm_beta
                + ring_slot * stride_beta_slot
                + i_hv * MAX_CACHE_LEN
                + o_t,
                b_beta,
                mask=m_ring,
            )

    # L2 norm (+ the q scale), as USE_QK_L2NORM_IN_KERNEL does per step.
    q_rsqrt = 1.0 / tl.sqrt(tl.sum(b_q * b_q, 1) + 1e-6)
    k_rsqrt = 1.0 / tl.sqrt(tl.sum(b_k * b_k, 1) + 1e-6)
    b_q = b_q * (q_rsqrt * scale)[:, None]
    b_k = b_k * k_rsqrt[:, None]

    # Cumulative log-decay, per K channel.
    b_G = tl.cumsum(b_g, axis=0)
    b_eG = tl.exp(b_G)

    # State-free intra-window pair matrices: A (strictly lower) and M (lower
    # inclusive). Both contract over K with a per-channel decay, so split the
    # exponent around the window's last cumulative gate and let one dot do it.
    b_G_ref = tl.sum(tl.where((o_t == T - 1)[:, None], b_G, 0.0), 0)
    b_dG = b_G - b_G_ref[None, :]
    b_dec_row = tl.exp(tl.clamp(b_dG, -_MAX_DECAY_EXP, _MAX_DECAY_EXP))
    b_dec_col = tl.exp(tl.clamp(-b_dG, -_MAX_DECAY_EXP, _MAX_DECAY_EXP))
    b_kd = tl.trans(b_k * b_dec_col)
    b_A = tl.dot(b_k * b_dec_row, b_kd, input_precision=PREC)
    b_M = tl.dot(b_q * b_dec_row, b_kd, input_precision=PREC)
    b_A = tl.where(o_t[:, None] > o_t[None, :], b_A, 0.0)
    b_M = tl.where(o_t[:, None] >= o_t[None, :], b_M, 0.0)

    # The only two passes over the streamed state tile.
    b_W = tl.dot(b_eG * b_k, b_h, input_precision=PREC)
    b_Qh = tl.dot(b_eG * b_q, b_h, input_precision=PREC)

    # Forward substitution: u_t = beta_t (v_t - W_t - sum_{s<t} A_ts u_s).
    b_R = b_v - b_W
    b_U = tl.zeros([BT, BV], dtype=tl.float32)
    for t in tl.static_range(T):
        sel = o_t == t
        b_a_row = tl.sum(tl.where(sel[:, None], b_A, 0.0), 0)
        b_r_t = tl.sum(tl.where(sel[:, None], b_R, 0.0), 0)
        b_beta_t = tl.sum(tl.where(sel, b_beta, 0.0), 0)
        b_u_t = b_beta_t * (b_r_t - tl.sum(b_a_row[:, None] * b_U, 0))
        b_U = tl.where(sel[:, None], b_u_t[None, :], b_U)

    b_o = b_Qh + tl.dot(b_M, b_U, input_precision=PREC)
    tl.store(
        o + (tok[:, None] * HV + i_hv) * V + o_v[None, :],
        b_o.to(o.dtype.element_ty),
        mask=m_tv,
    )


def kda_rowstream_verify_ring(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    *,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    draft_token_num: int,
    replayssm_rawv: torch.Tensor,
    replayssm_rawk: torch.Tensor,
    replayssm_g: torch.Tensor,
    replayssm_beta: torch.Tensor,
    ring_indices: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    lower_bound: Optional[float] = None,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
    block_v: Optional[int] = None,
    num_warps: Optional[int] = None,
) -> torch.Tensor:
    """Dense chain target-verify in ReplaySSM ring mode.

    ``q`` / ``k`` are ``[1, N*T, H, K]``, ``v`` is ``[1, N*T, HV, V]``, ``a`` is
    ``[N*T, HV*K]`` and ``b`` is ``[N*T, HV]`` -- the layout
    ``fused_sigmoid_gating_delta_rule_update`` takes on the verify path. Returns
    the per-token outputs shaped like ``v``, and fills the ReplaySSM ring. The
    committed state pool is never written (verify is ``disable_state_update``).
    """
    H, K = q.shape[-2], q.shape[-1]
    HV, V = v.shape[-2], v.shape[-1]
    T = draft_token_num

    if scale is None:
        scale = K**-0.5
    if block_v is None:
        # One CTA per (row, head): the pair matrices above are per-row work, so
        # splitting V would recompute them once per tile.
        block_v = V
    BV = min(block_v, V)
    if num_warps is None:
        num_warps = 4
    BT = max(_MIN_DOT_M, triton.next_power_of_2(T))

    o = torch.empty(v.shape, dtype=q.dtype, device=v.device)
    n_rows = cu_seqlens.shape[0] - 1
    grid = (triton.cdiv(V, BV), n_rows, HV)
    _kda_rowstream_verify_ring_kernel[grid](
        A_log,
        a,
        dt_bias,
        lower_bound if lower_bound is not None else 0.0,
        softplus_beta,
        softplus_threshold,
        q,
        k,
        v,
        b,
        o,
        initial_state_source,
        initial_state_indices,
        initial_state_source.stride(0),
        cu_seqlens,
        ring_indices if ring_indices is not None else initial_state_indices,
        replayssm_rawv,
        replayssm_rawk,
        replayssm_g,
        replayssm_beta,
        replayssm_rawv.stride(0),
        replayssm_rawk.stride(0),
        replayssm_g.stride(0),
        replayssm_beta.stride(0),
        scale,
        a.stride(-2),
        q.stride(-3),
        k.stride(-3),
        v.stride(-3),
        b.stride(-2),
        T=T,
        BT=BT,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BV=BV,
        MAX_CACHE_LEN=replayssm_rawv.shape[-2],
        USE_LOWER_BOUND=lower_bound is not None,
        SEPARATE_RING_INDEX=ring_indices is not None,
        PREC=_DOT_PRECISION,
        num_warps=num_warps,
        num_stages=1,
    )
    return o
