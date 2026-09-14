"""Fused KDA chain-verify kernel: causal-conv1d update + sigmoid-gating delta rule.

Fuses the KDA (Kimi Delta Attention) MTP target_verify hot path

    causal_conv1d_update (chain mode, SAVE_INTERMEDIATE)
  + fused_sigmoid_gating_delta_rule_update (T-step recurrence,
    intermediate-state caching, state update disabled)

into a single Triton kernel, removing per-layer-per-verify: one kernel
launch, the mixed_qkv HBM round-trip between conv and recurrence, and the
two transpose copies the unfused path needs to feed the conv kernel.

Scope (v1): chain speculation only (``speculative_eagle_topk == 1``, i.e.
``retrieve_next_token is None``). The tree path keeps the unfused reference
kernels. Requires ``T >= kernel_width - 1`` (the rolled conv state is then
exactly the last ``kernel_width - 1`` input tokens, matching the reference
kernel's store).

Numerics: deliberately bit-aligned with the unfused pair. The conv output is
rounded to the activation dtype (bf16) before entering the recurrence —
exactly what the unfused path does through its intermediate tensor — and all
expressions mirror the reference kernels line by line, with the same
num_warps so reduction order matches.
"""

from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.kernels.jit.utils import is_arch_support_pdl

# V-tile width of the fused verify kernel. Tuned on B200 at T=5 with
# benchmark/kernels/bench_kda_verify_sweep.py; any power of two is
# numerics-safe at num_warps=4 (bit-exact vs the BV=32 original).
KDA_VERIFY_BLOCK_V = 4


@triton.jit
def fused_kda_conv_gating_verify_kernel(
    x,  # [seq_len, dim] packed qkv, pre-conv
    w,  # [dim, W] conv weights
    conv_bias,  # [dim] or dummy
    conv_state,  # [lines, dim, state_len], dim contiguous
    conv_state_indices,  # [B]
    inter_conv_window,  # [lines, steps, dim, W-1] (as strides)
    inter_state_indices,  # [B]
    a,  # [seq_len, HV*K] gate input
    b_gate,  # [seq_len, HV] beta input
    A_log,  # [HV]
    dt_bias,  # [HV*K]
    lower_bound,
    softplus_beta,
    softplus_threshold,
    h0_source,  # [slots, HV, V, K] fp32 ssm states
    h0_indices,  # [B]
    inter_states,  # [lines, cache_steps, HV, V, K] fp32
    o,  # [seq_len, HV, V]
    scale,
    cache_steps,  # allocated step-dim of inter_states
    stride_x_tok,
    stride_w_dim,
    stride_cs_line,
    stride_cs_tok,
    stride_iw_line,
    stride_iw_step,
    stride_iw_dim,
    stride_iw_win,
    stride_a_tok,
    stride_b_tok,
    T: tl.constexpr,
    W: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
    SAVE_INTERMEDIATE_WINDOW: tl.constexpr,
    CACHE_INTERMEDIATE_STATES: tl.constexpr,
    USE_GDC: tl.constexpr = False,
):
    # PDL: overlap prologue with the tail of the producer qkv-projection GEMM;
    # every global load (conv_state_indices, mixed_qkv, weights) happens after
    # the wait. The immediate trigger releases the LAUNCH of the PDL'd gated
    # norm so its prologue overlaps this kernel's whole (long, latency-bound)
    # body -- consumers' own gdc_wait still fences on full completion. Fired
    # before the padded-slot early return so every CTA triggers explicitly.
    if USE_GDC:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()

    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    bos = i_n * T
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    # Packed channel offsets inside x / conv_state / weights.
    q_ch = i_h * K + o_k
    k_ch = H * K + i_h * K + o_k
    v_ch = 2 * H * K + i_hv * V + o_v

    # The q/k channels of head i_h are shared by every (v-tile, hv) program
    # mapping to it; exactly one of them owns the state/window writes so the
    # shared channels are written once (values are identical either way).
    is_qk_owner = (i_v == 0) & (i_hv % (HV // H) == 0)

    cs_idx = tl.load(conv_state_indices + i_n).to(tl.int64)
    # Padded rows carry -1 slots; the reference conv kernel early-returns on
    # them (their outputs are never consumed), so skip the whole program.
    if cs_idx < 0:
        return
    cs_base = conv_state + cs_idx * stride_cs_line

    # Conv history (state_len = W-1 columns, oldest -> newest). Matches the
    # reference kernel's col0..col2 preload (KERNEL_WIDTH == 4).
    tl.static_assert(W == 4, "fused KDA verify kernel supports kernel width 4")
    q_c0 = tl.load(cs_base + q_ch + 0 * stride_cs_tok, mask=mask_k, other=0.0)
    q_c1 = tl.load(cs_base + q_ch + 1 * stride_cs_tok, mask=mask_k, other=0.0)
    q_c2 = tl.load(cs_base + q_ch + 2 * stride_cs_tok, mask=mask_k, other=0.0)
    k_c0 = tl.load(cs_base + k_ch + 0 * stride_cs_tok, mask=mask_k, other=0.0)
    k_c1 = tl.load(cs_base + k_ch + 1 * stride_cs_tok, mask=mask_k, other=0.0)
    k_c2 = tl.load(cs_base + k_ch + 2 * stride_cs_tok, mask=mask_k, other=0.0)
    v_c0 = tl.load(cs_base + v_ch + 0 * stride_cs_tok, mask=mask_v, other=0.0)
    v_c1 = tl.load(cs_base + v_ch + 1 * stride_cs_tok, mask=mask_v, other=0.0)
    v_c2 = tl.load(cs_base + v_ch + 2 * stride_cs_tok, mask=mask_v, other=0.0)

    # Conv weights per channel group (column-major over width).
    wq0 = tl.load(w + q_ch * stride_w_dim + 0, mask=mask_k, other=0.0)
    wq1 = tl.load(w + q_ch * stride_w_dim + 1, mask=mask_k, other=0.0)
    wq2 = tl.load(w + q_ch * stride_w_dim + 2, mask=mask_k, other=0.0)
    wq3 = tl.load(w + q_ch * stride_w_dim + 3, mask=mask_k, other=0.0)
    wk0 = tl.load(w + k_ch * stride_w_dim + 0, mask=mask_k, other=0.0)
    wk1 = tl.load(w + k_ch * stride_w_dim + 1, mask=mask_k, other=0.0)
    wk2 = tl.load(w + k_ch * stride_w_dim + 2, mask=mask_k, other=0.0)
    wk3 = tl.load(w + k_ch * stride_w_dim + 3, mask=mask_k, other=0.0)
    wv0 = tl.load(w + v_ch * stride_w_dim + 0, mask=mask_v, other=0.0)
    wv1 = tl.load(w + v_ch * stride_w_dim + 1, mask=mask_v, other=0.0)
    wv2 = tl.load(w + v_ch * stride_w_dim + 2, mask=mask_v, other=0.0)
    wv3 = tl.load(w + v_ch * stride_w_dim + 3, mask=mask_v, other=0.0)

    if HAS_BIAS:
        bias_q = tl.load(conv_bias + q_ch, mask=mask_k, other=0.0).to(tl.float32)
        bias_k = tl.load(conv_bias + k_ch, mask=mask_k, other=0.0).to(tl.float32)
        bias_v = tl.load(conv_bias + v_ch, mask=mask_v, other=0.0).to(tl.float32)

    # Recurrent state tile [BK, BV] over the [V, K]-major state layout.
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    h0_idx = tl.load(h0_indices + i_n)
    if h0_idx >= 0:
        p_h0 = (
            h0_source
            + h0_idx.to(tl.int64) * HV * K * V
            + i_hv * K * V
            + o_v[None, :] * K
            + o_k[:, None]
        )
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    cache_idx = -1
    if CACHE_INTERMEDIATE_STATES:
        cache_idx = tl.load(inter_state_indices + i_n)
    iw_idx = tl.zeros([], dtype=tl.int64)
    if SAVE_INTERMEDIATE_WINDOW:
        iw_idx = tl.load(inter_state_indices + i_n).to(tl.int64)

    b_A_log = tl.load(A_log + i_hv).to(tl.float32)
    b_dt_bias = tl.load(dt_bias + i_hv * K + o_k, mask=mask_k, other=0.0).to(tl.float32)

    for t in tl.static_range(T):
        # ---- inline causal conv (reference: bias + c0*w0 + c1*w1 + c2*w2 + x*w3,
        # then silu; accumulation order and dtypes mirror the unfused kernel) ----
        x_q = tl.load(x + (bos + t) * stride_x_tok + q_ch, mask=mask_k, other=0.0)
        x_k = tl.load(x + (bos + t) * stride_x_tok + k_ch, mask=mask_k, other=0.0)
        x_v = tl.load(x + (bos + t) * stride_x_tok + v_ch, mask=mask_v, other=0.0)

        if HAS_BIAS:
            acc_q = bias_q
            acc_k = bias_k
            acc_v = bias_v
        else:
            acc_q = tl.zeros([BK], dtype=tl.float32)
            acc_k = tl.zeros([BK], dtype=tl.float32)
            acc_v = tl.zeros([BV], dtype=tl.float32)
        acc_q += q_c0 * wq0
        acc_q += q_c1 * wq1
        acc_q += q_c2 * wq2
        acc_q += x_q * wq3
        acc_k += k_c0 * wk0
        acc_k += k_c1 * wk1
        acc_k += k_c2 * wk2
        acc_k += x_k * wk3
        acc_v += v_c0 * wv0
        acc_v += v_c1 * wv1
        acc_v += v_c2 * wv2
        acc_v += x_v * wv3

        # Slide the window (reference: col0=col1; col1=col2; col2=x).
        q_c0 = q_c1
        q_c1 = q_c2
        q_c2 = x_q
        k_c0 = k_c1
        k_c1 = k_c2
        k_c2 = x_k
        v_c0 = v_c1
        v_c1 = v_c2
        v_c2 = x_v

        if SAVE_INTERMEDIATE_WINDOW:
            iw_base = inter_conv_window + iw_idx * stride_iw_line + t * stride_iw_step
            if is_qk_owner:
                tl.store(
                    iw_base + q_ch * stride_iw_dim + 0 * stride_iw_win,
                    q_c0,
                    mask=mask_k,
                )
                tl.store(
                    iw_base + q_ch * stride_iw_dim + 1 * stride_iw_win,
                    q_c1,
                    mask=mask_k,
                )
                tl.store(
                    iw_base + q_ch * stride_iw_dim + 2 * stride_iw_win,
                    q_c2,
                    mask=mask_k,
                )
                tl.store(
                    iw_base + k_ch * stride_iw_dim + 0 * stride_iw_win,
                    k_c0,
                    mask=mask_k,
                )
                tl.store(
                    iw_base + k_ch * stride_iw_dim + 1 * stride_iw_win,
                    k_c1,
                    mask=mask_k,
                )
                tl.store(
                    iw_base + k_ch * stride_iw_dim + 2 * stride_iw_win,
                    k_c2,
                    mask=mask_k,
                )
            tl.store(
                iw_base + v_ch * stride_iw_dim + 0 * stride_iw_win, v_c0, mask=mask_v
            )
            tl.store(
                iw_base + v_ch * stride_iw_dim + 1 * stride_iw_win, v_c1, mask=mask_v
            )
            tl.store(
                iw_base + v_ch * stride_iw_dim + 2 * stride_iw_win, v_c2, mask=mask_v
            )

        # SiLU, then round to the activation dtype: the unfused path stores the
        # conv output to a bf16 tensor and reloads it for the recurrence; the
        # explicit round-trip keeps the fused kernel bit-identical.
        acc_q = acc_q / (1 + tl.exp(-acc_q))
        acc_k = acc_k / (1 + tl.exp(-acc_k))
        acc_v = acc_v / (1 + tl.exp(-acc_v))
        b_q = acc_q.to(o.dtype.element_ty).to(tl.float32)
        b_k = acc_k.to(o.dtype.element_ty).to(tl.float32)
        b_v = acc_v.to(o.dtype.element_ty).to(tl.float32)

        # ---- sigmoid-gating delta rule step (mirrors the reference kernel) ----
        b_b = tl.load(b_gate + (bos + t) * stride_b_tok + i_hv).to(tl.float32)
        b_a = tl.load(
            a + (bos + t) * stride_a_tok + i_hv * K + o_k, mask=mask_k, other=0.0
        ).to(tl.float32)

        gx = b_a + b_dt_bias
        if USE_LOWER_BOUND:
            b_g = lower_bound * tl.sigmoid(tl.exp(b_A_log) * gx)
        else:
            beta_x = softplus_beta * gx
            softplus_x = tl.where(
                beta_x <= softplus_threshold,
                (1.0 / softplus_beta) * tl.log(1.0 + tl.exp(beta_x)),
                gx,
            )
            b_g = -tl.exp(b_A_log) * softplus_x

        b_beta = 1.0 / (1.0 + tl.exp(-b_b))

        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / (tl.sqrt(tl.sum(b_q * b_q) + 1e-6))
            b_k = b_k / (tl.sqrt(tl.sum(b_k * b_k) + 1e-6))

        b_q = b_q * scale

        b_h *= tl.exp(b_g[:, None])
        b_v -= tl.sum(b_h * b_k[:, None], 0)
        b_v *= b_beta
        b_h += b_k[:, None] * b_v[None, :]
        b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(
            o + ((bos + t) * HV + i_hv) * V + o_v,
            b_o.to(o.dtype.element_ty),
            mask=mask_v,
        )

        if CACHE_INTERMEDIATE_STATES:
            if cache_idx >= 0:
                cache_ptr = (
                    inter_states
                    + cache_idx.to(tl.int64) * cache_steps * HV * K * V
                    + t * HV * K * V
                    + i_hv * K * V
                    + o_v[None, :] * K
                    + o_k[:, None]
                )
                tl.store(cache_ptr, b_h.to(cache_ptr.dtype.element_ty), mask=mask_h)

    # Rolled conv state after consuming T >= W-1 tokens is exactly the last
    # W-1 input tokens — which are the current window registers. The verify
    # pass never writes the ssm state back (rollback happens at commit).
    if is_qk_owner:
        tl.store(cs_base + q_ch + 0 * stride_cs_tok, q_c0, mask=mask_k)
        tl.store(cs_base + q_ch + 1 * stride_cs_tok, q_c1, mask=mask_k)
        tl.store(cs_base + q_ch + 2 * stride_cs_tok, q_c2, mask=mask_k)
        tl.store(cs_base + k_ch + 0 * stride_cs_tok, k_c0, mask=mask_k)
        tl.store(cs_base + k_ch + 1 * stride_cs_tok, k_c1, mask=mask_k)
        tl.store(cs_base + k_ch + 2 * stride_cs_tok, k_c2, mask=mask_k)
    tl.store(cs_base + v_ch + 0 * stride_cs_tok, v_c0, mask=mask_v)
    tl.store(cs_base + v_ch + 1 * stride_cs_tok, v_c1, mask=mask_v)
    tl.store(cs_base + v_ch + 2 * stride_cs_tok, v_c2, mask=mask_v)


def fused_kda_conv_gating_verify(
    mixed_qkv: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: Optional[torch.Tensor],
    conv_state: torch.Tensor,
    conv_state_indices: torch.Tensor,
    intermediate_conv_window: Optional[torch.Tensor],
    intermediate_state_indices: Optional[torch.Tensor],
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    ssm_states: torch.Tensor,
    cache_indices: torch.Tensor,
    intermediate_states_buffer: Optional[torch.Tensor],
    scale: float,
    T: int,
    num_q_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    lower_bound: Optional[float] = None,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
    use_qk_l2norm_in_kernel: bool = True,
    # num_warps=4 is ~1.3x faster than the unfused pair in-graph; the output,
    # conv_state and conv-window caches stay bit-identical to the reference.
    # Only the fp32 intermediate-ssm rollback cache differs: the tl.sum
    # reduction-order delta (~1 ulp/step) compounds through the delta-rule
    # recurrence — measured ~6e-8 at T=4 standard gate (the production MTP
    # shape), ~1.5e-5 at T=4 safe gate, ~2e-3 at T=8 safe gate. num_warps=1
    # reproduces the reference reduction order exactly (all buffers
    # bit-identical) but is ~2.4x slower in-graph — numerics debugging only.
    num_warps: int = 4,
) -> torch.Tensor:
    """Chain-verify fast path. Returns ``o`` of shape [1, seq_len, HV, V],
    matching the unfused ``target_verify`` output layout."""
    H, HV, K, V = num_q_heads, num_v_heads, head_k_dim, head_v_dim
    seq_len, dim = mixed_qkv.shape
    B = seq_len // T
    W = conv_weight.shape[1]

    assert mixed_qkv.stride(-1) == 1, "mixed_qkv must be contiguous in dim"
    assert dim == 2 * H * K + HV * V, f"packed dim mismatch: {dim}"
    assert W == 4, "fused KDA verify supports conv width 4 only"
    assert T >= W - 1, "fused KDA verify requires T >= conv width - 1"
    assert seq_len == B * T
    assert conv_state.stride(1) == 1, "conv_state must be dim-contiguous"
    assert conv_weight.stride(1) == 1
    assert ssm_states.is_contiguous()
    BK = triton.next_power_of_2(K)
    assert BK == K, "K must be a power of two (NK==1)"
    # Smaller V tiles keep winning on this latency-bound grid (serial T-step
    # recurrence per CTA; more CTAs = shorter per-step chains, and the
    # duplicated per-head q/k conv work stays cheaper than the parallelism
    # gain all the way down): B200 T=5 sweep (us/layer, warps=4) measured
    # 4 -> 11.56, 8 -> 12.53, 16 -> 12.83, 32 -> 14.26, 64 -> 20.7,
    # 128 -> 38 (benchmark/kernels/bench_kda_verify_sweep.py; H20-3e ranks
    # 16 first but B200 is the production target). Bit-exact across BV at
    # num_warps=4: the V tiling never touches the K-axis reduction order.
    # BV=128 (the norm-fusion single-tile probe) measured 2x slower -- folding
    # the gated RMSNorm into this kernel's epilogue is a dead end; it is
    # PDL-chained behind this kernel instead (see fused_norm_gate.py).
    BV = min(triton.next_power_of_2(V), KDA_VERIFY_BLOCK_V)
    NV = triton.cdiv(V, BV)

    a2 = a.reshape(seq_len, HV * K)
    b2 = b.reshape(seq_len, HV)
    assert a2.stride(-1) == 1 and b2.stride(-1) == 1

    o = mixed_qkv.new_empty(seq_len, HV, V)

    if intermediate_conv_window is not None:
        s_iw = intermediate_conv_window.stride()
        s_iw_line, s_iw_step, s_iw_dim, s_iw_win = s_iw[0], s_iw[1], s_iw[2], s_iw[3]
        assert intermediate_state_indices is not None
    else:
        s_iw_line = s_iw_step = s_iw_dim = s_iw_win = 0

    cache_steps = (
        intermediate_states_buffer.shape[1]
        if intermediate_states_buffer is not None
        else 0
    )
    if intermediate_states_buffer is not None:
        assert intermediate_states_buffer.is_contiguous()

    grid = (NV, B * HV)
    # PDL (sm90+): chain behind the producer qkv-projection GEMM and signal the
    # downstream o_norm / o_proj. Scheduling only — bit-exactness unaffected.
    pdl_kwargs = {"USE_GDC": True, "launch_pdl": True} if is_arch_support_pdl() else {}
    fused_kda_conv_gating_verify_kernel[grid](
        x=mixed_qkv,
        w=conv_weight,
        conv_bias=conv_bias if conv_bias is not None else conv_weight,
        conv_state=conv_state,
        conv_state_indices=conv_state_indices,
        inter_conv_window=(
            intermediate_conv_window
            if intermediate_conv_window is not None
            else mixed_qkv
        ),
        inter_state_indices=(
            intermediate_state_indices
            if intermediate_state_indices is not None
            else conv_state_indices
        ),
        a=a2,
        b_gate=b2,
        A_log=A_log.reshape(-1),
        dt_bias=dt_bias.reshape(-1),
        lower_bound=lower_bound,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
        h0_source=ssm_states,
        h0_indices=cache_indices,
        inter_states=(
            intermediate_states_buffer
            if intermediate_states_buffer is not None
            else ssm_states
        ),
        o=o,
        scale=scale,
        cache_steps=cache_steps,
        stride_x_tok=mixed_qkv.stride(0),
        stride_w_dim=conv_weight.stride(0),
        stride_cs_line=conv_state.stride(0),
        stride_cs_tok=conv_state.stride(2),
        stride_iw_line=s_iw_line,
        stride_iw_step=s_iw_step,
        stride_iw_dim=s_iw_dim,
        stride_iw_win=s_iw_win,
        stride_a_tok=a2.stride(0),
        stride_b_tok=b2.stride(0),
        T=T,
        W=W,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        HAS_BIAS=conv_bias is not None,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        USE_LOWER_BOUND=lower_bound is not None,
        SAVE_INTERMEDIATE_WINDOW=intermediate_conv_window is not None,
        CACHE_INTERMEDIATE_STATES=intermediate_states_buffer is not None,
        # num_warps=1 matches the reference kernels' reduction order exactly;
        # higher values must be re-validated for bit-exactness before use.
        num_warps=num_warps,
        num_stages=3,
        **pdl_kwargs,
    )
    return o.view(1, seq_len, HV, V)
