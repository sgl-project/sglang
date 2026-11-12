"""ROCm: Inkling2 MTP kernels that fuse what the shared ops in mtp.py run as separate
launches (the f_rnn store, the Markov RNN step, the draft's context K/V store)."""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from sglang.kernels.ops.inkling2.mtp import _bf16, _mix_rmsnorm_row


@triton.jit
def _mix_rmsnorm_last(
    X,
    CommitLens,
    Slots,
    ALPHA,
    WEIGHT,
    OUT,
    ROW_STRIDE: tl.constexpr,
    TAP_STRIDE: tl.constexpr,
    HIDDEN_STRIDE: tl.constexpr,
    WIDTH: tl.constexpr,
    TAPS: tl.constexpr,
    HIDDEN: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    req = tl.program_id(0)
    h = tl.arange(0, BLOCK)
    last = tl.load(CommitLens + req).to(tl.int64) - 1
    last = tl.where(last < 0, last + WIDTH, last)
    out = _mix_rmsnorm_row(
        X + (req * WIDTH + last) * ROW_STRIDE,
        ALPHA,
        WEIGHT,
        TAP_STRIDE,
        HIDDEN_STRIDE,
        TAPS,
        HIDDEN,
        EPS,
        BLOCK,
    )
    # Rounded to X's dtype, as mix_rmsnorm's output is before the copy into OUT.
    out = out.to(X.dtype.element_ty)
    slot = tl.load(Slots + req).to(tl.int64)
    tl.store(OUT + slot * HIDDEN + h, out, h < HIDDEN)


def mix_rmsnorm_last(targets, width, commit_lens, slots, alpha, weight, eps, out):
    """out[slots[r]] = mix_rmsnorm(targets.view(R, width, ...)[r, commit_lens[r] - 1])
    in one kernel: the gather, the norm and the copy."""
    rows, taps, hidden = targets.shape
    reqs = commit_lens.numel()
    assert rows == reqs * width and slots.shape == (reqs,)
    assert alpha.shape == (taps,) and alpha.is_contiguous()
    assert weight.shape == (hidden,) and weight.is_contiguous()
    assert out.shape[1:] == (hidden,) and out.is_contiguous()
    if reqs:
        _mix_rmsnorm_last[(reqs,)](
            targets,
            commit_lens,
            slots,
            alpha,
            weight,
            out,
            *targets.stride(),
            width,
            taps,
            hidden,
            eps,
            triton.next_power_of_2(hidden),
            num_warps=8,
        )


@triton.jit
def _rnn_proj_partials(
    State,
    Tokens,
    Embedding,
    Hidden,
    FRnn,
    W,
    Partials,
    rows,
    token_stride,
    hidden_stride,
    R: tl.constexpr,
    H: tl.constexpr,
    MP: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    # One BK slice of K per program, read from the input it falls in (BK divides R
    # and H, so none straddles two): rnn_inputs' concatenation is never written.
    n_block = tl.program_id(0)
    split = tl.program_id(1)
    K: tl.constexpr = 2 * R + 2 * H
    k0 = split * BK
    m = tl.arange(0, MP)
    m_mask = m < rows
    rows_m = m.to(tl.int64)
    k = tl.arange(0, BK)
    if k0 < R:
        x = tl.load(State + rows_m[:, None] * R + (k0 + k)[None, :], m_mask[:, None], 0)
    elif k0 < 2 * R:
        token = tl.load(Tokens + rows_m * token_stride, m_mask, 0).to(tl.int64)
        x = tl.load(
            Embedding + token[:, None] * R + (k0 - R + k)[None, :], m_mask[:, None], 0
        )
    elif k0 < 2 * R + H:
        x = tl.load(
            Hidden + rows_m[:, None] * hidden_stride + (k0 - 2 * R + k)[None, :],
            m_mask[:, None],
            0,
        )
    else:
        x = tl.load(
            FRnn + rows_m[:, None] * H + (k0 - 2 * R - H + k)[None, :],
            m_mask[:, None],
            0,
        )
    n = n_block * BN + tl.arange(0, BN)
    w = tl.load(W + n[:, None] * K + (k0 + k)[None, :]).to(tl.float32)
    partial = tl.sum(x.to(tl.float32)[:, None, :] * w[None, :, :], 2)
    tl.store(Partials + (split * MP + m[:, None]) * (3 * R) + n[None, :], partial)


@triton.jit
def _rnn_gates_split(
    Partials,
    Bias,
    State,
    NewState,
    Latent,
    R: tl.constexpr,
    MP: tl.constexpr,
    SPLITS: tl.constexpr,
    SPLITS_POW2: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # rnn_proj's output (the split sum in split order plus the bias, in BF16), then
    # _rnn_gates on it.
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    s = tl.arange(0, SPLITS_POW2)
    partials = Partials + (s[:, None] * MP + row) * (3 * R) + cols[None, :]
    s_mask = (s < SPLITS)[:, None]
    gate_raw = tl.sum(tl.load(partials, s_mask, 0), 0)
    cand_raw = tl.sum(tl.load(partials + R, s_mask, 0), 0)
    out_raw = tl.sum(tl.load(partials + 2 * R, s_mask, 0), 0)
    gate_raw = _bf16(gate_raw + tl.load(Bias + cols).to(tl.float32))
    cand_raw = _bf16(cand_raw + tl.load(Bias + R + cols).to(tl.float32))
    out_raw = _bf16(out_raw + tl.load(Bias + 2 * R + cols).to(tl.float32))
    state = tl.load(State + row * R + cols).to(tl.float32)
    gate = _bf16(tl.div_rn(1.0, 1.0 + libdevice.exp(-gate_raw)))
    kept = _bf16(gate * state)
    blend = _bf16(_bf16(1.0 - gate) * _bf16(libdevice.tanh(cand_raw)))
    tl.store(NewState + row * R + cols, kept + blend)
    tl.store(Latent + row * R + cols, libdevice.tanh(out_raw))


# rnn_step's FMA split-K GEMV serves up to this many rows; more take rnn_proj's GEMM.
RNN_STEP_MAX_ROWS = 4
_RNN_STEP_BK = 256


def rnn_step_fits(state, hidden, weight, bias) -> bool:
    """Whether rnn_step takes this Inkling2MTPHead._rnn_step."""
    rows, rank = state.shape
    width = hidden.shape[1]
    return (
        0 < rows <= RNN_STEP_MAX_ROWS
        and state.dtype == hidden.dtype == weight.dtype == torch.bfloat16
        and bias is not None
        and rank % 64 == 0
        and rank % _RNN_STEP_BK == 0
        and width % _RNN_STEP_BK == 0
        and weight.shape == (3 * rank, 2 * rank + 2 * width)
        and weight.is_contiguous()
    )


def rnn_step(state, tokens, embedding, hidden, f_rnn, weight, bias):
    """rnn_gates(F.linear(rnn_inputs(state, tokens, embedding, hidden, f_rnn),
    weight, bias), state) in two launches for a few rows (rnn_step_fits): split-K
    partials of the projection read straight from its four inputs, then the split
    sum, the bias and the gates. The projection's sum order is not the GEMM's."""
    rows, rank = state.shape
    width = hidden.shape[1]
    assert state.is_contiguous() and f_rnn.is_contiguous() and hidden.stride(1) == 1
    assert state.dtype == hidden.dtype == f_rnn.dtype and embedding.is_contiguous()
    splits = (2 * rank + 2 * width) // _RNN_STEP_BK
    mp = triton.next_power_of_2(rows)
    partials = torch.empty(
        (splits, mp, 3 * rank), dtype=torch.float32, device=state.device
    )
    # Swept on MI350X (rank 256, hidden 4608, 1-4 rows): 4 outputs per one-wave
    # program; wider programs or K slices are slower.
    _rnn_proj_partials[(3 * rank // 4, splits)](
        state,
        tokens,
        embedding,
        hidden,
        f_rnn,
        weight,
        partials,
        rows,
        tokens.stride(0),
        hidden.stride(0),
        rank,
        width,
        mp,
        4,
        _RNN_STEP_BK,
        num_warps=1,
    )
    new_state, latent = state.new_empty((rows, rank)), state.new_empty((rows, rank))
    _rnn_gates_split[(rows, rank // 64)](
        partials,
        bias,
        state,
        new_state,
        latent,
        rank,
        mp,
        splits,
        triton.next_power_of_2(splits),
        64,
        num_warps=1,
        enable_fp_fusion=False,
    )
    return new_state, latent


@triton.jit
def _ctx_kv_store(
    KV,
    Pos,
    W,
    CosSin,
    KCache,
    VCache,
    Loc,
    CommitLens,
    kv_stride,
    k_cache_stride,
    v_cache_stride,
    eps,
    WIDTH: tl.constexpr,
    KV_SIZE: tl.constexpr,
    D: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    req = row // WIDTH
    if row - req * WIDTH < tl.load(CommitLens + req):
        HALF: tl.constexpr = D // 2
        d = tl.arange(0, D)
        lo = tl.arange(0, HALF)
        k = KV + row * kv_stride + head * D
        # K's RMSNorm as rmsnorm_triton computes it.
        x = tl.load(k + d, d < D, other=0).to(tl.float32)
        inv = tl.math.rsqrt(tl.sum(x * x, 0) / D + eps)
        x1 = tl.load(k + lo).to(tl.float32) * inv
        x1 = _bf16(x1 * tl.load(W + lo).to(tl.float32))
        x2 = tl.load(k + HALF + lo).to(tl.float32) * inv
        x2 = _bf16(x2 * tl.load(W + HALF + lo).to(tl.float32))
        # Neox RoPE as the HIP rotary kernel computes it in BF16: cos and sin, each
        # product and each sum rounded to BF16.
        cos_sin = CosSin + tl.load(Pos + row) * D
        cos = _bf16(tl.load(cos_sin + lo))
        sin = _bf16(tl.load(cos_sin + HALF + lo))
        o1 = _bf16(x1 * cos) - _bf16(x2 * sin)
        o2 = _bf16(x2 * cos) + _bf16(x1 * sin)
        loc = tl.load(Loc + row)
        k_out = KCache + loc * k_cache_stride + head * D
        tl.store(k_out + lo, o1.to(KCache.dtype.element_ty))
        tl.store(k_out + HALF + lo, o2.to(KCache.dtype.element_ty))
        v = tl.load(KV + row * kv_stride + KV_SIZE + head * D + d)
        tl.store(VCache + loc * v_cache_stride + head * D + d, v)


def ctx_kv_store(
    kv, positions, k_weight, eps, cos_sin_cache, k_cache, v_cache, loc_2d, commit_lens
):
    """Store the DSpark draft's context K / V rows (kv_proj_ctx's output) as
    k_norm_ctx, the HIP rotary kernel's neox RoPE and set_kv_buffer_prefix_valid do:
    row (r, j) of loc_2d is written for j < commit_lens[r]."""
    heads, d = k_cache.shape[1:]
    rows = loc_2d.numel()
    assert kv.shape == (rows, 2 * heads * d) and kv.stride(-1) == 1
    assert k_cache.dtype == v_cache.dtype == kv.dtype == torch.bfloat16
    assert k_cache.shape[1:] == v_cache.shape[1:] and cos_sin_cache.shape[1] == d
    assert k_cache.stride()[1:] == v_cache.stride()[1:] == (d, 1)
    assert cos_sin_cache.dtype == torch.float32 and cos_sin_cache.is_contiguous()
    assert positions.shape == (rows,)
    loc_2d = loc_2d.to(torch.int64).contiguous()
    commit_lens = commit_lens.to(torch.int32).contiguous()
    positions = positions.to(torch.int64).contiguous()
    if rows:
        _ctx_kv_store[(rows, heads)](
            kv,
            positions,
            k_weight,
            cos_sin_cache,
            k_cache,
            v_cache,
            loc_2d,
            commit_lens,
            kv.stride(0),
            k_cache.stride(0),
            v_cache.stride(0),
            eps,
            loc_2d.shape[1],
            heads * d,
            d,
            num_warps=1,
        )
