import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _mix_rmsnorm(
    X,
    ALPHA,
    WEIGHT,
    OUT,
    ROW_STRIDE: tl.constexpr,
    TAP_STRIDE: tl.constexpr,
    HIDDEN_STRIDE: tl.constexpr,
    TAPS: tl.constexpr,
    HIDDEN: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    h = tl.arange(0, BLOCK)
    mixed = tl.full((BLOCK,), 0, tl.float32)
    for tap in tl.static_range(TAPS):
        x = tl.load(
            X + row * ROW_STRIDE + tap * TAP_STRIDE + h * HIDDEN_STRIDE,
            h < HIDDEN,
            0,
        ).to(tl.float32)
        alpha = tl.load(ALPHA + tap).to(tl.float32)
        mixed += x * alpha
    inv_rms = tl.rsqrt(tl.sum(mixed * mixed, 0) / HIDDEN + EPS)
    weight = tl.load(WEIGHT + h, h < HIDDEN, 0).to(tl.float32)
    tl.store(OUT + row * HIDDEN + h, mixed * inv_rms * weight, h < HIDDEN)


def mix_rmsnorm(
    targets: torch.Tensor, alpha: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    rows, taps, hidden = targets.shape
    assert alpha.shape == (taps,) and alpha.is_contiguous()
    assert weight.shape == (hidden,) and weight.is_contiguous()
    out = targets.new_empty((rows, hidden))
    if rows:
        _mix_rmsnorm[(rows,)](
            targets,
            alpha,
            weight,
            out,
            *targets.stride(),
            taps,
            hidden,
            eps,
            triton.next_power_of_2(hidden),
            num_warps=8,
        )
    return out


@triton.jit
def _bf16(x):
    return x.to(tl.bfloat16).to(tl.float32)


@triton.jit
def _rnn_gates(
    Proj, State, NewState, Latent, SP, SS, R: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < R
    gate_raw = tl.load(Proj + row * SP + cols, mask, other=0).to(tl.float32)
    cand_raw = tl.load(Proj + row * SP + R + cols, mask, other=0).to(tl.float32)
    out_raw = tl.load(Proj + row * SP + 2 * R + cols, mask, other=0).to(tl.float32)
    state = tl.load(State + row * SS + cols, mask, other=0).to(tl.float32)
    # Each op rounds to BF16 like the eager torch ops it replaces.
    gate = _bf16(tl.div_rn(1.0, 1.0 + libdevice.exp(-gate_raw)))
    kept = _bf16(gate * state)
    blend = _bf16(_bf16(1.0 - gate) * _bf16(libdevice.tanh(cand_raw)))
    tl.store(NewState + row * R + cols, kept + blend, mask)
    tl.store(Latent + row * R + cols, libdevice.tanh(out_raw), mask)


def rnn_gates(proj: torch.Tensor, state: torch.Tensor):
    """(state', latent) of Inkling2MTPHead._rnn_step from the rnn_proj output."""
    rows, rank = state.shape
    assert proj.shape == (rows, 3 * rank) and proj.stride(-1) == state.stride(-1) == 1
    new_state, latent = state.new_empty((rows, rank)), state.new_empty((rows, rank))
    if rows:
        _rnn_gates[(rows,)](
            proj,
            state,
            new_state,
            latent,
            proj.stride(0),
            state.stride(0),
            rank,
            triton.next_power_of_2(rank),
            num_warps=1,
            enable_fp_fusion=False,
        )
    return new_state, latent


@triton.jit
def _rnn_inputs(
    State,
    Tokens,
    Embedding,
    Hidden,
    FRnn,
    Out,
    token_stride,
    hidden_stride,
    RANK: tl.constexpr,
    HIDDEN: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    width: tl.constexpr = 2 * RANK + 2 * HIDDEN
    r = tl.arange(0, BLOCK_R)
    h = tl.arange(0, BLOCK_H)
    token = tl.load(Tokens + row * token_stride).to(tl.int64)
    out = Out + row * width
    tl.store(out + r, tl.load(State + row * RANK + r, r < RANK), r < RANK)
    embedding = tl.load(Embedding + token * RANK + r, r < RANK)
    tl.store(out + RANK + r, embedding.to(Out.dtype.element_ty), r < RANK)
    hidden = tl.load(Hidden + row * hidden_stride + h, h < HIDDEN)
    tl.store(out + 2 * RANK + h, hidden, h < HIDDEN)
    tl.store(
        out + 2 * RANK + HIDDEN + h,
        tl.load(FRnn + row * HIDDEN + h, h < HIDDEN),
        h < HIDDEN,
    )


def rnn_inputs(state, tokens, embedding, hidden, f_rnn):
    """torch.cat([state, embedding[tokens], hidden, f_rnn], -1) in one kernel."""
    rows, rank = state.shape
    width = hidden.shape[1]
    assert state.is_contiguous() and f_rnn.is_contiguous() and hidden.stride(1) == 1
    assert state.dtype == hidden.dtype == f_rnn.dtype and embedding.is_contiguous()
    out = state.new_empty((rows, 2 * rank + 2 * width))
    if rows:
        _rnn_inputs[(rows,)](
            state,
            tokens,
            embedding,
            hidden,
            f_rnn,
            out,
            tokens.stride(0),
            hidden.stride(0),
            rank,
            width,
            triton.next_power_of_2(rank),
            triton.next_power_of_2(width),
        )
    return out
