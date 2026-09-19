import torch
import triton
import triton.language as tl

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.kernels.kda_kernels import _cuda_source


@cache_once
def _producer_module():
    return load_jit(
        "qwen4_ple_producer",
        cuda_files=[_cuda_source("qwen4_ple_producer.cuh")],
        cuda_wrappers=[("produce", "qwen4_ple::produce")],
    )


@triton.jit
def _qwen4_ple_conv_residual(
    X,
    G,
    H,
    Weight,
    State,
    Indices,
    Valid,
    Residual,
    Cache,
    Track,
    TrackMask,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    IS: tl.constexpr,
    C0: tl.constexpr,
    C1: tl.constexpr,
    C2: tl.constexpr,
    C3: tl.constexpr,
    TS: tl.constexpr,
    TMS: tl.constexpr,
    W: tl.constexpr,
    HAS_CACHE: tl.constexpr,
    HAS_TRACK: tl.constexpr,
    HAS_TRACK_MASK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    req = tl.program_id(0)
    step = tl.program_id(1)
    ch = tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)
    col = tl.arange(0, 16)
    slot = tl.load(Indices + req * IS)
    hist = (
        tl.load(
            State + slot * S0 + ch[:, None] * S1 + col[None, :] * S2,
            (ch[:, None] < 10240) & (col[None, :] < 9),
            other=0,
        )
        .to(tl.bfloat16)
        .to(tl.float32)
    )
    acc = tl.full((BLOCK,), 0, tl.float32)
    for j in tl.static_range(4):
        c = step + j * 3
        gather_c = tl.full((BLOCK, 1), tl.minimum(c, 8), tl.int32)
        old = tl.gather(hist, gather_c, 1).reshape((BLOCK,))
        new = tl.load(
            X + (req * W + c - 9) * 10240 + ch, (ch < 10240) & (c >= 9), other=0
        ).to(tl.float32)
        v = tl.where(c < 9, old, new)
        w = tl.load(Weight + ch * 4 + j, ch < 10240, other=0).to(tl.float32)
        acc = tl.fma(v, w, acc)
    conv = acc.to(tl.bfloat16).to(tl.float32)
    act = (conv * tl.sigmoid(conv)).to(tl.bfloat16).to(tl.float32)
    token = req * W + step
    gated = tl.load(G + token * 10240 + ch, ch < 10240, other=0).to(tl.float32)
    ple = (gated + act).to(tl.bfloat16).to(tl.float32)
    if W == 4:
        valid = tl.load(Valid + token)
        ple = tl.where(valid, ple, 0)
    else:
        valid = True
    hidden = tl.load(H + token * 10240 + ch, ch < 10240, other=0).to(tl.float32)
    tl.store(Residual + token * 10240 + ch, hidden + ple, ch < 10240)
    src_col = step + 1 + col
    gather_col = tl.broadcast_to(tl.minimum(src_col, 8)[None, :], (BLOCK, 16))
    old = tl.gather(hist, gather_col, 1)
    new = tl.load(
        X + (req * W + src_col[None, :] - 9) * 10240 + ch[:, None],
        (ch[:, None] < 10240) & (src_col[None, :] >= 9) & (col[None, :] < 9),
        other=0,
    ).to(tl.float32)
    next_state = tl.where(valid, tl.where(src_col[None, :] < 9, old, new), 0)
    if HAS_CACHE:
        tl.store(
            Cache + req * C0 + step * C1 + ch[:, None] * C2 + col[None, :] * C3,
            next_state,
            (ch[:, None] < 10240) & (col[None, :] < 9),
        )
    if W == 1:
        tl.debug_barrier()
        tl.store(
            State + slot * S0 + ch[:, None] * S1 + col[None, :] * S2,
            next_state,
            (slot != 0) & (ch[:, None] < 10240) & (col[None, :] < 9),
        )

        if HAS_TRACK:
            track_slot = tl.load(Track + req * TS).to(tl.int64)
            if HAS_TRACK_MASK:
                track_slot = tl.where(tl.load(TrackMask + req * TMS), track_slot, 0)
            tl.store(
                State + track_slot * S0 + ch[:, None] * S1 + col[None, :] * S2,
                next_state,
                (track_slot != 0) & (ch[:, None] < 10240) & (col[None, :] < 9),
            )


def fused_ple(
    key,
    query,
    value,
    hidden,
    kw,
    qw,
    cw,
    weight,
    state,
    indices,
    valid,
    width,
    intermediate=None,
    track_indices=None,
    track_mask=None,
):
    x, gated = torch.empty_like(key), torch.empty_like(key)
    output = torch.empty_like(key)
    _producer_module().produce(key, query, value, kw, qw, cw, x, gated)
    block = 128
    _qwen4_ple_conv_residual[(indices.numel(), width, triton.cdiv(10240, block))](
        x,
        gated,
        hidden,
        weight,
        state,
        indices,
        valid,
        output,
        intermediate if intermediate is not None else state,
        track_indices if track_indices is not None else indices,
        track_mask if track_mask is not None else valid,
        *state.stride(),
        indices.stride(0),
        *(intermediate.stride() if intermediate is not None else (0, 0, 0, 0)),
        track_indices.stride(0) if track_indices is not None else 1,
        track_mask.stride(0) if track_mask is not None else 1,
        width,
        intermediate is not None,
        track_indices is not None,
        track_mask is not None,
        block,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output
