"""Fused ratio-2 decode pair-pooling with bitwise parity to torch pool_pairs.

Softmax follows torch's operation order with correctly rounded exp and division.
The weighted products must round separately before summation; FMA contraction
would change the latent and can change the indexer's top-k selection.
"""

from typing import Tuple

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from sglang.srt.utils import is_hip

# libdevice.exp / div_rn do not lower on HIP; tl.exp and `/` are exact against torch on gfx950
_USE_LIBDEVICE = not is_hip()


@triton.jit
def _pair_pool_decode_kernel(
    kv_ptr,  # [n, D] fp32
    score_ptr,  # [n, D] fp32
    pos_ptr,  # [n] int64
    raw_out_loc_ptr,  # [n] int32/int64
    out_loc_ptr,  # [n] int32/int64
    req_ptr,  # [n] int64
    state_kv_ptr,  # [R, D] fp32, in/out
    state_score_ptr,  # [R, D] fp32, in/out
    pooled_ptr,  # [n, D] fp32, out
    group_pos_ptr,  # [n] int64, out
    slots_ptr,  # [n] int64, out
    pad_row,
    RING_SIZE: tl.constexpr,
    STATE_KV_STRIDE: tl.constexpr,
    STATE_SCORE_STRIDE: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    LIBDEVICE: tl.constexpr,
):
    row = tl.program_id(0)

    pos = tl.load(pos_ptr + row)
    raw_loc = tl.load(raw_out_loc_ptr + row)
    out_loc = tl.load(out_loc_ptr + row)
    req = tl.load(req_ptr + row)

    # `pos % 2 == 1`, and the padded-graph-row reroute: a row whose raw location
    # is the reserved slot 0 carries req_pool_idx 0 -- possibly a live request --
    # so its pair state goes to the spare row instead.
    odd = (pos % 2) == 1
    if RING_SIZE:
        r = tl.where(
            (raw_loc == 0) | (pos == 0),
            pad_row,
            req * RING_SIZE + (pos - 1) % RING_SIZE,
        )
    else:
        r = tl.where(raw_loc == 0, pad_row, req)

    offs = tl.arange(0, BLOCK_D)
    mask = offs < D

    kv = tl.load(kv_ptr + row * D + offs, mask=mask, other=0.0)
    score = tl.load(score_ptr + row * D + offs, mask=mask, other=0.0)
    p_kv = tl.load(state_kv_ptr + r * STATE_KV_STRIDE + offs, mask=mask, other=0.0)
    p_score = tl.load(
        state_score_ptr + r * STATE_SCORE_STRIDE + offs, mask=mask, other=0.0
    )

    if RING_SIZE:
        # Each live request has one decode row. Pad rows never modify the ring.
        write_row = req * RING_SIZE + pos % RING_SIZE
        tl.store(
            state_kv_ptr + write_row * STATE_KV_STRIDE + offs,
            kv,
            mask=mask & (raw_loc != 0),
        )
        tl.store(
            state_score_ptr + write_row * STATE_SCORE_STRIDE + offs,
            score,
            mask=mask & (raw_loc != 0),
        )
    else:
        tl.store(
            state_kv_ptr + r * STATE_KV_STRIDE + offs,
            tl.where(odd, p_kv, kv),
            mask=mask,
        )
        tl.store(
            state_score_ptr + r * STATE_SCORE_STRIDE + offs,
            tl.where(odd, p_score, score),
            mask=mask,
        )

    # torch's pair-axis softmax order with an exact exp: libdevice on CUDA, tl.exp on gfx950
    m = tl.maximum(p_score, score)
    if LIBDEVICE:
        e0 = libdevice.exp(p_score - m)
        e1 = libdevice.exp(score - m)
    else:
        e0 = tl.exp(p_score - m)
        e1 = tl.exp(score - m)
    denom = e0 + e1
    # The + 0.0 prevents FMA contraction: torch rounds both products before summing.
    # torch's correctly rounded division: div_rn on CUDA, / on gfx950
    if LIBDEVICE:
        t0 = p_kv * libdevice.div_rn(e0, denom)
        t1 = kv * libdevice.div_rn(e1, denom)
    else:
        t0 = p_kv * (e0 / denom)
        t1 = kv * (e1 / denom)
    t0 = t0 + 0.0
    t1 = t1 + 0.0
    pooled = t0 + t1
    tl.store(pooled_ptr + row * D + offs, pooled, mask=mask)

    # One program per row, so these are written exactly once each; no guard.
    tl.store(group_pos_ptr + row, tl.where(odd, pos - 1, pos))
    tl.store(slots_ptr + row, tl.where(out_loc >= 0, out_loc, 0))


def pair_pool_decode(
    kv: torch.Tensor,
    score: torch.Tensor,
    pos: torch.Tensor,
    raw_out_loc: torch.Tensor,
    out_loc: torch.Tensor,
    req: torch.Tensor,
    state_kv: torch.Tensor,
    state_score: torch.Tensor,
    pad_row: int,
    *,
    ring_size: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One launch for the ratio-2 decode epilogue.

    Returns `(pooled, group_pos, slots)` and updates `state_kv` / `state_score`
    in place. With ring_size > 0, the state halves may be views of an interleaved
    CompressStatePool ring; pad_row is its sentinel row and stays untouched.
    The default retains the legacy single-row-per-request state contract.
    """
    assert kv.is_contiguous() and score.is_contiguous()
    assert state_kv.stride(1) == state_score.stride(1) == 1
    assert kv.dtype == torch.float32 and score.dtype == torch.float32
    n, D = kv.shape
    pooled = torch.empty_like(kv)
    group_pos = torch.empty_like(pos)
    slots = torch.empty(n, dtype=out_loc.dtype, device=out_loc.device)
    _pair_pool_decode_kernel[(n,)](
        kv,
        score,
        pos,
        raw_out_loc,
        out_loc,
        req,
        state_kv,
        state_score,
        pooled,
        group_pos,
        slots,
        pad_row,
        RING_SIZE=ring_size,
        STATE_KV_STRIDE=state_kv.stride(0),
        STATE_SCORE_STRIDE=state_score.stride(0),
        D=D,
        BLOCK_D=triton.next_power_of_2(D),
        LIBDEVICE=_USE_LIBDEVICE,
        num_warps=4,
    )
    return pooled, group_pos, slots
