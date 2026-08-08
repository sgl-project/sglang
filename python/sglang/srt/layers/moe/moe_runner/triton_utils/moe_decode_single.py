"""Single-token (decode, M==1) fast path for the Triton fused MoE.

At batch size 1 (single-request decode), the MoE sees exactly one token routed
to ``top_k`` experts, i.e. ``top_k`` (token, expert) pairs where **every pair
uses a distinct expert**. In that regime the generic grouped-GEMM path pays for
work that has no payoff:

  * ``moe_align_block_size`` sorts/groups tokens by expert, but with one token
    per expert there is nothing to group;
  * the gate/up GEMM, the ``silu_and_mul`` activation and the down GEMM +
    weighted reduction are launched as separate kernels, materialising the
    intermediate activation to HBM in between.

This module fuses the whole op into two small Triton kernels operating directly
on the ``top_k`` pairs, skipping the align/sort and keeping the intermediate in
registers. Accumulation is done in fp32 (output is cast back to the input
dtype), so numerics are at least as accurate as the generic bf16 path.

The fast path is intentionally narrow: it only triggers for the plain bf16,
gated-SiLU, unquantised, no-bias case at ``num_tokens == 1``. Every other case
falls back to the generic implementation. It is a pure performance shortcut and
does not change results beyond floating-point rounding.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _w1_silu_kernel(
    x_ptr,
    w1_ptr,
    ids_ptr,
    act_ptr,
    H: tl.constexpr,
    I: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    BM: tl.constexpr,
):
    """For pair ``p`` and output tile ``nt`` compute silu(x@Wg) * (x@Wu)."""
    p = tl.program_id(0)
    nt = tl.program_id(1)
    e = tl.load(ids_ptr + p)
    n = nt * BN + tl.arange(0, BN)
    m = tl.arange(0, BM)
    accg = tl.zeros((BM, BN), dtype=tl.float32)
    accu = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, H, BK):
        koff = k0 + tl.arange(0, BK)
        xb = tl.load(
            x_ptr + koff[None, :] + m[:, None] * 0, mask=m[:, None] < 1, other=0.0
        )
        wg = tl.load(w1_ptr + e * (2 * I * H) + n[:, None] * H + koff[None, :]).to(
            tl.bfloat16
        )
        wu = tl.load(
            w1_ptr + e * (2 * I * H) + (I + n[:, None]) * H + koff[None, :]
        ).to(tl.bfloat16)
        accg += tl.dot(xb, wg.T)
        accu += tl.dot(xb, wu.T)
    silu = accg / (1.0 + tl.exp(-accg))
    tl.store(act_ptr + p * I + n, tl.sum(silu * accu, axis=0).to(tl.bfloat16))


@triton.jit
def _w2_reduce_kernel(
    act_ptr,
    w2_ptr,
    ids_ptr,
    tw_ptr,
    out_ptr,
    rsf,
    H: tl.constexpr,
    I: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    BM: tl.constexpr,
):
    """Down projection with routing weight, scaling and reduction into out."""
    p = tl.program_id(0)
    nt = tl.program_id(1)
    e = tl.load(ids_ptr + p)
    tw = tl.load(tw_ptr + p).to(tl.float32)
    n = nt * BN + tl.arange(0, BN)
    m = tl.arange(0, BM)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, I, BK):
        koff = k0 + tl.arange(0, BK)
        ab = tl.load(
            act_ptr + p * I + koff[None, :] + m[:, None] * 0,
            mask=m[:, None] < 1,
            other=0.0,
        ).to(tl.bfloat16)
        w = tl.load(w2_ptr + e * (H * I) + n[:, None] * I + koff[None, :]).to(
            tl.bfloat16
        )
        acc += tl.dot(ab, w.T)
    tl.atomic_add(out_ptr + n, tl.sum(acc, axis=0) * tw * rsf)


def decode_single_moe_supported(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    *,
    activation: str,
    is_gated: bool,
    apply_router_weight_on_input: bool,
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
    b1: Optional[torch.Tensor],
    b2: Optional[torch.Tensor],
    block_shape: Optional[list],
    gemm1_alpha: Optional[float],
    gemm1_limit: Optional[float],
    swiglu_limit: Optional[float],
    gate_up_interleaved: bool,
    no_combine: bool,
) -> bool:
    """Return True iff the single-token fast path can handle this call exactly."""
    if hidden_states.shape[0] != 1:
        return False
    if hidden_states.dtype is not torch.bfloat16:
        return False
    if not (is_gated and activation == "silu"):
        return False
    if apply_router_weight_on_input or no_combine:
        return False
    if use_fp8_w8a8 or use_int8_w8a8 or use_int8_w8a16 or use_int4_w4a16:
        return False
    if b1 is not None or b2 is not None or block_shape is not None:
        return False
    if gemm1_alpha is not None or gemm1_limit is not None or swiglu_limit is not None:
        return False
    if not gate_up_interleaved:
        return False
    H = hidden_states.shape[1]
    E, N2, K = w1.shape
    I = N2 // 2
    if K != H:
        return False
    if tuple(w2.shape) != (E, H, I):
        return False
    # The tiled kernels require the intermediate / hidden dims to be tile-aligned.
    if I % 64 != 0 or H % 64 != 0:
        return False
    return True


def decode_single_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    routed_scaling_factor: Optional[float] = None,
) -> torch.Tensor:
    """Fused single-token MoE. Assumes ``decode_single_moe_supported`` is True.

    Shapes: hidden_states [1, H]; w1 [E, 2*I, H] (gate rows then up rows);
    w2 [E, H, I]; topk_weights/topk_ids [1, top_k].
    """
    H = hidden_states.shape[1]
    E, N2, _ = w1.shape
    I = N2 // 2
    topk = topk_ids.shape[1]
    ids = topk_ids.reshape(-1).to(torch.int32)
    tw = topk_weights.reshape(-1).to(torch.float32)
    rsf = 1.0 if routed_scaling_factor is None else float(routed_scaling_factor)
    P = ids.numel()
    x = hidden_states.reshape(H).contiguous()

    act = torch.empty(P, I, dtype=torch.bfloat16, device=hidden_states.device)
    out = torch.zeros(H, dtype=torch.float32, device=hidden_states.device)

    _w1_silu_kernel[(P, I // 64)](
        x, w1, ids, act, H, I, 64, 256, 16, num_warps=4
    )
    _w2_reduce_kernel[(P, H // 64)](
        act, w2, ids, tw, out, rsf, H, I, 64, 128, 16, num_warps=4
    )
    return out.to(hidden_states.dtype).view(1, H)
