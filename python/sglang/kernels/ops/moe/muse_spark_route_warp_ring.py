"""Physical expert id = holder rank * 2L + slot; slots [L, 2L) of rank r hold rank (r + 1) % R's home experts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# Static shared memory bounds the warps per CTA; a portable cluster has 8 CTAs.
MAX_WARPS_PER_CTA = 8
MAX_CLUSTER_CTAS = 8
MAX_FUSED_TOKENS = MAX_WARPS_PER_CTA * MAX_CLUSTER_CTAS


@cache_once
def _jit_route_warp_ring_module(
    num_experts: int, topk: int, num_ranks: int, chunks_per_home: int
) -> Module:
    args = make_cpp_args(
        num_experts,
        topk,
        num_ranks,
        chunks_per_home,
        MAX_WARPS_PER_CTA,
        is_arch_support_pdl(),
    )
    return load_jit(
        "moe_route_warp_ring",
        *args,
        cuda_files=["moe/route_warp_ring.cuh"],
        cuda_wrappers=[
            ("run", f"moe::route_warp_ring::RouteWarpRingKernel<{args}>::run")
        ],
    )


def launch_shape(num_tokens: int) -> Optional[Tuple[int, int]]:
    """Depends on ``num_tokens`` only, so every rank launches the same shape."""
    if not 1 <= num_tokens <= MAX_FUSED_TOKENS:
        return None
    if num_tokens <= MAX_WARPS_PER_CTA:
        return num_tokens, 1
    ctas = -(-num_tokens // MAX_WARPS_PER_CTA)
    warps = -(-num_tokens // ctas)
    return warps, -(-num_tokens // warps)


def muse_spark_route_warp_ring(
    scores: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    route_scale: float,
    *,
    num_ranks: int,
    home: int,
    router_output_multiplier: Optional[float] = None,
    division_eps: Optional[float] = None,
    num_valid_tokens: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Needs SM90+ and a batch that ``launch_shape`` accepts."""
    num_tokens, num_experts = scores.shape
    shape = launch_shape(num_tokens)
    assert shape is not None and num_experts == num_ranks * home and home % 32 == 0
    module = _jit_route_warp_ring_module(num_experts, topk, num_ranks, home // 32)
    out_w = torch.empty((num_tokens, topk), dtype=torch.float32, device=scores.device)
    out_i = torch.empty((num_tokens, topk), dtype=torch.int32, device=scores.device)
    out_packed = torch.empty_like(out_i)
    module.run(
        scores,
        bias,
        out_w,
        out_i,
        out_packed,
        num_valid_tokens,
        float(route_scale),
        float(division_eps if division_eps is not None else 0.0),
        float(
            router_output_multiplier if router_output_multiplier is not None else 1.0
        ),
        shape[0],
        shape[1],
    )
    return out_w, out_i, out_packed


@triton.jit(do_not_specialize=["num_entries"])
def _ring_pick_kernel(
    ids_ptr,
    packed_ptr,
    out_packed_ptr,
    scratch_ptr,
    num_entries,
    NUM_RANKS: tl.constexpr,
    HOME: tl.constexpr,
    RANKS_PAD: tl.constexpr,
    HOME_PAD: tl.constexpr,
    FLOW_PAD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    NUM_LOGICAL = NUM_RANKS * HOME
    r_idx = tl.arange(0, RANKS_PAD)
    j_idx = tl.arange(0, HOME_PAD)
    cell = r_idx[:, None] * HOME + j_idx[None, :]
    cell_mask = (r_idx[:, None] < NUM_RANKS) & (j_idx[None, :] < HOME)
    offs = tl.arange(0, BLOCK)

    tl.store(
        scratch_ptr + cell,
        tl.zeros([RANKS_PAD, HOME_PAD], dtype=tl.int32),
        mask=cell_mask,
    )
    tl.debug_barrier()
    for start in range(0, num_entries, BLOCK):
        idx = start + offs
        m = idx < num_entries
        e = tl.load(ids_ptr + idx, mask=m, other=-1)
        valid = m & (e >= 0) & (e < NUM_LOGICAL)
        tl.store(scratch_ptr + e, tl.full([BLOCK], 1, dtype=tl.int32), mask=valid)
    tl.debug_barrier()
    act = tl.load(scratch_ptr + cell, mask=cell_mask, other=0, volatile=True)
    k = tl.sum(act, axis=1)

    # Same (T, y0) search and tie-break as muse_ring_pick.h, over the full grid.
    zeros2 = tl.zeros([FLOW_PAD, FLOW_PAD], dtype=tl.int32)
    t2 = tl.arange(0, FLOW_PAD)[:, None] + zeros2
    y02 = tl.arange(0, FLOW_PAD)[None, :] + zeros2
    k0 = tl.sum(tl.where(r_idx == 0, k, 0), axis=0)
    ok = (y02 <= k0) & (t2 <= HOME)
    prev = y02
    for rr in tl.static_range(NUM_RANKS - 1):
        kr = tl.sum(tl.where(r_idx == (NUM_RANKS - 1 - rr), k, 0), axis=0)
        need = kr + prev - t2
        need = tl.where(need > 0, need, 0)
        ok = ok & (need <= kr)
        prev = need
    ok = ok & ((k0 + prev - t2) <= y02)
    t_best = tl.min(tl.min(tl.where(ok, t2, 1048576), axis=1), axis=0)
    y0_best = tl.min(
        tl.min(tl.where(ok & (t2 == t_best), y02, 1048576), axis=1), axis=0
    )
    y_vec = tl.where(r_idx == 0, y0_best, 0)
    prev_s = y0_best
    for rr in tl.static_range(NUM_RANKS - 1):
        kr = tl.sum(tl.where(r_idx == (NUM_RANKS - 1 - rr), k, 0), axis=0)
        need_s = kr + prev_s - t_best
        need_s = tl.where(need_s > 0, need_s, 0)
        y_vec = tl.where(r_idx == (NUM_RANKS - 1 - rr), need_s, y_vec)
        prev_s = need_s

    excl = tl.cumsum(act, axis=1) - act
    keep = k - y_vec
    moved = (act > 0) & (excl >= keep[:, None])
    buddy_holder = (r_idx + NUM_RANKS - 1) % NUM_RANKS
    home_phys = r_idx[:, None] * (2 * HOME) + j_idx[None, :]
    buddy_phys = buddy_holder[:, None] * (2 * HOME) + HOME + j_idx[None, :]
    phys = tl.where(moved, buddy_phys, home_phys)
    tl.store(scratch_ptr + NUM_LOGICAL + cell, phys, mask=cell_mask)
    tl.debug_barrier()

    for start in range(0, num_entries, BLOCK):
        idx = start + offs
        m = idx < num_entries
        e = tl.load(ids_ptr + idx, mask=m, other=-1)
        valid = m & (e >= 0) & (e < NUM_LOGICAL)
        p = tl.load(scratch_ptr + NUM_LOGICAL + e, mask=valid, other=0, volatile=True)
        pk = tl.load(packed_ptr + idx, mask=m, other=0)
        out_pk = tl.where(valid, (p << 16) | (pk & 0xFFFF), pk)
        tl.store(out_packed_ptr + idx, out_pk, mask=m)


def _next_pow2(x: int) -> int:
    return 1 << max(0, (x - 1).bit_length())


def muse_spark_ring_pick(
    topk_ids: torch.Tensor,
    packed_topk_ids: torch.Tensor,
    scratch: torch.Tensor,
    *,
    num_ranks: int,
    home: int,
) -> torch.Tensor:
    """``scratch`` is int32 [2 * num_ranks * home]; one CTA, no host sync."""
    topk_ids = topk_ids.to(torch.int32).contiguous()
    packed_topk_ids = packed_topk_ids.contiguous()
    out_packed = torch.empty_like(packed_topk_ids)
    _ring_pick_kernel[(1,)](
        topk_ids,
        packed_topk_ids,
        out_packed,
        scratch,
        topk_ids.numel(),
        NUM_RANKS=num_ranks,
        HOME=home,
        RANKS_PAD=_next_pow2(num_ranks),
        HOME_PAD=_next_pow2(home),
        FLOW_PAD=_next_pow2(home + 1),
        BLOCK=256,
        num_warps=4,
    )
    return out_packed
