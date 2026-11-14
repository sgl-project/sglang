"""Kimi-K3 routed experts on gfx950: MXFP8 activations x MXFP4 weights (A8W4), the precision of
AITER's ``AITER_SITUV2_A8W4`` path, on AITER's preshuffled weights.

Pipeline: AITER moe_sorting -> AITER per-1x32 MXFP8 quant (natural scales, no scale sort) ->
gateup (SiTUv2 + MXFP8 requant fused) -> down (bf16 per sorted row) -> fp32 top-k reduce.
Kernels: ``python/sglang/kernels/jit/csrc/moe/k3_a8w4_gfx950.cuh``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# Blocks of one expert per down work item, so a hot expert's blocks spread over workgroups.
_DOWN_SEG_BLOCKS = 8
# The down's y column chunk: 8 waves x 64 columns a workgroup (DownWsCfg::BN in the .cuh).
_Y_CHUNK_COLS = 512


@cache_once
def _module() -> Module:
    return load_jit(
        "k3_a8w4_moe_gfx950",
        cuda_files=["moe/k3_a8w4_gfx950.cuh"],
        cuda_wrappers=[
            ("gateup", "K3A8W4MoE::gateup"),
            ("down", "K3A8W4MoE::down"),
            ("reduce", "K3A8W4MoE::reduce"),
        ],
        extra_cuda_cflags=["-O3"],
    )


def pick_block_m(num_tokens: int, top_k: int, num_experts: int) -> int:
    # MI350X block_m sweeps (uniform routing): 32 is best near 1K tokens (~18 rows an expert),
    # 64 from 2K (~37) up; 16 only at decode sizes, where neither beats AITER.
    rows_per_expert = num_tokens * top_k / num_experts
    if rows_per_expert <= 4:
        return 16
    if rows_per_expert <= 32:
        return 32
    return 64


def _sort_routes(
    topk_ids, topk_weights, *, num_experts: int, hidden: int, block_m: int
):
    """(sorted_ids, expert_ids, num_valid) in AITER's moe_sorting format, padded per block_m."""
    from aiter.fused_moe import moe_sorting

    sorted_ids, _, expert_ids, num_valid, _ = moe_sorting(
        topk_ids,
        topk_weights,
        num_experts,
        hidden,
        torch.bfloat16,
        block_size=block_m,
        accumulate=False,
    )
    return sorted_ids, expert_ids, num_valid


def quant_mxfp8(x: torch.Tensor):
    """[T, K] bf16 -> ([T, K] e4m3, [T, K/32] e8m0) as uint8, AITER's A8W4 activation quant."""
    from aiter import dtypes
    from aiter.ops.quant import per_1x32_mx_quant_hip

    q, s = per_1x32_mx_quant_hip(x, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0)
    return q.view(torch.uint8), s.view(torch.uint8)


def _gateup(
    a,
    a_scale,
    w13,
    w13_scale,
    sorted_ids,
    expert_ids,
    num_valid,
    *,
    block_m: int,
    beta: float,
    linear_beta: float,
):
    """-> (inter [rows, I] e4m3, inter_scale [rows, I/32] e8m0), rows in sorted order."""
    inter_dim = w13.shape[1] // 2
    rows = expert_ids.shape[0] * block_m
    inter = torch.empty(rows, inter_dim, dtype=torch.uint8, device=a.device)
    inter_scale = torch.empty(rows, inter_dim // 32, dtype=torch.uint8, device=a.device)
    _module().gateup(
        a,
        a_scale,
        w13.view(torch.uint8),
        w13_scale.view(torch.uint8),
        sorted_ids,
        expert_ids,
        num_valid,
        inter,
        inter_scale,
        block_m,
        float(beta),
        float(linear_beta),
    )
    return inter, inter_scale


def _down(
    inter,
    inter_scale,
    w2,
    w2_scale,
    sorted_ids,
    expert_ids,
    num_valid,
    topk_weights,
    *,
    block_m: int,
    seg_blocks: int,
):
    """-> (y, pos): y bf16 rows (unweighted) at their sorted row, laid out as
    [N / _Y_CHUNK_COLS][rows][_Y_CHUNK_COLS]; pos [T * top_k] int32 the sorted row of (token, slot)."""
    num_tokens, top_k = topk_weights.shape
    n, dev = w2.shape[1], inter.device
    y = torch.empty(sorted_ids.shape[0], n, dtype=torch.bfloat16, device=dev)
    pos = torch.empty(num_tokens * top_k, dtype=torch.int32, device=dev)
    # down_items_kernel's work list: a header and at most max_blocks / seg_blocks + E items, int4 each
    max_items = -(-expert_ids.shape[0] // seg_blocks) + w2.shape[0]
    work = torch.empty(4 * (1 + max_items), dtype=torch.int32, device=dev)
    _module().down(
        inter,
        inter_scale,
        w2.view(torch.uint8),
        w2_scale.view(torch.uint8),
        sorted_ids,
        expert_ids,
        num_valid,
        y,
        pos,
        topk_weights,
        work,
        block_m,
        seg_blocks,
    )
    return y, pos


def _k3_a8w4_moe(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    *,
    beta: float,
    linear_beta: float,
    out: Optional[torch.Tensor],
    block_m: int,
    seg_blocks: int,
    a_mx: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    num_tokens, hidden = hidden_states.shape
    num_experts = w13.shape[0]
    if out is None:
        out = torch.empty(
            num_tokens, w2.shape[1], dtype=torch.bfloat16, device=hidden_states.device
        )
    if num_tokens == 0:
        return out
    sorted_ids, expert_ids, num_valid = _sort_routes(
        topk_ids, topk_weights, num_experts=num_experts, hidden=hidden, block_m=block_m
    )
    a, a_scale = quant_mxfp8(hidden_states) if a_mx is None else a_mx
    inter, inter_scale = _gateup(
        a,
        a_scale,
        w13,
        w13_scale,
        sorted_ids,
        expert_ids,
        num_valid,
        block_m=block_m,
        beta=beta,
        linear_beta=linear_beta,
    )
    y, pos = _down(
        inter,
        inter_scale,
        w2,
        w2_scale,
        sorted_ids,
        expert_ids,
        num_valid,
        topk_weights,
        block_m=block_m,
        seg_blocks=seg_blocks,
    )
    _module().reduce(y, topk_weights, pos, out, _Y_CHUNK_COLS)
    return out


def k3_a8w4_moe(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    *,
    beta: float,
    linear_beta: float,
    out: Optional[torch.Tensor] = None,
    a_mx: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    """Routed experts of one K3 MoE layer.

    hidden_states [T, K] bf16; topk_ids [T, topk] int32; topk_weights [T, topk] fp32;
    a_mx: hidden_states already through quant_mxfp8 (hidden_states then only gives the shape);
    w13 [E, 2I, K/2] / w13_scale [E * 2I, K/32] in AITER's gate/up-interleaved preshuffle
    (``shuffle_weight_a16w4(w, 16, True)`` / ``shuffle_scale_a16w4(s, E, True)``);
    w2 [E, N, I/2] / w2_scale [E * N, pad8(I/32)] likewise with ``gate_up=False``.
    out [T, N] bf16 may be a row-strided view.
    """
    block_m = pick_block_m(hidden_states.shape[0], topk_ids.shape[1], w13.shape[0])
    return _k3_a8w4_moe(
        hidden_states,
        topk_ids,
        topk_weights,
        w13,
        w13_scale,
        w2,
        w2_scale,
        beta=beta,
        linear_beta=linear_beta,
        out=out,
        block_m=block_m,
        seg_blocks=_DOWN_SEG_BLOCKS,
        a_mx=a_mx,
    )
