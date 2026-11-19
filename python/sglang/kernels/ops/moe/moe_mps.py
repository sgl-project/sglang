"""Unquantized MoE expert compute for Apple MPS.

Small batches (decode) run two Metal GEMV kernels (``moe.metal``, compiled at
runtime with ``torch.mps.compile_shader``) that read the selected experts'
weights in place; larger batches sort the (token, expert) pairs by expert and
run one torch GEMM pair per active expert.
"""

from __future__ import annotations

import functools
from pathlib import Path

import torch
import torch.nn.functional as F

# Up to this many (token, expert) pairs take the Metal GEMV path.
_GEMV_MAX_PAIRS = 128
_ROWS_PER_TG = 8  # one simdgroup per output row
_MOE_METAL_PATH = Path(__file__).with_name("moe.metal")


@functools.cache
def _moe_lib():
    src = _MOE_METAL_PATH.read_text(encoding="utf-8")
    return torch.mps.compile_shader(f"#define ROWS {_ROWS_PER_TG}\n" + src)


def _gemv_supported(hidden_states, w1, w2) -> bool:
    hidden = hidden_states.shape[1]
    inter = w2.shape[2]
    return (
        hidden % 4 == 0
        and inter % 4 == 0
        and hidden_states.dtype == w1.dtype == w2.dtype == torch.bfloat16
        and hidden_states.is_contiguous()
        and w1.is_contiguous()
        and w2.is_contiguous()
    )


def _experts_gemv(hidden_states, w1, w2, topk_weights, topk_ids) -> torch.Tensor:
    num_tokens, hidden = hidden_states.shape
    top_k = topk_ids.shape[1]
    inter = w2.shape[2]
    pairs = num_tokens * top_k
    ids = topk_ids.to(torch.int32).reshape(-1).contiguous()
    act = torch.empty(pairs, inter, dtype=torch.float32, device=hidden_states.device)
    out = torch.empty_like(hidden_states)
    lib = _moe_lib()
    group = _ROWS_PER_TG * 32
    order = ids.argsort().to(torch.int32)
    lib.moe_gate_up(
        hidden_states,
        w1,
        ids,
        order,
        act,
        hidden,
        inter,
        top_k,
        pairs,
        threads=pairs * -(-inter // _ROWS_PER_TG) * group,
        group_size=group,
    )
    lib.moe_down(
        act,
        w2,
        ids,
        topk_weights.float().reshape(-1).contiguous(),
        out,
        hidden,
        inter,
        top_k,
        num_tokens,
        threads=num_tokens * -(-hidden // _ROWS_PER_TG) * group,
        group_size=group,
    )
    return out


def _silu_and_mul(gate_up: torch.Tensor) -> torch.Tensor:
    gate, up = gate_up.chunk(2, dim=-1)
    return F.silu(gate) * up


def _experts_sorted(hidden_states, w1, w2, topk_weights, topk_ids) -> torch.Tensor:
    """One GEMM pair per active expert; the per-expert counts cost a host sync."""
    num_tokens, hidden = hidden_states.shape
    top_k = topk_ids.shape[1]
    ids = topk_ids.reshape(-1).to(torch.int64)
    order = ids.argsort()
    counts = torch.bincount(ids, minlength=w1.shape[0]).tolist()
    x_sorted = hidden_states[order // top_k]
    y_sorted = torch.empty_like(x_sorted)
    start = 0
    for expert, count in enumerate(counts):
        if count == 0:
            continue
        end = start + count
        # F.linear takes the MPS no-graph GEMM; ``x @ w.t()`` would compile an
        # MPSGraph per new row count.
        act = _silu_and_mul(F.linear(x_sorted[start:end], w1[expert]))
        y_sorted[start:end] = F.linear(act, w2[expert])
        start = end
    y = torch.empty_like(y_sorted)
    y[order] = y_sorted
    # A batched GEMV; reducing over the middle axis is ~10x slower on MPS.
    weighted = torch.bmm(
        topk_weights.float().unsqueeze(1), y.view(num_tokens, top_k, hidden).float()
    )
    return weighted.squeeze(1).to(hidden_states.dtype)


@torch.no_grad()
def fused_experts_mps(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    """SiLU-gated experts with routed-weight sum.

    ``w1`` [E, 2I, H] stacks gate then up rows, ``w2`` [E, H, I]; returns [T, H].
    """
    if topk_ids.numel() <= _GEMV_MAX_PAIRS and _gemv_supported(hidden_states, w1, w2):
        return _experts_gemv(hidden_states, w1, w2, topk_weights, topk_ids)
    return _experts_sorted(hidden_states, w1, w2, topk_weights, topk_ids)
