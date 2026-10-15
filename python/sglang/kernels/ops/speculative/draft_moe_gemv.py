from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# Tunable launch shapes (bench_draft_moe.py sweeps them).
_UP_PAIRS_PER_CTA = 4
_DOWN_WARPS_PER_CTA = 8
_UP_STAGES = 8
_DOWN_COLS = 27


@cache_once
def _draft_moe_gemv_module(
    hidden: int,
    inter: int,
    topk: int,
    pairs: int,
    down_warps: int,
    stages: int,
    cols: int,
) -> Module:
    args = make_cpp_args(
        hidden, inter, topk, pairs, down_warps, stages, cols, is_arch_support_pdl()
    )
    # Same flags as the act-and-mul kernel, so the SiLU rounds identically.
    return load_jit(
        "draft_moe_gemv",
        *args,
        cuda_files=["speculative/draft_moe_gemv.cuh"],
        cuda_wrappers=[
            ("up", f"DraftMoeGemv<{args}>::up"),
            ("down", f"DraftMoeGemv<{args}>::down"),
            ("up_tma", f"DraftMoeGemv<{args}>::up_tma"),
            ("down_tma", f"DraftMoeGemv<{args}>::down_tma"),
        ],
        extra_cuda_cflags=["--use_fast_math"],
    )


def draft_moe_gemv(
    x: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_shared: torch.Tensor,
    w2_shared: torch.Tensor,
    shared_gate_w: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    out: torch.Tensor,
    act: torch.Tensor,
    gate: torch.Tensor,
    use_tma: bool = True,
    pairs_per_cta: int = _UP_PAIRS_PER_CTA,
    down_warps_per_cta: int = _DOWN_WARPS_PER_CTA,
    up_stages: int = _UP_STAGES,
    down_cols: int = _DOWN_COLS,
) -> torch.Tensor:
    """Single-token bf16 MoE (routed top-k + gated shared expert) into ``out``.

    x [1, H]; w13 [E, 2I, H]; w2 [E, H, I]; w13_shared [2I, H]; w2_shared [H, I];
    shared_gate_w [H]; topk_ids int32 [1, K]; topk_weights fp32 [1, K];
    out [1, H]; scratch act [K + 1, I] bf16 and gate [1] fp32.
    """
    module = _draft_moe_gemv_module(
        x.shape[-1],
        w2.shape[-1],
        topk_ids.shape[-1],
        pairs_per_cta,
        down_warps_per_cta,
        up_stages,
        down_cols,
    )
    if use_tma:
        module.up_tma(x, w13, w13_shared, shared_gate_w, topk_ids, act, gate)
        module.down_tma(act, w2, w2_shared, topk_ids, topk_weights, gate, out)
    else:
        module.up(x, w13, w13_shared, shared_gate_w, topk_ids, act, gate)
        module.down(act, w2, w2_shared, topk_ids, topk_weights, gate, out)
    return out
