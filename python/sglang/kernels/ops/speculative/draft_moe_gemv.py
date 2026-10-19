from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# Tunable launch shapes.
_UP_PAIRS_PER_CTA = 4
_DOWN_WARPS_PER_CTA = 8
_UP_STAGES = 8
_DOWN_COLS = 27
_UP_CONTIG = True
_DOWN_SMEM_BYTES = 200 * 1024


@cache_once
def _draft_moe_gemv_module(
    hidden: int,
    inter: int,
    topk: int,
    pairs: int,
    down_warps: int,
    stages: int,
    cols: int,
    contig: bool,
) -> Module:
    args = make_cpp_args(
        hidden,
        inter,
        topk,
        pairs,
        down_warps,
        stages,
        cols,
        contig,
        is_arch_support_pdl(),
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
            ("down_v3", f"DraftMoeGemv<{args}>::down_v3"),
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
    up_contig: bool = _UP_CONTIG,
    down_v3: bool = True,
) -> torch.Tensor:
    """Single-token bf16 MoE (routed top-k + gated shared expert) into ``out``.

    x [1, H]; w13 [E, 2I, H]; w2 [E, H, I]; w13_shared [2I, H]; w2_shared [H, I];
    shared_gate_w [H]; topk_ids int32 [1, K]; topk_weights fp32 [1, K];
    out [1, H]; scratch act [K + 1, I] bf16 and gate [1] fp32.
    """
    inter, topk = w2.shape[-1], topk_ids.shape[-1]
    # The down pass keeps (topk + 1) weight slices of down_cols rows in shared memory.
    slice_bytes = (topk + 1) * inter * w2.element_size()
    down_cols = min(down_cols, _DOWN_SMEM_BYTES // slice_bytes)
    module = _draft_moe_gemv_module(
        x.shape[-1],
        inter,
        topk,
        pairs_per_cta,
        down_warps_per_cta,
        up_stages,
        down_cols,
        up_contig,
    )
    if use_tma:
        module.up_tma(x, w13, w13_shared, shared_gate_w, topk_ids, act, gate)
        down = module.down_v3 if down_v3 else module.down_tma
        down(act, w2, w2_shared, topk_ids, topk_weights, gate, out)
    else:
        module.up(x, w13, w13_shared, shared_gate_w, topk_ids, act, gate)
        module.down(act, w2, w2_shared, topk_ids, topk_weights, gate, out)
    return out


# Ring stages (16 KB each) of the fused kernel; at least topk, since every routed
# down slice holds a stage until the end.
_FUSED_STAGES = 11


# Routed pairs a CTA keeps in flight (tunable): 4 at I = 128, the full ring at I = 256.
def _fused_window(inter: int) -> int:
    return 4 if inter <= 128 else _FUSED_STAGES


@cache_once
def _draft_moe_fused_module(
    hidden: int,
    inter: int,
    experts: int,
    topk: int,
    cols: int,
    stages: int,
    window: int,
    trace: bool,
    push: bool,
) -> Module:
    args = make_cpp_args(
        hidden,
        inter,
        experts,
        topk,
        cols,
        stages,
        window,
        trace,
        is_arch_support_pdl(),
        push,
    )
    return load_jit(
        "draft_moe_fused",
        *args,
        cuda_files=["speculative/draft_moe_fused.cuh"],
        cuda_wrappers=[("run", f"DraftMoeFused<{args}>::run")],
        extra_cuda_cflags=["--use_fast_math"],
    )


@cache_once
def _sm_count(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


# Copies of every tagged handoff word (kFusedReplicas in draft_moe_fused.cuh).
_FUSED_REPLICAS = 152


def draft_moe_fused_scratch(
    experts: int, topk: int, inter: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Zeroed scratch of draft_moe_fused, reused across calls: tagged logits and
    gate, tagged act values (both replicated), and the call epoch."""
    logit_words = (experts + 1 + 15) // 16 * 16
    return (
        torch.zeros(_FUSED_REPLICAS * logit_words, dtype=torch.int64, device=device),
        torch.zeros(
            _FUSED_REPLICAS * (topk + 1) * inter, dtype=torch.int32, device=device
        ),
        torch.zeros(1, dtype=torch.int32, device=device),
    )


def draft_moe_fused(
    x: torch.Tensor,
    router_w: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_shared: torch.Tensor,
    w2_shared: torch.Tensor,
    shared_gate_w: torch.Tensor,
    scratch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    out: torch.Tensor,
    trace: Optional[torch.Tensor] = None,
    stages: int = _FUSED_STAGES,
    window: Optional[int] = None,
    hop_site: Optional[torch.Tensor] = None,
    hop_token: int = 0,
) -> torch.Tensor:
    """draft_router_topk + draft_moe_gemv as one launch, bitwise equal in ids,
    weights and output. Shapes as in those two; scratch from
    draft_moe_fused_scratch. trace: int64 [grid * 16] globaltimer stamps, or None.
    hop_site: host int64 [7] from Q35Hop.site_fields(); the output is then also pushed
    into that hop as row hop_token, and its all-reduce only consumes."""
    hidden, inter = x.shape[-1], w2.shape[-1]
    cols = -(-hidden // _sm_count(x.device.index))
    module = _draft_moe_fused_module(
        hidden,
        inter,
        router_w.shape[0],
        topk_ids.shape[-1],
        cols,
        stages,
        _fused_window(inter) if window is None else window,
        trace is not None,
        hop_site is not None,
    )
    logit_buf, act_buf, epoch = scratch
    module.run(
        x,
        router_w,
        w13,
        w2,
        w13_shared,
        w2_shared,
        shared_gate_w,
        logit_buf,
        act_buf,
        epoch,
        topk_weights,
        topk_ids,
        out,
        epoch if trace is None else trace,
        epoch if hop_site is None else hop_site,
        hop_token,
    )
    return out


@cache_once
def _draft_router_module(hidden: int, experts: int, topk: int) -> Module:
    args = make_cpp_args(hidden, experts, topk, is_arch_support_pdl())
    return load_jit(
        "draft_router",
        *args,
        cuda_files=["speculative/draft_moe_gemv.cuh"],
        cuda_wrappers=[("run", f"DraftRouter<{args}>::run")],
        extra_cuda_cflags=["--use_fast_math"],
    )


def draft_router_topk(
    x: torch.Tensor,
    router_w: torch.Tensor,
    logits: torch.Tensor,
    counter: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> None:
    """Single-token softmax router: x [1, H] @ router_w [E, H]^T rounded to bf16,
    top-K (lowest id on ties) with renormalized weights. Scratch: logits fp32 [E],
    counter int32 [1] (zero between calls)."""
    _draft_router_module(x.shape[-1], router_w.shape[0], topk_ids.shape[-1]).run(
        x, router_w, logits, counter, topk_weights, topk_ids
    )
