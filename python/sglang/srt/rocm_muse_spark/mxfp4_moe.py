"""Muse Spark MoE experts with MXFP4 activations (Quark MXFP4 checkpoints, MI355X).

The Quark checkpoint declares dynamic per-32 MXFP4 activations, so A4W4 is the checkpoint numerics. Both paths
produce unweighted per-(token, slot) expert rows; Muse's FlyDSL finalize then applies the per-expert RMSNorm
with input scale before the route-weighted sum, which a combining MoE epilogue cannot express.

* Compact A4 (``SGLANG_MUSE_COMPACT_A4_VERIFY``): target-verify batches (a multiple of 8 rows, up to 512) at
  TP8 / EP8 or TP4 / EP4 run the Triton ``mxfp4_compact_a4_moe`` experts on the checkpoint weights as stored.
* a4w4 (``SGLANG_MUSE_A4W4_MOE``): prefill batches of at least ``A4W4_MIN_ROWS`` rows run aiter's FlyDSL
  stage-1 / stage-2 MXFP4 kernels on weights shuffled once at load into aiter's preshuffle layout (an extra copy
  beside the native one); stage 1 writes MXFP4 + sorted scales itself.
"""

from __future__ import annotations

import torch

from sglang.kernels.ops.moe.mxfp4_compact_a4_moe import MAX_ROWS as COMPACT_A4_MAX_ROWS
from sglang.kernels.ops.moe.mxfp4_compact_a4_moe import (
    compact_a4_experts,
    compact_a4_experts_replicated,
)
from sglang.kernels.ops.quantization.mxfp4_quant_rows import mxfp4_quant_rows
from sglang.srt.environ import envs
from sglang.srt.rocm_muse_spark import muse_moe_flydsl

COMPACT_A4_ENABLED = envs.SGLANG_MUSE_COMPACT_A4_VERIFY.get()
A4W4_ENABLED = envs.SGLANG_MUSE_A4W4_MOE.get()
A4W4_MIN_ROWS = 256
# Stage-1 launch (tile_n, async copy, waves per EU), MI355X prefill widths.
_S1_TILE_N = 128
_S1_ASYNC = True
_S1_WAVES = 3

__all__ = [
    "A4W4_ENABLED",
    "A4W4_MIN_ROWS",
    "COMPACT_A4_ENABLED",
    "COMPACT_A4_MAX_ROWS",
    "prepare_a4w4_weights",
    "quantize_rows",
    "run_a4w4",
    "run_compact_a4",
]


def _finalize_per_slot(
    y, norm_weight, scores, ids, local_start, num_local_experts, eps, gain
):
    """Muse finalize: per-expert norm (with input scale) and route-weighted sum of per-slot rows."""
    m, topk = ids.shape
    h = y.shape[-1]
    out = torch.empty((m, h), dtype=torch.bfloat16, device=ids.device)
    fx, runner = muse_moe_flydsl._require_flydsl()
    runner(
        muse_moe_flydsl._cached_finalize_launch(h, topk, True, True, "bf16", 1),
        y.view(m * topk, h),
        norm_weight,
        scores,
        ids,
        out,
        local_start,
        num_local_experts,
        float(eps),
        float(gain),
        m,
        fx.Stream(torch.cuda.current_stream(ids.device).cuda_stream),
    )
    return out


def compact_a4_with_replicas(x, ids, experts, replicas, quantized):
    """(per-slot rows, local expert index per route) of this rank's step share of home + replica experts."""
    return compact_a4_experts_replicated(
        x, ids, *replicas.operands, table=replicas.table, num_entries=replicas.num_entries,
        shuffled=experts.native_dropped, quantized=quantized,
    )  # fmt: skip


def run_compact_a4(
    x,
    scores,
    ids,
    experts,
    local_start,
    norm_weight,
    eps,
    gain,
    quantized=None,
    replicas=None,
):
    """Compact A4 target-verify experts of this rank, finalized; partial sums before the TP reduction."""
    if replicas is not None:
        y, lids = compact_a4_with_replicas(x, ids, experts, replicas, quantized)
        return _finalize_per_slot(
            y, norm_weight, scores, lids, 0, replicas.num_local, eps, gain
        )
    y = compact_a4_experts(
        x, ids, *_compact_a4_operands(experts), local_start=local_start,
        shuffled=experts.native_dropped, quantized=quantized,
    )  # fmt: skip
    return _finalize_per_slot(
        y, norm_weight, scores, ids, local_start, experts.num_local_experts, eps, gain
    )


def _compact_a4_operands(experts):
    """(w13, w2, s13, s2) as uint8 [E, N, K] views: the stored tensors, or the a4w4 preshuffled copies when the
    stored ones were dropped (the preshuffled scales are [E * N, K] with each expert's block contiguous)."""
    if not experts.native_dropped:
        return (
            experts.w13_weight,
            experts.w2_weight,
            experts.w13_weight_scale,
            experts.w2_weight_scale,
        )
    e, h, i = (
        experts.num_local_experts,
        experts.expert_hidden_size,
        experts.intermediate_size,
    )
    w13, w2, s13, s2 = experts.a4w4_weights
    return (
        w13.view(torch.uint8).view(e, 2 * i, h // 2),
        w2.view(torch.uint8).view(e, h, i // 2),
        s13.view(torch.uint8).view(e, 2 * i, h // 32),
        s2.view(torch.uint8).view(e, h, i // 32),
    )


@torch.no_grad()
def prepare_a4w4_weights(experts, *, keep_native: bool = True) -> None:
    """Shuffle the MXFP4 expert weights and scales into aiter's FlyDSL preshuffle layout: a second copy, or
    (``keep_native`` False) the only one, with every path of these experts reading the preshuffled tensors."""
    from aiter import dtypes
    from aiter.ops.shuffle import (
        shuffle_scale_a16w4,
        shuffle_weight,
        shuffle_weight_a16w4,
    )
    from aiter.utility.fp4_utils import e8m0_shuffle

    w13, w2 = experts.w13_weight, experts.w2_weight
    s13, s2 = experts.w13_weight_scale, experts.w2_weight_scale
    e, two_i, h_half = w13.shape
    h, i = w2.shape[1], two_i // 2
    experts.a4w4_weights = (
        shuffle_weight(w13.view(dtypes.fp4x2), (16, 16)),
        shuffle_weight_a16w4(w2.view(dtypes.fp4x2), 16, False),
        e8m0_shuffle(s13.view(dtypes.fp8_e8m0).view(e * two_i, h_half * 2 // 32)),
        shuffle_scale_a16w4(s2.view(dtypes.fp8_e8m0).view(e * h, i // 32), e, False),
    )
    if not keep_native:
        for param in (w13, w2, s13, s2):
            param.data = param.data.new_empty(0)
        experts.native_dropped = True


def quantize_rows(x):
    """Per-row MXFP4 expert input: (fp4x2 [m, H/2], e8m0 [m, H/32]). Row-local, so quantizing token shards
    and gathering them equals quantizing the gathered rows."""
    return mxfp4_quant_rows(x)


def run_a4w4(
    x, scores, ids, experts, local_start, norm_weight, eps, gain, *, quantized=None
):
    """a4w4 prefill experts of this rank, finalized; ``quantized`` = (fp4, scales) replaces ``x``."""
    from aiter.fused_moe import moe_sorting
    from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage1, flydsl_moe_stage2
    from aiter.utility.fp4_utils import moe_mxfp4_sort

    w1s, w2s, s1s, s2s = experts.a4w4_weights
    m, h = ids.shape[0], experts.expert_hidden_size
    topk = ids.shape[1]
    e_local = experts.num_local_experts
    block_m = 32 if m <= 1024 else 64
    mask = torch.zeros(256, dtype=torch.int32, device=ids.device)
    mask[local_start : local_start + e_local] = 1
    sorted_ids, _, sorted_eids, num_valid, _ = moe_sorting(
        ids, scores, 256, h, torch.bfloat16, block_m, expert_mask=mask
    )
    a1, a1s = quantize_rows(x) if quantized is None else quantized
    a1s = moe_mxfp4_sort(
        a1s[:m].view(m, 1, -1),
        sorted_ids=sorted_ids,
        num_valid_ids=num_valid,
        token_num=m,
        block_size=block_m,
    )
    a2, a2s = flydsl_moe_stage1(
        a1,
        w1s,
        sorted_ids,
        sorted_eids,
        num_valid,
        topk=topk,
        tile_m=block_m,
        tile_n=_S1_TILE_N,
        tile_k=256,
        a_dtype="fp4",
        b_dtype="fp4",
        out_dtype="fp4",
        w1_scale=s1s,
        a1_scale=a1s,
        sorted_weights=None,
        use_async_copy=_S1_ASYNC,
        waves_per_eu=_S1_WAVES,
    )
    y = flydsl_moe_stage2(
        a2.view(m, topk, -1),
        w2s,
        sorted_ids,
        sorted_eids,
        num_valid,
        topk=topk,
        tile_m=block_m,
        tile_n=256,
        tile_k=256,
        a_dtype="fp4",
        b_dtype="fp4",
        out_dtype="bf16",
        w2_scale=s2s,
        a2_scale=a2s,
        return_per_slot=True,
    )
    return _finalize_per_slot(
        y, norm_weight, scores, ids, local_start, e_local, eps, gain
    )
