"""Muse Spark MoE experts with MXFP4 activations (Quark MXFP4 checkpoints, MI355X).

The Quark checkpoint declares dynamic per-32 MXFP4 activations, so A4W4 is the checkpoint numerics. Both paths
produce unweighted per-(token, slot) expert rows; Muse's FlyDSL finalize then applies the per-expert RMSNorm
with input scale before the route-weighted sum, which a combining MoE epilogue cannot express.

* Compact A4 (``SGLANG_ROCM_MUSE_COMPACT_A4_VERIFY``): target-verify batches (a multiple of 8 rows, up to 512) at
  TP8 / EP8 or TP4 / EP4 run the Triton ``mxfp4_compact_a4_moe`` experts on the checkpoint weights as stored.
* a4w4 (``SGLANG_ROCM_MUSE_A4W4_MOE``): prefill batches of at least ``A4W4_MIN_ROWS`` rows run aiter's FlyDSL
  stage-1 / stage-2 MXFP4 kernels on weights shuffled once at load into aiter's preshuffle layout (an extra copy
  beside the native one); stage 1 writes MXFP4 + sorted scales itself.
"""

from __future__ import annotations

import importlib.util
import os

import torch

from sglang.kernels.ops.moe.mxfp4_compact_a4_moe import MAX_ROWS as COMPACT_A4_MAX_ROWS
from sglang.kernels.ops.moe.mxfp4_compact_a4_moe import compact_a4_experts
from sglang.kernels.ops.quantization.mxfp4_quant_rows import mxfp4_quant_rows
from sglang.srt.environ import envs
from sglang.srt.rocm_muse_spark import muse_moe_flydsl
from sglang.srt.utils.device import is_gfx95_supported


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ImportError:  # a missing parent package
        return False


# Both paths finalize through the FlyDSL Muse finalize (sglang.kernels.ops.moe.muse_moe_finalize), built on
# FlyDSL and aiter's FlyDSL buffer helpers for gfx950; elsewhere the Triton expert path serves.
_HAS_FLYDSL = (
    is_gfx95_supported()
    and _has_module("flydsl")
    and _has_module("aiter.ops.flydsl.kernels.buffer_ops")
)
COMPACT_A4_ENABLED = envs.SGLANG_ROCM_MUSE_COMPACT_A4_VERIFY.get() and _HAS_FLYDSL
A4W4_ENABLED = envs.SGLANG_ROCM_MUSE_A4W4_MOE.get() and _HAS_FLYDSL
A4W4_MIN_ROWS = 256


def _is_gfx1250() -> bool:
    try:
        return "gfx1250" in torch.cuda.get_device_properties(0).gcnArchName
    except Exception:
        return False


# gfx1250 has no gfx950 a4w4 / compact A4 kernels; aiter's A8W4 TDM fused MoE runs all MXFP4 expert batches.
GFX1250_A8W4 = envs.SGLANG_ROCM_MUSE_GFX1250_A8W4.get() and _is_gfx1250()
if GFX1250_A8W4:
    # aiter keys its grouped-MoE CSV on topk=-1 under an EP mask and on the real topk without one; the local-id
    # calls below run without the mask, so this file repeats the TP8/EP8 tuned rows up to 256 tokens under topk=8.
    # It also holds the EP row for 16384 tokens (8193..16384-token prefill chunks), faster than aiter's default tile.
    # Up to 512 tokens both CSVs keep num_buffer_stage2 = 3: with 2 stage-2 LDS buffers the stage-2 output's
    # LDS + TDM store occasionally wrote stale LDS bytes (single non-finite elements, then NaN rows after the
    # per-expert norm) under verify load, the gfx1250 TDM-store pattern of ROCm/aiter#7202 / #7203.
    _LOCAL_IDS_CSV = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "configs",
        "gfx1250_tp8_ep8_a8w4_local_ids_fmoe.csv",
    )
    os.environ["AITER_CONFIG_GROUPED_FMOE"] = os.pathsep.join(
        p for p in (os.environ.get("AITER_CONFIG_GROUPED_FMOE"), _LOCAL_IDS_CSV) if p
    )
# Stage-1 launch (tile_n, async copy, waves per EU), MI355X prefill widths.
_S1_TILE_N = 128
_S1_ASYNC = True
_S1_WAVES = 3

__all__ = [
    "A4W4_ENABLED",
    "GFX1250_A8W4",
    "prepare_gfx1250_a8w4_weights",
    "run_gfx1250_a8w4",
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


def run_compact_a4(
    x, scores, ids, experts, local_start, norm_weight, eps, gain, quantized=None
):
    """Compact A4 target-verify experts of this rank, finalized; partial sums before the TP reduction."""
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


# Token counts 0..65535 on the experts' device (set by prepare_gfx1250_a8w4_weights); run_gfx1250_a8w4 passes a
# one-element view as aiter's num_local_tokens.
_TOKEN_COUNTS = torch.empty(0, dtype=torch.int32)


@torch.no_grad()
def prepare_gfx1250_a8w4_weights(experts) -> None:
    """gfx1250: shuffle the MXFP4 experts into aiter's gfx1250 layout (GU-interleaved w13, n32k4 scales) and drop
    the stored copy; every expert batch then runs ``run_gfx1250_a8w4``."""
    from aiter.ops.shuffle import moe_shuffle_scale, moe_shuffle_weight

    e = experts.num_local_experts
    w13 = moe_shuffle_weight(
        experts.w13_weight.data.contiguous(), is_guinterleave=True, gate_up=True
    )
    w2 = moe_shuffle_weight(experts.w2_weight.data.contiguous())
    s13 = moe_shuffle_scale(
        experts.w13_weight_scale.data.contiguous(),
        experts_cnt=e,
        is_guinterleave=True,
        gate_up=True,
    )
    s2 = moe_shuffle_scale(experts.w2_weight_scale.data.contiguous(), experts_cnt=e)
    w13 = w13.view(torch.float4_e2m1fn_x2)
    w2 = w2.view(torch.float4_e2m1fn_x2)
    w13.is_shuffled = True
    w2.is_shuffled = True
    mask = torch.zeros(256, dtype=torch.int32, device=w13.device)
    start = experts.local_expert_start_for_a8w4
    mask[start : start + e] = 1
    experts.gfx1250_a8w4 = (w13, w2, s13, s2, mask)
    global _TOKEN_COUNTS
    if _TOKEN_COUNTS.device != w13.device:
        _TOKEN_COUNTS = torch.arange(1 << 16, dtype=torch.int32, device=w13.device)
    for param in (
        experts.w13_weight,
        experts.w2_weight,
        experts.w13_weight_scale,
        experts.w2_weight_scale,
    ):
        param.data = param.data.new_empty(0)
    experts.native_dropped = True


import threading as _threading

_GFX1250_FINALIZE = _threading.local()
# (packed rows, routing words per row) of a run_gfx1250_a8w4(routed_mx=...) call while it runs
_GFX1250_DISPATCH = _threading.local()


def _install_gfx1250_gather_hook():
    """Swap aiter's TDM-MoE gather-reduce for Muse's finalize while a Muse call is active: the GEMM2 rows are
    per (token, slot) already, so the per-expert RMSNorm runs before the route-weighted sum, in one pass. And, while
    a routed-MX dispatch is active, aiter's prequantized input scatter for one that reads the packed rows."""
    import aiter.ops.flydsl.grouped_moe_gfx1250 as mk
    import aiter.ops.flydsl.moe_kernels as kernels

    if getattr(mk, "_muse_gfx1250_hooked", False):
        return
    orig = mk.flydsl_moe_gather_reduce
    orig_quant = kernels.flydsl_moe_fused_quant_preshuffle

    def gather_reduce(
        grouped_out, topids_to_rows, gather_w, out=None, num_valid_tokens=None, **kw
    ):
        ctx = getattr(_GFX1250_FINALIZE, "ctx", None)
        if ctx is None:
            return orig(
                grouped_out,
                topids_to_rows,
                gather_w,
                out=out,
                num_valid_tokens=num_valid_tokens,
                **kw,
            )
        _finalize_rows(grouped_out, topids_to_rows, out, *ctx)
        _GFX1250_FINALIZE.pushed = ctx[-1] is not None
        return out

    def quant_preshuffle(grouped_in, E, max_m, **kw):
        routed = getattr(_GFX1250_DISPATCH, "ctx", None)
        # Only aiter's input scatter of a routed-MX dispatch: contiguous grouped rows of a prequantized fp8 payload
        # read per route (every other caller, and every other form of this call, keeps aiter's kernel).
        if (
            routed is None
            or E != 1
            or kw.get("quant_mode") != "fp8"
            or kw.get("prequantized_scale") is None
            or kw.get("topids_to_rows") is None
            or kw.get("source_topk", 0) <= 0
            or any(
                kw.get(key) is not None
                for key in (
                    "masked_m",
                    "row_starts",
                    "out_payload",
                    "out_scale",
                    "row_to_token",
                    "ep_psum_params",
                )
            )
            or kw.get("row_major_scale")
            or kw.get("payload_preshuffled")
        ):
            return orig_quant(grouped_in, E, max_m, **kw)
        from sglang.srt.rocm_muse_spark.sequence_parallel import scatter_routed_mx

        packed, idw, scw = routed
        return scatter_routed_mx(
            packed, kw["topids_to_rows"], kw.get("num_valid_routes"), max_m, kw["wmma_rep"],
            grouped_in.shape[-1], idw, scw, kw["source_topk"],
        )  # fmt: skip

    mk.flydsl_moe_gather_reduce = gather_reduce
    kernels.flydsl_moe_fused_quant_preshuffle = quant_preshuffle
    mk._muse_gfx1250_hooked = True


def finalize_pushed() -> bool:
    """Whether the last run_gfx1250_a8w4(push=...) finalize pushed its rows (aiter took the hooked gather path)."""
    return getattr(_GFX1250_FINALIZE, "pushed", False)


def _finalize_rows(
    y, rows, out, scores, ids, norm_weight, local_start, num_local, eps, gain, push=None
):
    """``push`` = (slot bases, phase stride, counter, shard rows) from ``sequence_parallel.sp_push_target``: each
    finalized row goes straight into its owner rank's push slot (``sp_collective.reduce_scatter_pushed`` then
    reduces), not into ``out``."""
    import triton

    m, topk = ids.shape
    h = y.shape[-1]
    y = y.reshape(-1, h)
    if push is not None:
        slots, phase_stride, counter, shard_rows = push
        _finalize_rows_kernel[(m,)](
            y, rows, norm_weight, scores, ids, out, h, y.stride(0), local_start, num_local, eps, float(gain),
            slots, phase_stride, counter, shard_rows, TOP_K=topk, BLOCK_H=triton.next_power_of_2(h), PUSH=True,
            num_warps=4,
        )  # fmt: skip
        return
    _finalize_rows_kernel[(m,)](
        y,
        rows,
        norm_weight,
        scores,
        ids,
        out,
        h,
        y.stride(0),
        local_start,
        num_local,
        eps,
        float(gain),
        out,
        0,
        out,
        1,
        TOP_K=topk,
        BLOCK_H=triton.next_power_of_2(h),
        PUSH=False,
        num_warps=4,
    )


try:
    import triton
    import triton.language as tl

    @triton.jit
    def _finalize_rows_kernel(
        y_ptr,
        rows_ptr,
        w_ptr,
        scores_ptr,
        ids_ptr,
        out_ptr,
        H,
        stride_y,
        local_start,
        num_local,
        eps,
        gain,
        slots_ptr,
        phase_stride,
        counter_ptr,
        shard_rows,
        TOP_K: tl.constexpr,
        BLOCK_H: tl.constexpr,
        PUSH: tl.constexpr,
    ):
        m = tl.program_id(0)
        offs = tl.arange(0, BLOCK_H)
        mask = offs < H
        w = tl.load(w_ptr + offs, mask=mask, other=0.0)
        scale = (w.to(tl.float32) + gain).to(w_ptr.dtype.element_ty).to(tl.float32)
        acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
        for s in tl.static_range(TOP_K):
            e = tl.load(ids_ptr + m * TOP_K + s)
            row = tl.load(rows_ptr + m * TOP_K + s).to(tl.int64)
            # branch, not mask: with EP8 ~7 of 8 slots are remote, and a masked slot still pays the row reduction
            if (e >= local_start) & (e < local_start + num_local) & (row >= 0):
                yv = tl.load(y_ptr + row * stride_y + offs, mask=mask, other=0.0).to(
                    tl.float32
                )
                # same numerics as muse_moe_triton.finalize_norm_weighted_sum_kernel: bf16(weight * y), fp32 RMS, no affine
                yv = (yv * scale).to(tl.bfloat16).to(tl.float32)
                inv = 1.0 / tl.sqrt(tl.sum(yv * yv, axis=0) / H + eps)
                sc = tl.load(scores_ptr + m * TOP_K + s).to(tl.float32)
                acc += (sc * inv) * yv
        if PUSH:
            # Lamport payload into slot (owner, this rank) of the current phase: bf16 pairs as 32-bit words, a zero
            # word (the empty marker) sent as 0x00008000 (-0 low half), as sp_collective's producers do.
            owner = m // shard_rows
            phase = (tl.load(counter_ptr) >> 32) & 1
            base = (
                tl.load(slots_ptr + owner)
                + phase * phase_stride
                + (m - owner * shard_rows).to(tl.int64) * H * 2
            )
            v = tl.reshape(
                acc.to(tl.bfloat16).to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF,
                (BLOCK_H // 2, 2),
            )
            lo, hi = tl.split(v)
            word = lo | (hi << 16)
            word = tl.where(word == 0, 0x8000, word)
            half = tl.arange(0, BLOCK_H // 2)
            tl.store(
                base.to(tl.pointer_type(tl.int32)) + half, word, mask=half < H // 2
            )
        else:
            tl.store(
                out_ptr + m * H + offs, acc.to(out_ptr.dtype.element_ty), mask=mask
            )

    @triton.jit
    def _local_expert_ids_kernel(
        ids_ptr, out_ptr, n, local_start, num_local, BLOCK: tl.constexpr
    ):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        e = tl.load(ids_ptr + offs, mask=mask, other=-1).to(tl.int32) - local_start
        tl.store(out_ptr + offs, tl.where((e >= 0) & (e < num_local), e, -1), mask=mask)

    @triton.jit
    def _mx_ru_quant(x):
        # aiter's gfx1250 MoE input quant (moe_fused_route_quant_scatter, MX RoundUp, FP8 e4m3) of x [blocks, 32] fp32:
        # e8m0 = exponent of amax * f32(1/448), bumped when its mantissa is non-zero; payload RNE(x / 2^(e8m0 - 127)).
        working = tl.max(tl.abs(x), axis=1) * tl.full(
            (x.shape[0],), 0x3B124925, tl.int32
        ).to(tl.float32, bitcast=True)
        bits = working.to(tl.int32, bitcast=True)
        e = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
        e = tl.minimum(tl.maximum(e, 0), 0xFF)
        inv = ((254 - e) << 23).to(tl.float32, bitcast=True)
        return (x * inv[:, None]).to(tl.float8e4nv, fp_downcast_rounding="rtne"), e
except Exception:  # pragma: no cover
    _finalize_rows_kernel = None
    _local_expert_ids_kernel = None
    _mx_ru_quant = None


def _local_expert_ids(ids, local_start, num_local):
    """Rank-local expert ids, -1 for routes to other ranks (aiter's route drops ids outside [0, E))."""
    import triton

    out = torch.empty_like(ids)
    n = ids.numel()
    _local_expert_ids_kernel[(triton.cdiv(n, 1024),)](
        ids, out, n, local_start, num_local, BLOCK=1024
    )
    return out


# SP prefill / verify MoE dispatch as MXFP8 rows (aiter's own input-quant rule) widened back to BF16 after the gather
# (exact): half the bytes of the BF16 gather, while aiter keeps quantizing its own BF16 input (no pre-quantized handoff).
GFX1250_SP_MX_GATHER = (
    os.environ.get("SGLANG_ROCM_MUSE_GFX1250_SP_MX_GATHER", "1") == "1"
)


def run_gfx1250_a8w4(
    x,
    scores,
    ids,
    experts,
    local_start,
    norm_weight,
    eps,
    gain,
    quantized=None,
    num_local_tokens=None,
    routed_mx=None,
    push=None,
):
    """gfx1250 experts of this rank, finalized. aiter's A8W4 TDM MoE reads token rows directly (source_topk) and
    its GEMM2 rows stay per (token, slot); the gather step is replaced by Muse's per-expert norm + weighted sum.
    ``routed_mx``: (rows, id words, score words, local ids or None) as ``sp_gather_routed_mx(mx=True)`` returned them
    (MXFP8 + routing words, int32 [rows, width]); aiter runs its pre-quantized dispatch and the hook scatters straight
    from them. ``push``: a ``sequence_parallel.sp_push_target``; the finalize then pushes the rows into the owners'
    slots instead of returning them, and ``finalize_pushed()`` tells the caller whether it did (else reduce the
    returned rows)."""
    from aiter import ActivationType, QuantType, dtypes
    from aiter.fused_moe import fused_moe
    from aiter.ops.flydsl.grouped_moe_gfx1250 import _FUSED_ROUTE_PSUM_MAX_NUMEL
    from aiter.ops.flydsl.moe_common import GateMode

    _install_gfx1250_gather_hook()
    w13, w2, s13, s2, mask = experts.gfx1250_a8w4
    if routed_mx is not None:
        # aiter checks the shapes / dtypes of its pre-quantized pair; these views of the packed rows have them and
        # are never read (the hook's scatter reads the packed rows).
        k = experts.expert_hidden_size
        u8 = routed_mx[0].view(torch.uint8)
        quantized = (u8[:, :k].view(torch.float8_e4m3fn), u8[:, k : k + k // 32])
    ids32 = ids.to(torch.int32).contiguous()
    route_ids = ids32
    if (
        ids32.numel() <= _FUSED_ROUTE_PSUM_MAX_NUMEL
        and (quantized is None or routed_mx is not None)
        and num_local_tokens is None
    ):
        # Local ids instead of the EP mask: small batches then take aiter's one-launch route + psum in place of
        # its LUT, counter fill, route and psum launches. The finalize still reads the global ids. Larger
        # batches keep the mask: without it aiter switches to a slower input quant at large batches.
        if routed_mx is not None and routed_mx[3] is not None:
            route_ids, mask = routed_mx[3], None
        else:
            route_ids, mask = (
                _local_expert_ids(ids32, int(local_start), experts.num_local_experts),
                None,
            )
    elif num_local_tokens is None and ids32.shape[0] < _TOKEN_COUNTS.numel():
        # The EP route's valid-token count as a view of a resident arange: aiter otherwise builds it with a fill
        # kernel per MoE layer.
        num_local_tokens = _TOKEN_COUNTS[ids32.shape[0] : ids32.shape[0] + 1]
    scores32 = scores.to(torch.float32).contiguous()
    _GFX1250_FINALIZE.ctx = (
        scores32,
        ids32,
        norm_weight,
        int(local_start),
        experts.num_local_experts,
        float(eps),
        float(gain),
        push,
    )
    _GFX1250_FINALIZE.pushed = False
    _GFX1250_DISPATCH.ctx = None if routed_mx is None else routed_mx[:3]
    try:
        a1_scale, out_dtype = None, None
        if quantized is not None:
            x, a1_scale = quantized
            out_dtype = torch.bfloat16
        out = fused_moe(
            x if routed_mx is not None else x.contiguous(),
            w13,
            w2,
            scores32,
            route_ids,
            expert_mask=mask,
            activation=ActivationType.Silu,
            quant_type=QuantType.per_1x32,
            w1_scale=s13,
            w2_scale=s2,
            gate_mode=GateMode.INTERLEAVE.value,
            quant_dtype_a=dtypes.fp8,
            a1_scale=a1_scale,
            dtype=out_dtype,
            num_local_tokens=num_local_tokens,
        )
    finally:
        _GFX1250_FINALIZE.ctx = None
        _GFX1250_DISPATCH.ctx = None
    return out
