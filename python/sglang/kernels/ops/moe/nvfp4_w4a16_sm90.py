"""Grouped NVFP4 W4A16 routed-expert GEMMs for Hopper (sm_90a), with the top-k weighted sum.

Weights come from ``sglang.kernels.ops.quantization.nvfp4_sm90_repack``; the kernels are in
``jit/csrc/moe/nvfp4_w4a16_sm90.cuh``. The route plan is computed on the device, so a call captures in a CUDA graph
and replays with any routing. The decode multiplies one E2M1 code pair with a single IMAD, a trick from onnxruntime's
FP4 MoE GEMV (moe_gemv_fp4.cu, Copyright (c) Microsoft Corporation, MIT License).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.kernels.jit.utils.wgmma_rs_hazard import (
    CHECK_VERSION,
    require_no_rs_wgmma_hazard,
)
from sglang.kernels.ops.quantization.nvfp4_sm90_repack import Nvfp4Sm90Experts

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_VARIANTS = {"auto": 0, "single": 1, "pair": 2}
# "auto" runs a decode kernel up to this many tokens; at 3 and 4 tokens the grouped kernels stay.
DECODE_MAX_TOKENS = 2
# Per device: its SM count, read once (decode_pdl's CTA cap under "auto").
_sm_counts: dict = {}
# Per device and stream: the one-launch decode kernel's workspace (its call epoch, then its tagged intermediate).
_fused_ws: dict = {}


@cache_once
def _module(debug_delay_ns: int = 0) -> Module:
    # debug_delay_ns > 0 only in tests: it stretches the kernels' global-memory hand-offs so that a missing wait or a
    # reused buffer fails every time instead of rarely.
    return load_jit(
        "nvfp4_w4a16_sm90",
        str(debug_delay_ns),
        f"rs-check-{CHECK_VERSION}",
        cuda_files=["moe/nvfp4_w4a16_sm90.cuh"],
        cuda_wrappers=[
            ("run", "nvfp4_w4a16_sm90::run"),
            ("run_decode_fused", "nvfp4_w4a16_sm90::run_decode_fused"),
            ("run_decode_pdl", "nvfp4_w4a16_sm90::run_decode_pdl"),
            ("capture_id", "nvfp4_w4a16_sm90::capture_id"),
        ],
        extra_cuda_cflags=[f"-DNVFP4_MOE_DEBUG_DELAY_NS={debug_delay_ns}"]
        if debug_delay_ns
        else [],
        post_build=require_no_rs_wgmma_hazard,
    )


def _fused_workspace(x: torch.Tensor, words: int) -> torch.Tensor:
    """The one-launch decode kernel's epoch and tagged intermediate for x's device and current stream. A word is
    current only when it carries the call's epoch, so epoch and words start zeroed together and persist across calls.
    The buffer only grows (a new one is all zero, epoch included); old buffers stay alive for CUDA graphs captured with
    them. A buffer first made during a CUDA graph capture is zeroed by a node of that graph, not at capture time, so
    it is kept for that capture alone (keyed by its capture id): other graphs and eager calls never read it before a
    replay has zeroed it. Each replay of that graph then zeroes it again before its first decode call. An eager call
    before capture (SGLang's warm-up) makes a buffer that graphs captured later on that stream share without a fill."""
    dev = x.device
    key = (dev.index, torch.cuda.current_stream(dev).cuda_stream)
    ws = _fused_ws.get(key)
    if ws is not None and ws[0].numel() >= words:
        return ws[0]
    cap = _module(0).capture_id(x) if torch.cuda.is_current_stream_capturing() else 0
    if cap:
        key = key + (cap,)
        ws = _fused_ws.get(key)
        if ws is not None and ws[0].numel() >= words:
            return ws[0]
    ws = (torch.zeros(words, dtype=torch.int64, device=dev), ws)
    _fused_ws[key] = ws
    return ws[0]


# decode_fused's shared memory (decf_k's DF_* constants): the fixed region, the ring, the per-CTA maximum
_DF_FIXED, _DF_RING, _DF_SMEM = 43008, 2 * 36864, 232448


def _decode_fused_fits(T: int, H: int, I: int) -> bool:
    """decode_fused_fits in the kernel: the x rows, the ring and two intermediate rows per token."""
    oring = (_DF_FIXED + T * (2 * H + 16) + 1023) // 1024 * 1024
    return oring + _DF_RING + 2 * T * (2 * I + 16) <= _DF_SMEM


def _decode_variant(T: int, R: int, E: int, H: int, I: int) -> str:
    """The kernel "auto" takes, by measured speed on H100 (1 and 2 tokens, five model shapes): the one-launch decode
    kernel for large experts (I >= 1024), whose tagged exchange hides behind a long weight stream, the two-launch one
    for small experts, and the grouped kernels past DECODE_MAX_TOKENS or where decode_fused's shared memory does not
    fit. I is the padded width: the decode kernels read whole 256-row GEMM1 blocks, so a width that ends on a half
    block (I % 128 == 64: DeepSeek-V4.1-Flash TP8, Qwen3.8-Flash-Next TP2 and TP4) stays on the grouped kernels.

    decode_fused is a cooperative launch of one CTA per SM, so it starts only once every SM is free: next to another
    stream's kernel it waits for that kernel to end, and it cannot launch at all where a context sees fewer SMs than
    the device reports (an MPS SM limit, a green context). It stays the choice for large experts because it is still
    the fastest with DeepSeek's shared expert on a second stream (H100, Muse and DeepSeek EP8 at 1 and 2 tokens:
    65 to 136us against 74 to 140us for the grouped kernels and 78 to 178us for decode_pdl, bench/decode/overlap.py).
    A caller that overlaps a long-running kernel with the routed experts, or runs under an SM limit, pins variant."""
    if not (
        0 < T <= DECODE_MAX_TOKENS
        and R <= 256
        and E <= 1024
        and H % 256 == 0
        and I % 128 == 0
    ):
        return "auto"
    if I < 1024:
        return "decode_pdl"
    # Mixtral-8x7B and 8x22B at TP1 fit at 1 token, not at 2: there the grouped kernels run, as before
    return "decode_fused" if _decode_fused_fits(T, H, I) else "auto"


def _decode_ctas(variant: str, T: int, dev: torch.device) -> int:
    """The CTA count "auto" launches its decode kernel with (0: one CTA per SM). decode_pdl at 1 token takes three
    quarters of the SMs (99 of 132 on H100). Each of its CTAs reserves almost all of an SM's shared memory for the
    whole call, so with every SM taken, a shared expert that SGLang runs on a second stream beside the routed experts
    (GLM-5.3, Qwen3.6, GLM-5.3-Flash, Qwen3.8) waits for the routed experts to finish: at GLM-5.3 TP8 that cost 6us per
    layer in the server. With a quarter of the SMs left free, the shared GEMMs run beside it (H100, ten-layer CUDA
    graph with each model's shared expert on a second stream, bench/decode/chain2.py): GLM-5.3 TP8 40.1 to 32.9us
    per layer, Marlin 33.8us. Without a shared expert the cap costs 0.3us at Qwen3-30B and 1.4us at GLM-5.3. At 2
    tokens the capped launch moves GEMM2 to more or smaller pieces per CTA and loses up to 15us at GLM-5.3-Flash TP4,
    so 2 tokens keep one CTA per SM. decode_fused stays on every SM: next to DeepSeek's shared expert it is still the
    fastest uncapped, and every cap from 84 to 124 CTAs measured slower."""
    if variant != "decode_pdl" or T != 1:
        return 0
    sms = _sm_counts.get(dev.index)
    if sms is None:
        sms = _sm_counts[dev.index] = torch.cuda.get_device_properties(
            dev
        ).multi_processor_count
    return sms - sms // 4


def nvfp4_w4a16_sm90_moe(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    experts: Nvfp4Sm90Experts,
    topk_weights: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    scale: float = 1.0,
    swiglu_limit: Optional[float] = None,
    variant: str = "auto",
    ctas: int = 0,
    debug_delay_ns: int = 0,
) -> torch.Tensor:
    """With topk_weights: ``out[t] = scale * sum_k topk_weights[t, k] * expert(topk_ids[t, k], x[t])`` over the
    local routes (ids in [0, E); other ranks' routes are -1 or any id outside it), into ``out`` [T, H] when given.
    Without: the per-route outputs [T * top_k, H] (row t * top_k + k, zeros for non-local routes), into ``out`` when
    given.
    x is BF16 [T, H] in natural column order, topk_ids int32 [T, top_k] with distinct local ids per token (the
    decode kernels assume it and SGLang's routers guarantee it). With swiglu_limit L, the expert computes
    SiLU(min(gate, L)) * clamp(up, -L, L), as SGLang's other MoE paths do for that config field. variant ("auto",
    "single", "pair", "decode_fused" or "decode_pdl") and ctas pin the kernel for tests; "auto" takes a decode kernel
    up to DECODE_MAX_TOKENS tokens when the padded width is whole 128-column blocks, with the CTA count of
    _decode_ctas unless ctas is given."""
    T, H = x.shape
    R = T * topk_ids.shape[1]
    E, I = experts.gs13.shape[0], experts.padded
    dev = x.device
    # As in Marlin's path, a limit of 0 or below means no clamp.
    lim = float(swiglu_limit) if swiglu_limit and swiglu_limit > 0 else math.inf
    if variant == "auto":
        variant = _decode_variant(T, R, E, H, I)
        if ctas == 0:
            ctas = _decode_ctas(variant, T, dev)
    if variant in ("decode_fused", "decode_pdl"):
        if topk_weights is None:
            y = (
                out
                if out is not None
                else torch.empty((R, H), dtype=torch.bfloat16, device=dev)
            )
            combined = None
        else:  # the decode kernels write the weighted sum directly and take no per-route buffer
            y = None
            combined = (
                out
                if out is not None
                else torch.empty((T, H), dtype=torch.bfloat16, device=dev)
            )
        args = (
            x,
            topk_ids,
            experts.w13,
            experts.w2,
            experts.gs13,
            experts.gs2,
            experts.gsr,
        )
        if variant == "decode_fused":
            ws = _fused_workspace(x, 2 + R * I // 2)
            _module(debug_delay_ns).run_decode_fused(
                *args, ws, y, topk_weights, combined, float(scale), lim, ctas
            )
        else:
            inter = torch.empty((R, I), dtype=torch.bfloat16, device=dev)
            _module(debug_delay_ns).run_decode_pdl(
                *args, inter, y, topk_weights, combined, float(scale), lim, ctas
            )
        return y if combined is None else combined
    meta = torch.empty(9 * E + 68 + R // 128, dtype=torch.int32, device=dev)
    order = torch.empty(R, dtype=torch.int64, device=dev)
    gathered = torch.empty((R, H), dtype=torch.bfloat16, device=dev)
    inter = torch.empty((R, I), dtype=torch.bfloat16, device=dev)
    combined = None
    if topk_weights is None:
        y = (
            out
            if out is not None
            else torch.empty((R, H), dtype=torch.bfloat16, device=dev)
        )
    else:
        y = torch.empty((R, H), dtype=torch.bfloat16, device=dev)
        combined = (
            out
            if out is not None
            else torch.empty((T, H), dtype=torch.bfloat16, device=dev)
        )
    _module(debug_delay_ns).run(
        x,
        topk_ids,
        experts.w13,
        experts.w2,
        experts.gs13,
        experts.gs2,
        experts.gsr,
        meta,
        order,
        gathered,
        inter,
        y,
        topk_weights,
        combined,
        float(scale),
        lim,
        _VARIANTS[variant],
        ctas,
    )
    return y if combined is None else combined
