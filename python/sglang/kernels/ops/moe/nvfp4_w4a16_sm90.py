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


@cache_once
def _module(debug_delay_ns: int = 0) -> Module:
    # debug_delay_ns > 0 only in tests: it stretches the kernels' global-memory hand-offs so that a missing wait or a
    # reused buffer fails every time instead of rarely.
    return load_jit(
        "nvfp4_w4a16_sm90",
        str(debug_delay_ns),
        f"rs-check-{CHECK_VERSION}",
        cuda_files=["moe/nvfp4_w4a16_sm90.cuh"],
        cuda_wrappers=[("run", "nvfp4_w4a16_sm90::run")],
        extra_cuda_cflags=[f"-DNVFP4_MOE_DEBUG_DELAY_NS={debug_delay_ns}"]
        if debug_delay_ns
        else [],
        post_build=require_no_rs_wgmma_hazard,
    )


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
    x is BF16 [T, H] in natural column order, topk_ids int32 [T, top_k]. With swiglu_limit L, the expert computes
    SiLU(min(gate, L)) * clamp(up, -L, L), as SGLang's other MoE paths do for that config field. variant and ctas pin
    the kernel for tests."""
    T, H = x.shape
    R = T * topk_ids.shape[1]
    E, I = experts.gs13.shape[0], experts.padded
    dev = x.device
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
        # As in Marlin's path, a limit of 0 or below means no clamp.
        float(swiglu_limit) if swiglu_limit and swiglu_limit > 0 else math.inf,
        _VARIANTS[variant],
        ctas,
    )
    return y if combined is None else combined
