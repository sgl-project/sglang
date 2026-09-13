"""Fused K3 MoE-front prep: radix routing + trtllm id pack + mxfp8 quant.

One launch replaces the three tiny kernels between the K3 fused-front GEMM and
the trtllm-gen routed-MoE op at decode batch sizes (route_radix -> triton
(id<<16|bf16(w)) pack -> per_token_group_quant, ~7.5us busy + 2 extra launches
per MoE layer with the SMs near idle). CTAs [0, M) run the route_radix body
with the pack folded into its epilogue; CTAs [M, 2M) run the
per_token_group_quant math, one CTA per token row. Both halves reuse the
standalone kernels' device code, so ids/weights, packed ids, and quantized
activations are bit-identical to the unfused chain.

Specialized like route_radix itself: 896 experts, top-16, bf16/fp32 scores,
and a 3584-wide bf16 activation row quantized to fp8 with row-major packed
UE8M0 group-32 scales (the trtllm-gen SiTU MoE input format). Wired into
serving through sglang.srt.layers.moe.route_quant_handoff.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
from sglang.kernels.ops.moe import moe_route_radix

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_HIDDEN = 3584
_GROUP_SIZE = 32
_NUM_GROUPS = _HIDDEN // _GROUP_SIZE
# Fusion trades the flat quant grid for one 224-thread CTA per token; that (and
# the win itself, which is launch overhead) only makes sense at small decode
# batches. Above the cap the callers run the unfused chain.
_MAX_TOKENS = 64


@cache_once
def _jit_module() -> Module:
    args = make_cpp_args(is_arch_support_pdl())
    return load_jit(
        "moe_route_quant_fused",
        *args,
        cuda_files=["moe/route_quant_fused.cuh"],
        cuda_wrappers=[("run", f"RouteQuantFusedKernel<{args}>::run")],
        # No fast-math: the routing half must stay bit-identical to
        # route_radix (see its module comment); the quant half's math is
        # fast-math-independent (explicit intrinsics + bit manipulation).
        extra_cuda_cflags=["-O3"],
    )


@cache_once
def available() -> bool:
    import logging

    try:
        _jit_module()
        return True
    except Exception as e:  # pragma: no cover - toolchain dependent
        logging.getLogger(__name__).warning(
            f"Failed to load the JIT fused route+quant kernel: {e}"
        )
        return False


def covered(
    scores: torch.Tensor, bias: torch.Tensor, topk: int, x: torch.Tensor
) -> bool:
    """route_radix coverage plus the quant half: [M<=64, 3584] bf16 rows with
    32B-aligned starts (base and stride), same token count as the scores."""
    return (
        moe_route_radix.covered(scores, bias, topk)
        and x.dim() == 2
        and x.shape[0] == scores.shape[0]
        and 0 < x.shape[0] <= _MAX_TOKENS
        and x.shape[1] == _HIDDEN
        and x.dtype in (torch.bfloat16, torch.float32)
        and x.stride(1) == 1
        and x.data_ptr() % 32 == 0
        and (x.stride(0) * x.element_size()) % 32 == 0
    )


def route_quant_fused(
    scores: torch.Tensor,
    bias: torch.Tensor,
    x: torch.Tensor,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
    apply_scale: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns ``(weights [M, topk] fp32, ids [M, topk] int32, packed [M, topk]
    int32, x_q [M, 3584] fp8_e4m3, x_s [M, 28] int32 row-major packed UE8M0)``.
    Caller must have checked covered(). Winners come out in expert-id-ascending
    order (the standalone production dispatch's sorted=False)."""
    M = scores.shape[0]
    device = scores.device
    out_w = torch.empty((M, topk), dtype=torch.float32, device=device)
    out_i = torch.empty((M, topk), dtype=torch.int32, device=device)
    out_packed = torch.empty((M, topk), dtype=torch.int32, device=device)
    out_q = torch.empty((M, _HIDDEN), dtype=torch.float8_e4m3fn, device=device)
    out_s = torch.empty((M, _NUM_GROUPS // 4), dtype=torch.int32, device=device)
    _jit_module().run(
        scores,
        bias,
        out_w,
        out_i,
        out_packed,
        x,
        out_q,
        out_s,
        topk,
        float(routed_scaling_factor),
        bool(renormalize),
        bool(apply_scale),
    )
    return out_w, out_i, out_packed, out_q, out_s
