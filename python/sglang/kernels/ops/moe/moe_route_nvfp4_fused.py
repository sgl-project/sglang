"""Experimental K3 route + NVFP4 quantization with default-off serving integration."""

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
_GROUP_SIZE = 16
_NUM_GROUPS = _HIDDEN // _GROUP_SIZE
# Fusion trades the flat quant grid for one 224-thread CTA per token; that (and
# the win itself, which is launch overhead) only makes sense at small decode
# batches. Above the cap the callers run the unfused chain.
_MAX_TOKENS = 64


@cache_once
def _jit_module() -> Module:
    args = make_cpp_args(is_arch_support_pdl())
    return load_jit(
        "moe_route_nvfp4_fused",
        *args,
        cuda_files=["moe/route_nvfp4_fused.cuh"],
        cuda_wrappers=[("run", f"RouteNvfp4FusedKernel<{args}>::run")],
        # No fast-math: the routing half must stay bit-identical to
        # route_radix. NVFP4 byte equivalence is verified by the probe.
        extra_cuda_cflags=["-O3", "-DENABLE_BF16", "-DENABLE_FP4"],
        extra_dependencies=["flashinfer", "flashinfer_nv_internal"],
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
    """Experimental coverage: BF16/FP32 [M<=64,3584], 32-byte aligned rows."""
    return (
        moe_route_radix.covered(scores, bias, topk)
        and x.dim() == 2
        and x.shape[0] == scores.shape[0]
        and x.device == scores.device
        and 0 < x.shape[0] <= _MAX_TOKENS
        and x.shape[1] == _HIDDEN
        and x.dtype in (torch.bfloat16, torch.float32)
        and x.stride(1) == 1
        and x.stride(0) % 16 == 0
        and x.data_ptr() % 32 == 0
        and (x.stride(0) * x.element_size()) % 32 == 0
    )


def route_nvfp4_fused(
    scores: torch.Tensor,
    bias: torch.Tensor,
    x: torch.Tensor,
    global_scale: torch.Tensor,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
    apply_scale: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return FP32 weights, int32 ids/packed ids, uint8 FP4 and linear E4M3 scales."""
    assert covered(scores, bias, topk, x)
    assert global_scale.dtype == torch.float32 and global_scale.numel() == 1
    assert global_scale.device == x.device
    M = scores.shape[0]
    device = scores.device
    out_w = torch.empty((M, topk), dtype=torch.float32, device=device)
    out_i = torch.empty((M, topk), dtype=torch.int32, device=device)
    out_packed = torch.empty((M, topk), dtype=torch.int32, device=device)
    out_q = torch.empty((M, _HIDDEN // 2), dtype=torch.uint8, device=device)
    out_s = torch.empty((M, _NUM_GROUPS), dtype=torch.uint8, device=device)
    _jit_module().run(
        scores,
        bias,
        out_w,
        out_i,
        out_packed,
        x,
        out_q,
        out_s,
        global_scale.reshape(1),
        topk,
        float(routed_scaling_factor),
        bool(renormalize),
        bool(apply_scale),
    )
    return out_w, out_i, out_packed, out_q, out_s
