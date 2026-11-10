"""SM70 warp-per-token sigmoid top-8 router; bitwise equal to _router_triton_kernel."""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module

TOPK = 8
MIN_EXPERTS = 257
MAX_EXPERTS = 512


@cache_once
def _module() -> Module:
    return load_jit(
        "sm70_router_topk",
        cuda_files=["moe/sm70_router_topk.cuh"],
        cuda_wrappers=[("route", "sm70_router_topk::route")],
        extra_cuda_cflags=["--fmad=false"],
    )


def sm70_router_covered(scores: torch.Tensor, bias: torch.Tensor, topk: int) -> bool:
    return (
        topk == TOPK
        and scores.dim() == 2
        and MIN_EXPERTS <= scores.size(1) <= MAX_EXPERTS
        and scores.dtype == torch.float32
        and scores.stride(1) == 1
        # Triton specializes 16-aligned strides and sums in another order otherwise.
        and scores.size(1) % 16 == 0
        and scores.stride(0) % 16 == 0
        and bias.dtype == torch.float32
        and bias.is_contiguous()
        and scores.is_cuda
        and torch.cuda.get_device_capability(scores.device) == (7, 0)
    )


def sm70_router_topk(
    scores: torch.Tensor,
    bias: torch.Tensor,
    renormalize: bool,
    routed_scaling_factor: float,
    apply_scale: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    m = scores.size(0)
    weights = torch.empty((m, TOPK), dtype=torch.float32, device=scores.device)
    ids = torch.empty((m, TOPK), dtype=torch.int32, device=scores.device)
    _module().route(
        weights,
        ids,
        scores,
        bias,
        float(routed_scaling_factor),
        bool(renormalize),
        bool(apply_scale),
    )
    return weights, ids
