"""SM70 decode MoE glue: fp32 router logits from fp16 operands, the clamped
SwiGLU bitwise equal to swiglu_limit_func, and the route sum plus shared add."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module

MAX_LOGITS_TOKENS = 4


@cache_once
def _module() -> Module:
    return load_jit(
        "sm70_moe_glue",
        cuda_files=["moe/sm70_moe_glue.cuh"],
        cuda_wrappers=[
            ("router_logits", "sglang::sm70_moe_glue::router_logits"),
            ("swiglu_clamp", "sglang::sm70_moe_glue::swiglu_clamp"),
            ("sum_routes_add_shared", "sglang::sm70_moe_glue::sum_routes_add_shared"),
        ],
        extra_cuda_cflags=["--fmad=false"],
    )


@cache_once
def _is_sm70(device_index: int) -> bool:
    return torch.cuda.get_device_capability(device_index) == (7, 0)


def router_logits_covered(x: torch.Tensor, weight: torch.Tensor) -> bool:
    return (
        x.is_cuda
        and x.dim() == 2
        and 1 <= x.size(0) <= MAX_LOGITS_TOKENS
        and x.dtype == torch.float16
        and weight.dtype == torch.float16
        and x.size(1) == weight.size(1)
        and x.size(1) % 8 == 0
        and x.is_contiguous()
        and weight.is_contiguous()
        and x.data_ptr() % 16 == 0
        and weight.data_ptr() % 16 == 0
        and _is_sm70(x.device.index)
    )


def router_logits(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    out = torch.empty((x.size(0), weight.size(0)), dtype=torch.float32, device=x.device)
    _module().router_logits(out, x, weight)
    return out


def swiglu_clamp(out: torch.Tensor, gate_up: torch.Tensor, limit: float) -> None:
    _module().swiglu_clamp(out, gate_up, float(limit))


def sum_routes_add_shared(
    out: torch.Tensor, rows: torch.Tensor, topk: int, routed_scale: float
) -> None:
    _module().sum_routes_add_shared(out, rows, int(topk), float(routed_scale))
