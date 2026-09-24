"""BS1 BF16 shared gate/up projection with native SiTU shared activation."""

from __future__ import annotations

from typing import Optional

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
from sglang.srt.utils.custom_op import register_custom_op


def can_use_k3_shared_situ(x: torch.Tensor, weight: torch.Tensor) -> bool:
    return (
        x.is_cuda
        and x.dtype == torch.bfloat16
        and x.shape == (1, 7168)
        and x.is_contiguous()
        and weight.dtype == torch.bfloat16
        and weight.shape == (1536, 7168)
        and weight.is_contiguous()
        and weight.device == x.device
    )


@cache_once
def _jit_module():
    args = make_cpp_args(is_arch_support_pdl())
    return load_jit(
        "k3_shared_situ",
        *args,
        cuda_files=["gemm/k3_shared_situ.cuh"],
        cuda_wrappers=[("run", f"K3SharedSituKernel<{args}>::run")],
        extra_cuda_cflags=["-O3"],
    )


@register_custom_op(mutates_args=["front", "activated"])
def _shared_situ_op(
    x: torch.Tensor,
    weight: torch.Tensor,
    front: torch.Tensor,
    activated: torch.Tensor,
    beta: float,
    linear_beta: float,
    has_linear_beta: bool,
) -> None:
    _jit_module().run(x, weight, front, activated, beta, linear_beta, has_linear_beta)


def k3_shared_situ(
    x: torch.Tensor,
    weight: torch.Tensor,
    beta: float,
    linear_beta: Optional[float],
    *,
    front: Optional[torch.Tensor] = None,
    activated: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return FP32 `[gate | up]` and BF16 activated shared rows.

    This keeps the tiny-N GEMV's FP32 accumulation order and native SiTU math.
    PDL prefetches immutable BF16 weights; input reads wait for their producer.
    Callers use the ordinary front path when :func:`can_use_k3_shared_situ` fails.
    """
    if not can_use_k3_shared_situ(x, weight):
        raise ValueError("Unsupported K3 front SiTU input")
    if front is None:
        front = torch.empty((1, 1536), dtype=torch.float32, device=x.device)
    elif (
        front.shape != (1, 1536)
        or front.dtype != torch.float32
        or front.device != x.device
        or not front.is_contiguous()
    ):
        raise ValueError(
            "Front output must be contiguous FP32 [1, 1536] on the input device"
        )
    if activated is None:
        activated = torch.empty((1, 768), dtype=torch.bfloat16, device=x.device)
    elif (
        activated.shape != (1, 768)
        or activated.dtype != torch.bfloat16
        or activated.device != x.device
        or not activated.is_contiguous()
    ):
        raise ValueError(
            "Activated output must be contiguous BF16 [1, 768] on the input device"
        )
    _shared_situ_op(
        x,
        weight,
        front,
        activated,
        float(beta),
        float(linear_beta) if linear_beta is not None else 0.0,
        linear_beta is not None,
    )
    return front, activated
