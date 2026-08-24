"""FlashInfer-backed router GEMM adapter for small-token MoE routing."""

from __future__ import annotations

import functools
import importlib
from typing import Callable, Optional

import torch

from sglang.kernels.jit.utils import is_arch_support_pdl
from sglang.kernels.kernel_api_logging import debug_kernel_api

_ROUTER_GEMM_SUPPORTED_SMS = frozenset({90, 100, 103, 107})

# FlashInfer's fixed-shape router GEMMs are keyed by
# (hidden_dim, num_experts, output_dtype). Keep the names as strings so importing
# this module remains cheap and works when FlashInfer is unavailable.
_ROUTER_GEMM_OP_NAMES = {
    (7168, 128, torch.bfloat16): "mm_M1_16_K7168_N128",
    (6144, 256, torch.float32): "mm_M1_16_K6144_N256",
    (7168, 256, torch.float32): "mm_M1_16_K7168_N256",
    (7168, 256, torch.bfloat16): "mm_M1_16_K7168_N256_bf16",
    (7168, 384, torch.float32): "mm_M1_16_K7168_N384",
    (7168, 384, torch.bfloat16): "mm_M1_16_K7168_N384_bf16",
    (7168, 896, torch.float32): "mm_M1_16_K7168_N896",
    (7168, 896, torch.bfloat16): "mm_M1_16_K7168_N896_bf16",
}


@functools.cache
def _resolve_flashinfer_router_gemm_op(
    op_name: str,
) -> Optional[Callable[..., None]]:
    """Resolve one optional FlashInfer API without importing it at module load."""
    try:
        flashinfer_gemm = importlib.import_module("flashinfer.gemm")
    except ImportError:
        return None
    return getattr(flashinfer_gemm, op_name, None)


def is_flashinfer_router_gemm_supported(
    num_tokens: int,
    hidden_dim: int,
    num_experts: int,
    out_dtype: torch.dtype,
    device_sm: int,
) -> bool:
    """Return whether FlashInfer exposes a router GEMM for this configuration."""
    if not 1 <= num_tokens <= 16 or device_sm not in _ROUTER_GEMM_SUPPORTED_SMS:
        return False

    op_name = _ROUTER_GEMM_OP_NAMES.get((hidden_dim, num_experts, out_dtype))
    return (
        op_name is not None and _resolve_flashinfer_router_gemm_op(op_name) is not None
    )


@debug_kernel_api
def flashinfer_router_gemm(
    hidden_states: torch.Tensor,
    router_weights: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run a fixed-shape FlashInfer router GEMM and return its output tensor.

    SGLang stores router weights as row-major ``[num_experts, hidden_dim]``.
    FlashInfer consumes the column-major ``[hidden_dim, num_experts]`` view and
    mutates a caller-owned output tensor instead of returning one.
    """
    if output is None:
        output = torch.empty(
            hidden_states.shape[0],
            router_weights.shape[0],
            device=hidden_states.device,
            dtype=out_dtype,
        )

    hidden_dim = hidden_states.shape[1]
    num_experts = router_weights.shape[0]
    op_name = _ROUTER_GEMM_OP_NAMES.get((hidden_dim, num_experts, output.dtype))
    if op_name is None:
        raise ValueError(
            "Unsupported FlashInfer router GEMM configuration: "
            f"hidden_dim={hidden_dim}, num_experts={num_experts}, "
            f"out_dtype={output.dtype}"
        )

    op = _resolve_flashinfer_router_gemm_op(op_name)
    if op is None:
        raise RuntimeError(
            f"flashinfer.gemm.{op_name} is unavailable; install a FlashInfer "
            "build that includes router GEMM support"
        )

    op(
        hidden_states,
        router_weights.T,
        output,
        launch_with_pdl=is_arch_support_pdl(),
    )
    return output


__all__ = ["flashinfer_router_gemm", "is_flashinfer_router_gemm_supported"]
