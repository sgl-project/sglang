"""FlashInfer-backed small-M Router GEMM dispatch for datacenter Blackwell."""

from __future__ import annotations

import functools
from typing import Callable, NamedTuple, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.utils import (
    get_device_sm,
    is_flashinfer_available,
    print_info_once,
)


class _RouterGemmSpec(NamedTuple):
    op_name: str
    out_dtype: torch.dtype


_ROUTER_GEMM_SPECS = {
    (7168, 128): _RouterGemmSpec(
        "mm_M1_16_K7168_N128", torch.bfloat16
    ),
    (7168, 256): _RouterGemmSpec("mm_M1_16_K7168_N256", torch.float32),
    (6144, 256): _RouterGemmSpec("mm_M1_16_K6144_N256", torch.float32),
}


@functools.lru_cache(maxsize=1)
def _get_flashinfer_router_gemm_ops() -> dict[str, Callable]:
    if (
        not envs.SGLANG_ENABLE_FLASHINFER_ROUTER_GEMM.get()
        or get_device_sm() not in (100, 103)
        or not is_flashinfer_available()
    ):
        return {}

    try:
        from flashinfer import gemm as flashinfer_gemm
    except ImportError:
        return {}

    ops = {}
    for spec in _ROUTER_GEMM_SPECS.values():
        op = getattr(flashinfer_gemm, spec.op_name, None)
        if op is None:
            return {}
        ops[spec.op_name] = op
    return ops


def try_flashinfer_router_gemm(
    hidden_states: torch.Tensor,
    router_weights: torch.Tensor,
    *,
    launch_with_pdl: bool = True,
) -> Optional[torch.Tensor]:
    """Run a fixed-shape FlashInfer Router GEMM, or return ``None``.

    FlashInfer owns the source-built kernel, while SGLang retains allocation
    ownership and supplies the column-major transpose view required by its
    public API. Unsupported shapes and installations preserve the existing
    model-specific fallback paths.
    """

    if not envs.SGLANG_ENABLE_FLASHINFER_ROUTER_GEMM.get():
        return None
    if hidden_states.ndim != 2 or router_weights.ndim != 2:
        return None
    if not 1 <= hidden_states.shape[0] <= 16:
        return None
    if hidden_states.shape[1] != router_weights.shape[1]:
        return None
    if hidden_states.dtype != torch.bfloat16:
        return None
    if router_weights.dtype != torch.bfloat16:
        return None
    if hidden_states.device != router_weights.device:
        return None
    if hidden_states.stride(1) != 1 or router_weights.stride(1) != 1:
        return None

    hidden_dim = hidden_states.shape[1]
    num_experts = router_weights.shape[0]
    spec = _ROUTER_GEMM_SPECS.get((hidden_dim, num_experts))
    if spec is None:
        return None

    op = _get_flashinfer_router_gemm_ops().get(spec.op_name)
    if op is None:
        return None

    output = torch.empty(
        (hidden_states.shape[0], num_experts),
        dtype=spec.out_dtype,
        device=hidden_states.device,
    )
    op(hidden_states, router_weights.t(), output, launch_with_pdl)
    print_info_once(
        "Using FlashInfer Router GEMM for eligible DeepSeek, Mistral Large 3, "
        "and GLM small-token shapes."
    )
    return output


__all__ = ["try_flashinfer_router_gemm"]
