"""Fail-closed adapter for AITER's gfx950 Kimi-K3 KDA input projection."""

from __future__ import annotations

import torch

from sglang.srt.environ import envs
from sglang.srt.utils import is_hip


def enabled() -> bool:
    return is_hip() and envs.SGLANG_ROCM_K3_AITER_KDA_GROUP64.get()


def _b2_enabled() -> bool:
    return envs.SGLANG_ROCM_K3_AITER_B2_FUSIONS.get()


def _ops():
    try:
        from sglang.kernels.ops.attention.kda_flydsl.source import load_module

        module = load_module(
            "sglang.kernels.ops.gemm.kimi_k3_flydsl.kimi_k3_kda_input_group64",
            "aiter.ops.flydsl.kimi_k3_kda_input_group64",
        )
    except (ImportError, ModuleNotFoundError):
        return None, None, None
    return (
        module.kimi_k3_kda_input_group64,
        module.quantize_kimi_k3_kda_input_group64,
        module.supports_kimi_k3_kda_input_group64,
    )


def pack(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    _, quantize, _ = _ops()
    if quantize is None:
        raise RuntimeError("AITER KDA group64 quantizer is unavailable")
    return quantize(weight)


def covered(hidden: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor) -> bool:
    if not enabled():
        return False
    if hidden.shape[0] > 1 and (hidden.shape[0] != 2 or not _b2_enabled()):
        return False
    _, _, supports = _ops()
    return bool(supports is not None and supports(hidden, weight, scale))


def run(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    *,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    op, _, _ = _ops()
    if op is None:
        raise RuntimeError("AITER KDA group64 projection is unavailable")
    return op(hidden, weight, scale, output=output)


def warmup(weight: torch.Tensor, scale: torch.Tensor) -> None:
    if not enabled():
        return
    token_buckets = (1, 2) if _b2_enabled() else (1,)
    for num_tokens in token_buckets:
        hidden = torch.zeros(
            (num_tokens, 7168), dtype=torch.bfloat16, device=weight.device
        )
        if covered(hidden, weight, scale):
            run(hidden, weight, scale)
    torch.cuda.synchronize(weight.device)
