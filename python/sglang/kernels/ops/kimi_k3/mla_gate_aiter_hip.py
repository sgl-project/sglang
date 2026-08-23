"""Fail-closed adapter for AITER's gfx950 Kimi-K3 MLA output gate."""

from __future__ import annotations

import os

import torch

from sglang.srt.utils import is_hip


def enabled() -> bool:
    return is_hip() and os.environ.get("SGLANG_K3_AITER_MLA_GATE", "0").lower() in (
        "1",
        "true",
    )


def _ops():
    try:
        from sglang.kernels.ops.kimi_k3.flydsl.source import load_module

        module = load_module(
            "sglang.kernels.ops.kimi_k3.flydsl.kimi_k3_mla_gate",
            "aiter.ops.flydsl.kimi_k3_mla_gate",
        )
    except (ImportError, ModuleNotFoundError):
        return None, None
    return module.kimi_k3_mla_gate, module.supports_kimi_k3_mla_gate


def covered(
    hidden: torch.Tensor,
    gate_weight: torch.Tensor,
    attention_output: torch.Tensor,
) -> bool:
    if not enabled():
        return False
    _, supports = _ops()
    return bool(
        supports is not None and supports(hidden, gate_weight, attention_output)
    )


def run(
    hidden: torch.Tensor,
    gate_weight: torch.Tensor,
    attention_output: torch.Tensor,
) -> torch.Tensor:
    op, _ = _ops()
    if op is None:
        raise RuntimeError("AITER Kimi-K3 MLA gate is unavailable")
    return op(hidden, gate_weight, attention_output, out=attention_output)


def warmup(gate_weight: torch.Tensor) -> None:
    if not enabled() or tuple(gate_weight.shape) != (1536, 7168):
        return
    hidden = torch.zeros((1, 7168), dtype=torch.bfloat16, device=gate_weight.device)
    attention = torch.zeros((1, 1536), dtype=torch.bfloat16, device=gate_weight.device)
    if covered(hidden, gate_weight, attention):
        run(hidden, gate_weight, attention)
        torch.cuda.synchronize(gate_weight.device)
