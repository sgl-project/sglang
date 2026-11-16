"""Fused per-head RMS norm + sigmoid gate (MuseSpark v1.2 attention output)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_module(dtype: torch.dtype, head_dim: int) -> Module:
    args = make_cpp_args(dtype, head_dim, is_arch_support_pdl())
    return load_jit(
        "muse_spark_norm_gate",
        *args,
        cuda_files=["muse_spark/norm_gate.cuh"],
        cuda_wrappers=[("rms_norm_gate", f"NormGateKernel<{args}>::run")],
    )


def _grouped_gate_view(
    x: torch.Tensor, gate: torch.Tensor, head_dim: int
) -> torch.Tensor:
    """Zero-copy ``[tokens, groups, group_width]`` view of a flat or grouped gate."""
    assert gate.ndim in (2, 3), "gate must be 2D or 3D"
    if gate.ndim == 2:
        assert gate.shape == x.shape, "flat gate shape must match x"
        gate = gate.unsqueeze(1)
    else:
        assert gate.shape[0] == x.shape[0], "gate token dimension must match x"
    assert gate.stride(-1) == 1, "gate's innermost dimension must be contiguous"
    assert gate.shape[-1] % head_dim == 0, (
        "gate group width must be divisible by head_dim"
    )
    assert gate.shape[1] * gate.shape[2] == x.shape[-1], (
        "grouped gate width must match x"
    )
    return gate


def rms_norm_gate(
    x: torch.Tensor,
    gate: torch.Tensor | None = None,
    weight: torch.Tensor | None = None,
    *,
    head_dim: int,
    eps: float = 0.0,
    has_norm: bool = True,
    out: torch.Tensor | None = None,
    config: int = -1,
) -> torch.Tensor:
    """``rms_norm(x) * weight * sigmoid(gate)``, per head, with all parts optional.

    :param x: ``[num_tokens, num_heads * head_dim]``, contiguous.
    :param gate: flat ``[num_tokens, num_heads * head_dim]`` or grouped
                 ``[num_tokens, groups, heads_per_group * head_dim]``; only the
                 innermost dimension must be contiguous.
    :param weight: ``[head_dim]`` effective weight (zero-centered gamma resolved
                   by the caller), same dtype as ``x``. Ignored when
                   ``has_norm`` is False.
    :param config: index into the kernel's (vector width, warps per block) table;
                   -1 selects it from the shape. Only for tuning sweeps.
    """
    assert x.is_contiguous(), "x must be contiguous"
    assert x.shape[-1] % head_dim == 0
    num_heads = x.shape[-1] // head_dim

    if out is None:
        out = torch.empty_like(x)
    x_rows = x.reshape(-1, head_dim)
    out_rows = out.reshape(-1, head_dim)

    if gate is not None:
        gate = _grouped_gate_view(x, gate, head_dim)

    module = _jit_module(x.dtype, head_dim)
    module.rms_norm_gate(
        x_rows,
        gate,
        weight if has_norm else None,
        out_rows,
        num_heads,
        float(eps),
        bool(has_norm),
        int(config),
    )
    return out
