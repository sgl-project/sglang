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
def precompile_rms_norm_gate(
    dtype: torch.dtype,
    head_dim: int,
    num_heads: int,
    has_norm: bool,
    has_weight: bool,
    has_gate: bool,
    single_group: bool,
) -> Module:
    """One module per layer shape and flags; it holds every (vector width, block)
    variant the kernel may pick per call, so no token count compiles at runtime."""
    args = make_cpp_args(
        dtype,
        head_dim,
        num_heads,
        has_norm,
        has_weight,
        has_gate,
        single_group,
        is_arch_support_pdl(),
    )
    return load_jit(
        "muse_spark_norm_gate",
        *args,
        cuda_files=["muse_spark/norm_gate.cuh"],
        cuda_wrappers=[("rms_norm_gate", f"NormGateKernel<{args}>::run")],
    )


def _grouped_gate_view(gate: torch.Tensor) -> torch.Tensor:
    """Zero-copy ``[tokens, groups, group_width]`` view of a flat or grouped gate;
    the launcher checks its sizes and strides."""
    assert gate.ndim in (2, 3), "gate must be 2D or 3D"
    return gate.unsqueeze(1) if gate.ndim == 2 else gate


def rms_norm_gate(
    x: torch.Tensor,
    gate: torch.Tensor | None = None,
    weight: torch.Tensor | None = None,
    *,
    head_dim: int,
    eps: float = 0.0,
    has_norm: bool = True,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """``rms_norm(x) * weight * sigmoid(gate)``, per head, with all parts optional.

    :param x: ``[num_tokens, num_heads * head_dim]``, contiguous.
    :param gate: flat ``[num_tokens, num_heads * head_dim]`` or grouped
                 ``[num_tokens, groups, heads_per_group * head_dim]``; only the
                 innermost dimension must be contiguous.
    :param weight: ``[head_dim]`` effective weight (zero-centered gamma resolved
                   by the caller), same dtype as ``x``. Ignored when
                   ``has_norm`` is False.
    """
    assert x.is_contiguous(), "x must be contiguous"
    assert x.shape[-1] % head_dim == 0
    num_heads = x.shape[-1] // head_dim

    if out is None:
        out = torch.empty_like(x)
    x_rows = x.reshape(-1, head_dim)
    out_rows = out.reshape(-1, head_dim)

    single_group = True
    if gate is not None:
        gate = _grouped_gate_view(gate)
        single_group = gate.shape[1] == 1
    if not has_norm:
        weight = None

    module = precompile_rms_norm_gate(
        x.dtype,
        head_dim,
        num_heads,
        has_norm,
        weight is not None,
        gate is not None,
        single_group,
    )
    module.rms_norm_gate(x_rows, gate, weight, out_rows, float(eps))
    return out
