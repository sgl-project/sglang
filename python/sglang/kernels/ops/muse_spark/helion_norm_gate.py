"""Helion attention-output normalization and gate kernel."""

from __future__ import annotations

import helion
import helion.language as hl
import torch

from sglang.kernels.ops.muse_spark.helion_runtime import (
    token_count_specialization_key,
)

RMS_NORM_GATE_CONFIGS: dict[str, helion.Config] = {
    "short": helion.Config(
        atomic_indexing=[],
        block_sizes=[16],
        indexing=["pointer", "pointer", "pointer"],
        load_eviction_policies=["first", "first"],
        num_stages=8,
        num_warps=8,
        pid_type="flat",
        reduction_loops=[None],
    ),
    "long": helion.Config(
        atomic_indexing=[],
        block_sizes=[32],
        indexing=["pointer", "pointer", "pointer"],
        load_eviction_policies=["first", "first"],
        num_stages=1,
        num_warps=8,
        pid_type="flat",
        reduction_loops=[None],
    ),
}


def _rsqrt_approx_ftz_f32(value: torch.Tensor) -> torch.Tensor:
    """Match the reciprocal-square-root instruction used by the CUDA kernel."""
    return hl.inline_asm_elementwise(
        "rsqrt.approx.ftz.f32 $0, $1;",
        "=f,f",
        [value],
        dtype=torch.float32,
        is_pure=True,
        pack=1,
    )


def _rcp_approx_ftz_f32(value: torch.Tensor) -> torch.Tensor:
    """Use the fast reciprocal selected by CUDA's ``__fdividef(1, value)``."""
    return hl.inline_asm_elementwise(
        "rcp.approx.ftz.f32 $0, $1;",
        "=f,f",
        [value],
        dtype=torch.float32,
        is_pure=True,
        pack=1,
    )


def _exp2_approx_ftz_f32(value: torch.Tensor) -> torch.Tensor:
    """Use the single-instruction exponential underlying CUDA's ``__expf``."""
    return hl.inline_asm_elementwise(
        "ex2.approx.ftz.f32 $0, $1;",
        "=f,f",
        [value],
        dtype=torch.float32,
        is_pure=True,
        pack=1,
    )


@helion.kernel(
    key=token_count_specialization_key,
    static_shapes=False,
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def _rms_norm_gate_helion_kernel(
    x: torch.Tensor,
    gate: torch.Tensor | None,
    eps: hl.constexpr,
    head_dim: hl.constexpr,
    weight: torch.Tensor | None = None,
    has_norm: hl.constexpr = True,
) -> torch.Tensor:
    """Match every production norm/weight/gate combination."""
    tokens, width = x.size()
    width = hl.specialize(width)
    hl.specialize(x.stride())
    num_heads = hl.specialize(width // head_dim)
    num_rows = tokens * num_heads
    x_flat = x.view(-1)
    output = torch.empty_like(x)
    output_flat = output.view(-1)
    if gate is not None:
        gate_stride_token = hl.specialize(gate.stride(0))
        if gate.ndim == 2:
            gate_stride_group = 0
            heads_per_group = num_heads
            gate_span = (tokens - 1) * gate_stride_token + width
        else:
            gate_groups = gate
            gate_stride_group = hl.specialize(gate_groups.stride(1))
            group_width = hl.specialize(gate_groups.size(2))
            heads_per_group = group_width // head_dim
            gate_span = (
                (tokens - 1) * gate_stride_token
                + (gate_groups.size(1) - 1) * gate_stride_group
                + group_width
            )
        gate_storage = gate.as_strided((gate_span,), (1,))

    for tile_rows in hl.tile(num_rows):
        hl.inline_triton(
            "tl.extra.cuda.gdc_wait()",
            args=(),
            output_like=None,
        )
        flat_element = hl.arange(tile_rows.block_size * head_dim)
        row = tile_rows.begin + flat_element // head_dim
        col = flat_element % head_dim
        values = (
            x_flat[row * head_dim + col]
            .to(torch.float32)
            .reshape(tile_rows.block_size, head_dim)
        )
        if gate is not None:
            token = row // num_heads
            head = row % num_heads
            group = head // heads_per_group
            head_in_group = head % heads_per_group
            gate_offset = (
                token * gate_stride_token
                + group * gate_stride_group
                + head_in_group * head_dim
                + col
            )
            gate_values = (
                gate_storage[gate_offset]
                .to(torch.float32)
                .reshape(tile_rows.block_size, head_dim)
            )
        if has_norm:
            variance = torch.sum(values * values, dim=-1) / head_dim
            rstd = _rsqrt_approx_ftz_f32(variance + eps)
            values = values * rstd[:, None]
            if weight is not None:
                norm_weight = weight[hl.arange(head_dim)].to(torch.float32)
                values = values * norm_weight[None, :]
        if gate is not None:
            # CUDA's __expf(x) lowers to ex2.approx(x * log2(e)); spelling it
            # explicitly avoids Triton's more expensive generic exp lowering.
            gate_exp = _exp2_approx_ftz_f32(-gate_values * 1.4426950408889634)
            gate_scale = _rcp_approx_ftz_f32(1.0 + gate_exp)
            values = values * gate_scale
        output_flat[row * head_dim + col] = values.reshape(-1).to(x.dtype)
        hl.inline_triton(
            "tl.extra.cuda.gdc_launch_dependents()",
            args=(),
            output_like=None,
        )

    return output


__all__ = [
    "RMS_NORM_GATE_CONFIGS",
    "_rms_norm_gate_helion_kernel",
]
