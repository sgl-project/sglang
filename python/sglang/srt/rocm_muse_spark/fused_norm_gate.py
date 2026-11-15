"""Fused RMS-norm + sigmoid gate operation.

Fuses the pattern (when both are enabled):
    x_normed = rms_norm(x)  # per-head normalization
    output   = x_normed * sigmoid(g)

Supports all combinations of o_norm and o_gate being enabled/disabled.
Used in MuseSpark1x attention for the output norm + gate path.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.verify_reduce_norm_gate import (
    verify_reduce_norm_gate,
)
from sglang.srt.rocm_muse_spark import fused_op_tuning
from sglang.srt.rocm_muse_spark.base_fused_op import BaseFusedOp, FusedOpBackend


def _as_grouped_gate_view(
    x: torch.Tensor, gate: torch.Tensor, head_dim: int
) -> torch.Tensor:
    """Return a zero-copy ``[tokens, groups, group_width]`` gate view."""
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


@triton.jit
def _rms_norm_gate_fwd_kernel(
    X,  # [N, D] input (contiguous, row-stride = D)
    G,  # [T, groups, heads_per_group*D] gate values
    Y,  # [N, D] output
    W,  # [D] optional weight (effective weight, already adjusted for zcg)
    eps,
    N,  # total rows = tokens * H
    H,  # number of heads (for mapping kernel row to g indices)
    stride_g0,  # stride between gate tokens
    stride_g1,  # stride between gate groups
    HEADS_PER_GROUP: tl.constexpr,
    D: tl.constexpr,
    BD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_NORM: tl.constexpr,
    HAS_GATE: tl.constexpr,
):
    # BLOCK_M (token, head) rows per program; fp32 RMS reduction over head_dim.
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < N
    cols = tl.arange(0, BD)
    col_mask = cols < D
    mask = row_mask[:, None] & col_mask[None, :]

    x = tl.load(X + rows[:, None] * D + cols[None, :], mask=mask, other=0.0).to(
        tl.float32
    )

    if HAS_NORM:
        var = tl.sum(tl.where(mask, x * x, 0.0), axis=1) / D
        rstd = 1.0 / tl.sqrt(var + eps)
        x = x * rstd[:, None]
        if HAS_WEIGHT:
            w = tl.load(W + cols, mask=col_mask).to(tl.float32)
            x = x * w[None, :]

    # The canonical grouped gate view may be strided between tokens and groups.
    if HAS_GATE:
        token_idx = rows // H
        head_idx = rows % H
        group_idx = head_idx // HEADS_PER_GROUP
        head_in_group = head_idx % HEADS_PER_GROUP
        head_offset = group_idx * stride_g1 + head_in_group * D
        g = tl.load(
            G + token_idx[:, None] * stride_g0 + head_offset[:, None] + cols[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        x = x * tl.sigmoid(g)

    tl.store(Y + rows[:, None] * D + cols[None, :], x, mask=mask)


def _rms_norm_gate_config(
    n_rows: int,
    num_heads: int,
    head_dim: int,
    gate_stride: int,
    input_dtype: torch.dtype,
    has_norm: bool,
    has_gate: bool,
    has_weight: bool,
    device_index: int,
    *,
    op_name: str = "rms_norm_gate",
) -> tuple[int, int, dict]:
    """Resolve launch parameters for the exact norm-gate input layout."""
    cfg = fused_op_tuning.resolve(
        op_name,
        {"block_m": 8, "num_warps": 4},
        device_index=device_index,
        n_rows=n_rows,
        num_heads=num_heads,
        head_dim=head_dim,
        gate_stride=gate_stride,
        input_dtype=str(input_dtype),
        has_norm=has_norm,
        has_gate=has_gate,
        has_weight=has_weight,
    )
    extra = fused_op_tuning.split_extra(cfg, ("block_m", "num_warps"))
    return cfg["block_m"], cfg["num_warps"], extra


class FusedRMSNormGate(BaseFusedOp):
    """Fuses per-head RMS norm and/or sigmoid gate for attention output.

    Supports all four combinations of ``use_o_norm`` and ``use_o_gate``.
    When neither is enabled, acts as identity (reshape only).

    Holds a reference to the ``MuseSpark1xRMSNorm`` module (when present) for
    its config (eps, optional weight).  No additional learnable parameters
    are created.

    The input tensor ``x`` is contiguous. The Triton path normalizes flat and
    rank-grouped gates to the same zero-copy grouped view before execution.
    """

    def __init__(self, o_norm, use_o_gate: bool, head_dim: int):
        self._o_norm = o_norm
        self._use_o_gate = use_o_gate
        self._use_o_norm = o_norm is not None
        self._head_dim = head_dim
        self._eps = o_norm.eps if o_norm is not None else 0.0

        # Cache effective weight once at init (inference-only, weights are frozen).
        if self._use_o_norm and o_norm.has_weight:
            self._effective_weight = o_norm._get_effective_weight()
        else:
            self._effective_weight = None

        super().__init__()

    def _backend_runtime_eligible(self, backend, x, g=None) -> bool:
        # Keep grouped gates on backends that do not materialize a flat copy.
        return (
            g is None
            or g.ndim != 3
            or backend
            in (
                FusedOpBackend.TRITON,
                FusedOpBackend.NATIVE,
                FusedOpBackend.TORCH_COMPILE,
            )
        )

    def merges_verify_partials(self) -> bool:
        """Whether ``forward_verify_partials`` serves this op (norm always on, head dim 64)."""
        return self._use_o_norm and self._head_dim == 64

    def forward_verify_partials(
        self, state, x: torch.Tensor, g: torch.Tensor = None
    ) -> torch.Tensor:
        """Norm-gate of the attention output whose split-KV merge the verify kernels deferred (``state``);
        ``x`` is that unwritten output, used for its shape and dtype."""
        if g is not None:
            g = _as_grouped_gate_view(x, g, self._head_dim)
        return verify_reduce_norm_gate(
            state, x.shape, x.dtype, g if self._use_o_gate else None,
            self._eps, self._effective_weight,
        )  # fmt: skip

    def forward_native(
        self,
        x: torch.Tensor,
        g: torch.Tensor = None,
    ) -> torch.Tensor:
        """Reference implementation using existing norm module.

        Args:
            x: ``[T, num_heads * head_dim]`` attention output (flat).
            g: Flat ``[T, H * D]`` or grouped ``[T, G, H/G * D]`` gate values.

        Returns:
            ``[T, num_heads * head_dim]`` output after optional norm and gate.
        """
        output_shape = x.shape
        D = self._head_dim
        if self._use_o_norm:
            x = self._o_norm(x.reshape(-1, D)).reshape(x.shape)
        if self._use_o_gate:
            if g.ndim == 3:
                x = x.reshape(g.shape)
            x = g.sigmoid() * x
        return x.reshape(output_shape)

    def forward_triton(
        self,
        x: torch.Tensor,
        g: torch.Tensor = None,
    ) -> torch.Tensor:
        """Triton kernel: fused RMS norm + sigmoid gate in a single pass.

        Args:
            x: ``[T, num_heads * head_dim]`` attention output (flat).
            g: Flat ``[T, H * D]`` or grouped ``[T, G, H/G * D]`` gate values.

        Returns:
            ``[T, num_heads * head_dim]`` output after optional norm and gate.
        """
        D = self._head_dim
        H = x.shape[-1] // D

        assert x.is_contiguous(), "x must be contiguous"
        x_flat = x.reshape(-1, D)
        N = x_flat.shape[0]

        y = torch.empty_like(x_flat)

        has_weight = self._effective_weight is not None
        BD = triton.next_power_of_2(D)
        if g is None:
            stride_g0 = 0
            stride_g1 = 0
            heads_per_group = H
        else:
            g = _as_grouped_gate_view(x, g, D)
            heads_per_group = g.shape[-1] // D
            stride_g0 = g.stride(0)
            stride_g1 = g.stride(1)
        device_index = x.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()

        block_m, num_warps, extra = _rms_norm_gate_config(
            N,
            H,
            D,
            stride_g0,
            x.dtype,
            self._use_o_norm,
            self._use_o_gate,
            has_weight,
            device_index,
        )
        grid = (triton.cdiv(N, block_m),)
        _rms_norm_gate_fwd_kernel[grid](
            X=x_flat,
            G=g,
            Y=y,
            W=self._effective_weight,
            eps=self._eps,
            N=N,
            H=H,
            stride_g0=stride_g0,
            stride_g1=stride_g1,
            HEADS_PER_GROUP=heads_per_group,
            D=D,
            BD=BD,
            BLOCK_M=block_m,
            HAS_WEIGHT=has_weight,
            HAS_NORM=self._use_o_norm,
            HAS_GATE=self._use_o_gate,
            num_warps=num_warps,
            **extra,
        )

        return y.reshape(x.shape)
