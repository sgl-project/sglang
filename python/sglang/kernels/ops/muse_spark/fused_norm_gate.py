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

from sglang.kernels.fused_op import BaseFusedOp
from sglang.kernels.ops.muse_spark import fused_op_tuning
from sglang.kernels.spec import CapabilityRequirement, KernelBackend


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


# Widths the JIT config table can always serve: the >16B entries additionally
# need 32B-aligned pointers, which only the kernel can check.
_JIT_VEC_BYTES = (4, 8, 16)


def _jit_head_dim_ok(head_dim: int, itemsize: int) -> bool:
    """Whether norm_gate.cuh has a config for this head_dim (see select_config)."""
    return any(
        head_dim % (vec_bytes // itemsize) == 0
        and head_dim // (vec_bytes // itemsize) <= 32
        for vec_bytes in _JIT_VEC_BYTES
    )


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

    op = "muse_spark.rms_norm_gate"
    priority = (KernelBackend.JIT, KernelBackend.HELION, KernelBackend.TRITON)
    capabilities = {
        KernelBackend.HELION: {CapabilityRequirement.CUDA},
        KernelBackend.JIT: {CapabilityRequirement.CUDA},
        KernelBackend.TRITON: {
            CapabilityRequirement.CUDA,
        },
    }

    def __init__(self, o_norm, use_o_gate: bool, head_dim: int):
        super().__init__()
        # The norm is already owned by the attention module. Keep a reference
        # without registering the same module twice in state_dict.
        object.__setattr__(self, "_o_norm", o_norm)
        self._use_o_gate = use_o_gate
        self._use_o_norm = o_norm is not None
        self._head_dim = head_dim
        self._eps = o_norm.eps if o_norm is not None else 0.0

        # Cache only after checkpoint loading. In particular, a zero-centered
        # gamma produces a derived tensor, so taking this snapshot during model
        # construction would preserve the pre-load initialization values.
        object.__setattr__(self, "_effective_weight", None)
        object.__setattr__(self, "_weight_cached", False)

    def _ensure_cached_weight(self) -> None:
        if self._weight_cached:
            return
        effective_weight = (
            self._o_norm._get_effective_weight().detach()
            if self._use_o_norm and self._o_norm.has_weight
            else None
        )
        # This inference-only view must not become a second registered Parameter.
        object.__setattr__(self, "_effective_weight", effective_weight)
        object.__setattr__(self, "_weight_cached", True)

    def backend_eligible(
        self,
        backend: KernelBackend,
        x: torch.Tensor,
        g: torch.Tensor = None,
    ) -> bool:
        if not super().backend_eligible(backend, x, g):
            return False
        if backend is KernelBackend.HELION:
            return self._helion_eligible(x, g)
        if backend is not KernelBackend.JIT:
            return True
        if not _jit_head_dim_ok(self._head_dim, x.dtype.itemsize):
            return False
        # norm_gate.cuh instantiates one dtype; the Triton kernel upcasts a
        # weight of any dtype, so a mismatched checkpoint stays on Triton.
        self._ensure_cached_weight()
        return self._effective_weight is None or self._effective_weight.dtype == x.dtype

    def forward_jit(
        self,
        x: torch.Tensor,
        g: torch.Tensor = None,
    ) -> torch.Tensor:
        """Fused JIT CUDA implementation."""
        return self._forward_jit(x, g)

    def _forward_jit(
        self,
        x: torch.Tensor,
        g: torch.Tensor = None,
    ) -> torch.Tensor:
        """Run JIT CUDA without model-specific dispatch."""
        if x.numel() == 0:
            return self.forward_native(x, g)

        from sglang.kernels.ops.muse_spark.norm_gate import (
            rms_norm_gate as _jit_rms_norm_gate,
        )

        self._ensure_cached_weight()
        return _jit_rms_norm_gate(
            x,
            g if self._use_o_gate else None,
            self._effective_weight,
            head_dim=self._head_dim,
            eps=self._eps,
            has_norm=self._use_o_norm,
        )

    def _helion_eligible(
        self,
        x: torch.Tensor,
        g: torch.Tensor = None,
    ) -> bool:
        from sglang.kernels.ops.muse_spark.helion_runtime import helion_supported

        if not helion_supported("rms_norm_gate", x):
            return False
        self._ensure_cached_weight()
        return (
            x.numel() > 0
            and x.dtype == torch.bfloat16
            and x.dim() == 2
            and x.is_contiguous()
            and self._head_dim == 64
            and x.shape[1] == 32 * self._head_dim
            and self._use_o_norm
            and self._use_o_gate
            and self._effective_weight is None
            and g is not None
            and g.device == x.device
            and g.dtype == x.dtype
            and g.dim() == 2
            and g.shape == x.shape
            and g.stride(-1) == 1
        )

    def forward_helion(
        self,
        x: torch.Tensor,
        g: torch.Tensor = None,
    ) -> torch.Tensor:
        """Run Helion behind the existing fused operation boundary."""
        from sglang.kernels.jit.utils import is_arch_support_pdl
        from sglang.kernels.ops.muse_spark.helion_norm_gate import (
            RMS_NORM_GATE_CONFIGS,
            _rms_norm_gate_helion_kernel,
        )
        from sglang.kernels.ops.muse_spark.helion_runtime import (
            run_pinned_helion,
        )

        return run_pinned_helion(
            _rms_norm_gate_helion_kernel,
            RMS_NORM_GATE_CONFIGS,
            (x, g, self._eps, self._head_dim, None, True),
            num_tokens=x.shape[0],
            launch_pdl=is_arch_support_pdl(),
        )

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

    def forward_cpu(
        self,
        x: torch.Tensor,
        g: torch.Tensor = None,
    ) -> torch.Tensor:
        """``forward_native`` with the per-head norm on sgl-kernel."""
        self._ensure_cached_weight()
        weight = self._effective_weight
        if self._use_o_norm and (
            x.dtype not in (torch.bfloat16, torch.float16)
            or (weight is not None and weight.dtype != x.dtype)
        ):
            return self.forward_native(x, g)

        output_shape = x.shape
        D = self._head_dim
        if self._use_o_norm:
            x_2d = x.reshape(-1, D)
            x = (
                torch.ops.sgl_kernel.l2norm_cpu(x_2d, self._eps)
                if weight is None
                else torch.ops.sgl_kernel.rmsnorm_cpu(x_2d, weight, self._eps)
            ).reshape(x.shape)
        if self._use_o_gate:
            if g.ndim == 3:
                x = x.reshape(g.shape)
            x = g.sigmoid().mul_(x)
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

        self._ensure_cached_weight()
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
