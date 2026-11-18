"""gfx950 layernorm kernels from fbtriton ``tlx.ops``, vendored in ``sglang.kernels.tlx_ops``.

``norm_residual_norm`` (``tlx.ops.norm_residual_norm``) RMS-normalizes a BF16 branch (weight after the norm, or the
input-scale form that applies it first), rounds it to BF16, updates the FP32 stream as ``alpha * stream + beta * y``
(per-channel or scalar gate) or ``stream + y``, and RMS-normalizes the new stream, optionally rounded to BF16 first.
It returns the BF16 output and the FP32 stream. Rejected (see ``norm_residual_norm_supported``): hidden sizes other
than 4096, 8192 and 16384, other dtypes, non-contiguous or empty inputs. No FP8 or router copy of the output. It uses
TLX primitives; runtime requirements are in ``sglang/kernels/jit/utils/tlx_runtime.py``.

``rmsnorm_gate`` (``tlx.ops.rmsnorm_gate``) computes what ``FusedRMSNormGate``
(``srt/rocm_muse_spark/fused_norm_gate.py``) does for a weightless norm and head dim 64, bit for bit: per head,
``x * rsqrt(mean(x * x) + eps) * sigmoid(gate)`` in FP32, rounded to BF16. It is plain Triton (no TLX primitives), so
stock ROCm Triton runs it.

The callers gate the architecture.
"""

import torch

_HIDDEN_SIZES = (4096, 8192, 16384)


def norm_residual_norm_supported(stream: torch.Tensor, branch: torch.Tensor) -> bool:
    """Whether ``norm_residual_norm`` handles these operands; host-only, never reads device data."""
    return (
        stream.dtype == torch.float32
        and branch.dtype == torch.bfloat16
        and stream.dim() == 2
        and stream.shape == branch.shape
        and stream.shape[0] > 0
        and stream.shape[1] in _HIDDEN_SIZES
        and stream.is_contiguous()
        and branch.is_contiguous()
    )


def norm_residual_norm(
    stream: torch.Tensor,
    branch: torch.Tensor,
    *,
    post_weight: torch.Tensor | None,
    post_norm_input_scale: bool,
    pre_weight: torch.Tensor | None,
    alpha: torch.Tensor | None,
    beta: torch.Tensor | None,
    post_eps: float,
    pre_eps: float,
    pre_norm_bf16: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(output, new_stream)``; ``alpha``/``beta`` FP32 ``(H,)`` or ``(1,)``, both or neither."""
    from sglang.kernels.tlx_ops.kernels.fused.norm_residual_norm.gfx950 import (
        norm_residual_norm as _norm_residual_norm,
    )

    return _norm_residual_norm(
        stream,
        branch,
        post_weight,
        post_norm_input_scale,
        pre_weight,
        alpha,
        beta,
        post_eps,
        pre_eps,
        pre_norm_bf16,
        None,
        None,
    )


def rmsnorm_gate_supported(x: torch.Tensor, gate: torch.Tensor) -> bool:
    """Whether ``rmsnorm_gate`` handles these operands; host-only, never reads device data.

    Head dim 64 and a weightless norm are the caller's to check; here: BF16 ``x`` and ``gate`` of one ``(T, H * 64)``
    shape, ``H`` a multiple of 8, contiguous rows.
    """
    return (
        x.dim() == 2
        and gate.shape == x.shape
        and x.shape[0] > 0
        and x.shape[1] % (8 * 64) == 0
        and x.dtype == gate.dtype == torch.bfloat16
        and x.stride(1) == 1
        and gate.stride(1) == 1
    )


def rmsnorm_gate(x: torch.Tensor, gate: torch.Tensor, eps: float) -> torch.Tensor:
    """Per 64-element head: ``bf16(x * rsqrt(mean(x * x) + eps) * sigmoid(gate))`` in FP32; contiguous output."""
    from sglang.kernels.tlx_ops.kernels.fused.rmsnorm_gate.gfx950 import (
        rmsnorm_gate as _rmsnorm_gate,
    )

    return _rmsnorm_gate(x, gate, eps, None)
