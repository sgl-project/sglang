"""gfx950 residual-stream boundary from fbtriton ``tlx.ops.norm_residual_norm``, vendored in ``sglang.kernels.tlx_ops``.

``norm_residual_norm`` RMS-normalizes a BF16 branch (weight after the norm, or the input-scale form that applies it
first), rounds it to BF16, updates the FP32 stream as ``alpha * stream + beta * y`` (per-channel or scalar gate) or
``stream + y``, and RMS-normalizes the new stream, optionally rounded to BF16 first. It returns the BF16 output and
the FP32 stream. Rejected (see ``norm_residual_norm_supported``): hidden sizes other than 4096, 8192 and 16384, other
dtypes, non-contiguous or empty inputs. No FP8 or router copy of the output. The caller gates the architecture.
Runtime requirements are in ``sglang/kernels/jit/utils/tlx_runtime.py``.
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
