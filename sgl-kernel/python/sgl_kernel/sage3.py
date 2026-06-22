"""SageAttention-3 Blackwell FP4 attention (sgl-kernel wrapper).

Wraps the upstream SageAttention-3 Blackwell kernels, contributed to sgl-kernel.
SM120a only — the underlying kernels use arch-conditional FP4 MMA, so this
extension is only built when CUDA >= 13.0 / sm_120a is available. On any other
platform, importing the ops raises a clear error rather than crashing.

The two ops exposed are intentionally GENERIC (no OmniDreams-specific packing):
  - sage3_mha_fwd: FP4 attention forward pass
  - scaled_fp4_quant: per-token FP4 quantization (3 layout variants)

Model-specific glue (packed-QKV layout, RoPE-fused quant, KV-cache stitching)
stays in the model layer and calls these ops.
"""
from typing import Optional

import torch

try:
    from sgl_kernel import sage3_ops  # triggers TORCH extension registration
except Exception as _e:
    _sage3_import_error = _e
else:
    _sage3_import_error = None


def _check_loaded() -> None:
    if _sage3_import_error is not None:
        raise ImportError(
            "sgl_kernel.sage3_ops extension is not available. It requires an "
            "sm_120a (Blackwell consumer) build of sgl-kernel with CUDA >= 13.0. "
            f"Underlying error: {_sage3_import_error!r}"
        )


def sage3_mha_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sfq: torch.Tensor,
    sfk: torch.Tensor,
    sfv: torch.Tensor,
    delta_s: torch.Tensor,
    unpadded_k: int,
    out: Optional[torch.Tensor] = None,
    softmax_scale: float = 1.0,
    is_causal: bool = False,
    per_block_mean: bool = True,
    is_bf16: bool = True,
):
    """SageAttention-3 FP4 attention forward pass (Blackwell SM120a).

    Parameters
    ----------
    q, k, v : uint8 tensors, FP4-packed [B, H, M, D/2].
    sfq, sfk, sfv : fp8 e4m3 block scales.
    delta_s : bf16/fp32 [B, H, Mq, Mk].
    unpadded_k : real (un-padded) key length.
    out : optional pre-allocated bf16 output [B, H, Mq, D].

    Returns
    -------
    (out, softmax_lse)
    """
    _check_loaded()
    return torch.ops.sgl_kernel.sage3_mha_fwd.default(
        q, k, v, sfq, sfk, sfv, delta_s, unpadded_k, out,
        softmax_scale, is_causal, per_block_mean, is_bf16,
    )


def scaled_fp4_quant(
    input: torch.Tensor,
    output: torch.Tensor,
    output_sf: torch.Tensor,
    tensor_layout: int = 1,
    variant: int = 0,
):
    """Per-token FP4 quantization (Blackwell SM120a).

    variant: 0 = plain, 1 = permute (for K), 2 = trans (for V).
    `output` is uint8 FP4-packed, `output_sf` is fp8 e4m3 scales [B,H,M,D/16].
    """
    _check_loaded()
    torch.ops.sgl_kernel.scaled_fp4_quant.default(
        input, output, output_sf, tensor_layout, variant,
    )
