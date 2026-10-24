"""Batched (3D) per-tensor-scale FP8 matmul through FlashInfer's cuBLAS backend."""

from typing import Optional

import torch

from sglang.srt.utils.custom_op import register_custom_op
from sglang.utils import LazyImport

# Resolved on first call: CUDA builds without FlashInfer must still import
# this module.
_raw_bmm_fp8_batched = LazyImport("flashinfer", "bmm_fp8")


@register_custom_op(op_name="flashinfer_bmm_fp8_batched", mutates_args=["out"])
def _bmm_fp8_batched_op(
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
) -> None:
    _raw_bmm_fp8_batched(A, B, A_scale, B_scale, out.dtype, out)


def bmm_fp8(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    dtype: torch.dtype,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Batched (3D) per-tensor-scale FP8 matmul, via flashinfer's cuBLAS backend."""
    if out is None:
        out = torch.empty(
            (A.shape[0], A.shape[1], B.shape[2]),
            device=A.device,
            dtype=dtype,
        )
    _bmm_fp8_batched_op(A, B, out, A_scale, B_scale)
    return out
