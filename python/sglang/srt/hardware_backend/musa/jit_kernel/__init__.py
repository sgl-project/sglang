"""MUSA TileLang kernels used by Qwen3.5."""

from sglang.srt.hardware_backend.musa.jit_kernel.tilelang import (
    RMSNorm,
    causal_conv1d_fn,
    causal_conv1d_fwd,
    fused_qkvzba_split_reshape_cat_contiguous,
)

__all__ = [
    "RMSNorm",
    "causal_conv1d_fn",
    "causal_conv1d_fwd",
    "fused_qkvzba_split_reshape_cat_contiguous",
]
