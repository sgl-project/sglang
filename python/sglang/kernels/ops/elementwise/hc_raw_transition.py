"""Shape validation for small-T raw-residual HC operations."""

import math

import torch


def _validate_state(r, c, eps):
    if (
        r.ndim != 2
        or not r.is_cuda
        or not r.is_contiguous()
        or r.dtype not in (torch.bfloat16, torch.float16)
        or c != 4
        or r.shape[0] > 24
        or r.shape[1] % c
    ):
        raise ValueError(
            "raw HC requires contiguous CUDA BF16/FP16 [T,C*H], C=4, T<=24"
        )
    h = r.shape[1] // c
    if h <= 0 or h % 512 or h > 16384 or r.data_ptr() % 32:
        raise ValueError(
            "raw transition requires 32B alignment and H a multiple of 512 <=16384"
        )
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("next RMS epsilon must be positive and finite")
    return r.shape[0], h
