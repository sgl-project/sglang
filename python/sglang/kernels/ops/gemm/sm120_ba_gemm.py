"""Experimental small-batch BF16 BA projection for SM120.

Only the measured TP2 projection (N=48, K=5120, 2 <= M <= 8) is
supported. M=1 deliberately uses PyTorch: changing its reduction order
caused observable downstream numerical differences in model testing.
This is not a batch-invariant or bitwise-equivalence guarantee.
"""

from functools import lru_cache
from typing import Optional

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# A cold shape takes the original path during graph capture. Warm up each
# desired M/device outside capture; never trigger Triton compilation in capture.
_warmed_shapes: set[tuple[int, int]] = set()


@lru_cache(None)
def _is_sm120(device: torch.device) -> bool:
    return torch.cuda.get_device_capability(device) == (12, 0)


def can_use_sm120_ba_gemm(
    x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None
) -> bool:
    """Reject unvalidated shapes, layouts, devices and execution modes."""
    return (
        not torch.compiler.is_compiling()
        and type(x) is torch.Tensor
        and type(weight) in (torch.Tensor, torch.nn.Parameter)
        and bias is None
        and x.ndim == 2
        and 2 <= x.shape[0] <= 8
        and x.shape[1] == 5120
        and tuple(weight.shape) == (48, 5120)
        and x.is_cuda
        and weight.is_cuda
        and x.device == weight.device
        and x.dtype == weight.dtype == torch.bfloat16
        and x.is_contiguous()
        and weight.is_contiguous()
        and not x.requires_grad
        and not weight.requires_grad
        and not torch.is_autocast_enabled("cuda")
        and not torch.are_deterministic_algorithms_enabled()
        and _is_sm120(x.device)
    )


@triton.jit
def _ba_dot_kernel(X, W, Y, M: tl.constexpr):
    mi = tl.arange(0, 16)
    ni = tl.program_id(0) * 16 + tl.arange(0, 16)
    ki = tl.arange(0, 128)
    acc = tl.zeros((16, 16), tl.float32)
    for start in range(40):
        kk = start * 128 + ki
        x = tl.load(X + mi[:, None] * 5120 + kk[None, :], mi[:, None] < M, 0)
        w = tl.load(W + ni[None, :] * 5120 + kk[:, None])
        acc = tl.dot(x, w, acc)
    tl.store(Y + mi[:, None] * 48 + ni[None, :], acc, mi[:, None] < M)


def sm120_ba_linear(
    x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Dispatch the allowlisted inference projection, or preserve F.linear."""
    if not can_use_sm120_ba_gemm(x, weight, bias):
        return F.linear(x, weight, bias)
    # Use the tensor's device/stream, not whichever GPU is current in the caller.
    with torch.cuda.device(x.device):
        key = (x.device.index, x.shape[0])
        capturing = torch.cuda.is_current_stream_capturing()
        if capturing and key not in _warmed_shapes:
            return F.linear(x, weight, bias)
        out = torch.empty((x.shape[0], 48), device=x.device, dtype=x.dtype)
        _ba_dot_kernel[(3,)](x, weight, out, x.shape[0], num_warps=4)
        if not capturing:
            _warmed_shapes.add(key)
    return out
