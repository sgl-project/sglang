# SPDX-License-Identifier: Apache-2.0
"""Decode-time bf16 GEMV for the YuE2 AR model on ROCm.

YuE2's AR decode is a batch-1 GEMV stream: every token reads the whole 3B
checkpoint, so the only thing that matters is how fast the weights can be
streamed. The custom kernel keeps x in shared memory, produces four rows per
warp and uses 16-byte vector loads; on gfx1151 it reaches ~180 GB/s on the
184704x2048 lm_head versus ~117 GB/s for rocBLAS.

Enabled with ``SGLANG_YUE2_JIT_GEMV=1``. Numerics are fp32 accumulation with a
bf16 output, i.e. within one bf16 ulp of ``F.linear`` (not bit-identical).
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn.functional as F

_CSRC = Path(__file__).resolve().parent / "csrc" / "gemv_amdgpu.cu"
_MODULE = None


def enabled() -> bool:
    return (
        os.environ.get("SGLANG_YUE2_JIT_GEMV", "0") == "1"
        and torch.version.hip is not None
        and torch.cuda.is_available()
    )


def _module():
    """Build/load the extension once (never inside graph capture)."""
    global _MODULE
    if _MODULE is None:
        from torch.utils.cpp_extension import load

        _MODULE = load(
            name="yue2_gemv_amdgpu",
            sources=[str(_CSRC)],
            extra_cflags=["-O2"],
            extra_cuda_cflags=[
                "-O3",
                "-D__HIP_PLATFORM_AMD__=1",
                *([f"--offload-arch={torch.cuda.get_device_properties(0).gcnArchName.split(':')[0]}"]
                  if torch.version.hip else []),
            ],
            verbose=os.environ.get("SGLANG_YUE2_JIT_GEMV_VERBOSE", "0") == "1",
        )
    return _MODULE


def linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """y = x @ weight^T for x[1, K], weight[N, K], bf16; falls back to F.linear."""
    if (
        x.dim() == 2
        and x.shape[0] == 1
        and x.dtype is torch.bfloat16
        and weight.dtype is torch.bfloat16
        and x.is_contiguous()
        and weight.is_contiguous()
        and weight.shape[1] == x.shape[1]
        and weight.shape[1] % 8 == 0
    ):
        out = torch.empty((1, weight.shape[0]), dtype=torch.bfloat16, device=x.device)
        _module().linear(out, x, weight)
        return out
    return F.linear(x, weight)


class GEMVLinear(torch.nn.Linear):
    """``nn.Linear`` subclass that routes batch-1 bf16 decode through the GEMV."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Decode is a single token, whatever the leading rank: [1, H],
        # [B=1, 1, H], ... Flatten it to one row for the GEMV.
        if self.bias is None and x.numel() == x.shape[-1]:
            flat = x.reshape(1, self.in_features)
            out = linear(flat, self.weight)
            return out.reshape(*x.shape[:-1], self.out_features)
        return super().forward(x)


def patch_linear_modules(model: torch.nn.Module) -> int:
    """Swap eligible bf16 linears to the GEMV class in place; returns the count."""
    if not enabled():
        return 0
    # Build before any graph capture so the first call never compiles in-graph.
    _module()
    patched = 0
    for module in model.modules():
        if type(module) is torch.nn.Linear and module.weight.dtype is torch.bfloat16:
            module.__class__ = GEMVLinear
            patched += 1
    return patched
