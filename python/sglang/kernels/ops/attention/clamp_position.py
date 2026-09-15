from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.fused_op import BaseFusedOp, register_fused_op
from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args
from sglang.kernels.spec import (
    CapabilityRequirement,
    FormatSignature,
    KernelBackend,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_clamp_position_module(dtype: torch.dtype) -> Module:
    """Compile and cache the JIT clamp_position module for a given dtype."""
    args = make_cpp_args(dtype)
    return load_jit(
        "clamp_position",
        *args,
        cuda_files=["elementwise/clamp_position.cuh"],
        cuda_wrappers=[
            ("clamp_position", f"ClampPosition<{args}>::run"),
        ],
    )


def clamp_position_cuda(seq_lens: torch.Tensor) -> torch.Tensor:
    """Compute positions = clamp(seq_lens - 1, min=0) on CUDA.

    Supported dtypes: torch.int32, torch.int64.
    """
    dst = torch.empty_like(seq_lens)
    module = _jit_clamp_position_module(seq_lens.dtype)
    module.clamp_position(dst, seq_lens)
    return dst


class ClampPositionOp(BaseFusedOp):
    """Compute non-negative, zero-based decode positions."""

    op = "attention.clamp_position"
    priority = (KernelBackend.JIT, KernelBackend.TORCH)
    capabilities = {
        KernelBackend.JIT: frozenset(
            {CapabilityRequirement.CUDA, CapabilityRequirement.HIP}
        )
    }
    format_signature = FormatSignature(
        supported_dtypes=("int32", "int64"),
        description="clamp(seq_lens - 1, min=0); returns int64 positions",
    )
    descriptions = {
        KernelBackend.JIT: "Fused clamp-position kernel (sglang.kernels.jit).",
    }

    def forward_native(self, seq_lens: torch.Tensor) -> torch.Tensor:
        return torch.clamp(seq_lens - 1, min=0).to(torch.int64)

    def forward_jit(self, seq_lens: torch.Tensor) -> torch.Tensor:
        return clamp_position_cuda(seq_lens).to(torch.int64)


_CLAMP_POSITION = register_fused_op(ClampPositionOp(), __name__, "_CLAMP_POSITION")


def clamp_position(seq_lens: torch.Tensor) -> torch.Tensor:
    return _CLAMP_POSITION(seq_lens)


__all__ = ["ClampPositionOp", "clamp_position", "clamp_position_cuda"]
