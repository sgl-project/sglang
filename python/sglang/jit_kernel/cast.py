from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_cast_module(dtype: torch.dtype) -> Module:
    args = make_cpp_args(dtype)
    return load_jit(
        "cast",
        *args,
        cuda_files=["elementwise/cast.cuh"],
        cuda_wrappers=[("downcast_fp8", f"downcast_fp8<{args}>")],
    )


def downcast_fp8(
    k: torch.Tensor,
    v: torch.Tensor,
    k_out: torch.Tensor,
    v_out: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    loc: torch.Tensor,
    mult: int = 1,
    offset: int = 0,
) -> None:
    """Fused downcast of KV cache tensors from bf16/fp16 to fp8 (E4M3).

    Scales each value by the inverse of its per-tensor scale, clamps to the
    fp8 representable range [-448, 448], then converts to fp8 storage.

    Args:
        k:       [input_sl, head, dim] bf16/fp16 CUDA tensor
        v:       [input_sl, head, dim] bf16/fp16 CUDA tensor
        k_out:   [out_sl, head, dim]   uint8 CUDA tensor (fp8 storage)
        v_out:   [out_sl, head, dim]   uint8 CUDA tensor (fp8 storage)
        k_scale: [1] float32 CUDA tensor, scale for k
        v_scale: [1] float32 CUDA tensor, scale for v
        loc:     [input_sl] int64 CUDA tensor, destination sequence indices
        mult:    stride multiplier for output index (default 1)
        offset:  offset added to output index (default 0)
    """
    module = _jit_cast_module(k.dtype)
    module.downcast_fp8(k, v, k_out, v_out, k_scale, v_scale, loc, mult, offset)
