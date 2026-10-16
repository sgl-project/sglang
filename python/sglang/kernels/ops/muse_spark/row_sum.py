"""Row sum over [N, K] fp32, bitwise-identical to ``sglang.kernels.ops.muse_spark.triton_sum``.

Same contract: the reduction order is fixed by the kernel (tl.sum's pairwise
tree), not by the CUDA runtime version, so results are stable across toolkits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_THREADS = 128


@cache_once
def _jit_module(block_k: int, threads: int) -> Module:
    args = make_cpp_args(block_k, threads, is_arch_support_pdl())
    return load_jit(
        "muse_spark_row_sum",
        *args,
        cuda_files=["muse_spark/row_sum.cuh"],
        cuda_wrappers=[("muse_spark_row_sum", f"muse_spark_row_sum<{args}>")],
    )


def row_sum(
    x: torch.Tensor,
    dim: int = -1,
    keepdim: bool = False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert x.dim() == 2, f"Only 2D tensors supported, got {x.dim()}D"
    if dim < 0:
        dim = x.dim() + dim
    assert dim == 1, f"Only dim=1 (or -1) supported, got dim={dim}"
    assert x.dtype == torch.float32

    n, k = x.shape
    block_k = 1 << max(0, (k - 1).bit_length())
    if out is None:
        out = torch.empty(n, dtype=x.dtype, device=x.device)
    flat = out.view(n)

    _jit_module(block_k, _THREADS).muse_spark_row_sum(x, flat)
    return flat.unsqueeze(-1) if keepdim else flat
