from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# Bytes per bulk prefetch instruction.
CHUNK = 64 * 1024


@cache_once
def _module() -> Module:
    return load_jit(
        "mkd_prefetch",
        cuda_files=["speculative/mkd_prefetch.cuh"],
        cuda_wrappers=[("run", "MkdPrefetch::run")],
    )


def mkd_l2_prefetch(weight: torch.Tensor, nbytes: int) -> None:
    """Issue L2 bulk prefetches for the first ``nbytes`` of a contiguous 2-D weight."""
    nbytes = min(nbytes, weight.numel() * weight.element_size()) // 16 * 16
    if nbytes > 0:
        _module().run(weight, nbytes, CHUNK)
