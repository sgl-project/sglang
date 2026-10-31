"""Top-k index page-table transform: maps request-relative key positions (``-1`` = empty) in place
to physical KV slots through a paged table. Kernel source:
``kernels/jit/csrc/attention/topk_page_table_transform.cuh``."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_topk_page_table_transform_module(page_size: int) -> Module:
    args = make_cpp_args(page_size, is_arch_support_pdl())
    return load_jit(
        "topk_page_table_transform",
        *args,
        cuda_files=["attention/topk_page_table_transform.cuh"],
        cuda_wrappers=[("run", f"TopkPageTableTransformKernel<{args}>::run")],
    )


def topk_page_table_transform_(
    indices: torch.Tensor,
    page_table: torch.Tensor,
    row_batch: torch.Tensor,
    page_size: int,
) -> torch.Tensor:
    """``indices`` int32 ``[rows, topk]`` (``topk`` a multiple of 4) holds request-relative key
    positions; each ``p >= 0`` becomes ``page_table[row_batch[r], p // page_size] * page_size +
    p % page_size``, ``-1`` stays. Returns ``indices``."""
    _jit_topk_page_table_transform_module(page_size).run(indices, page_table, row_batch)
    return indices
