"""SM70 UVA page-in of spilled expert rows into a landing pool."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.kernels.jit.utils.compile.paths import KERNEL_PATH

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _module() -> Module:
    cap = torch.cuda.get_device_capability()
    if cap[0] != 7:
        raise RuntimeError(
            f"sm70_dsv41 spill_page_in requires SM70 (Volta); got SM{cap[0]}{cap[1]}"
        )
    return load_jit(
        "sm70_dsv41_spill_pagein",
        cuda_files=["sm70_dsv41_spill_pagein.cuh"],
        cuda_wrappers=[("spill_page_in", "sm70_dsv41::spill_page_in")],
        extra_include_paths=[str(KERNEL_PATH / "csrc")],
    )


def spill_page_in(
    topk_ids: torch.Tensor,
    land_ids: torch.Tensor,
    slot_host_row: torch.Tensor,
    map_table: torch.Tensor,
    host_map: torch.Tensor,
    src_ptrs: torch.Tensor,
    dst_ptrs: torch.Tensor,
    row_bytes: torch.Tensor,
) -> None:
    """Remap ``topk_ids`` and copy UVA host rows into landing slots in place."""
    _module().spill_page_in(
        topk_ids,
        land_ids,
        slot_host_row,
        map_table,
        host_map,
        src_ptrs,
        dst_ptrs,
        row_bytes,
    )
