from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_free_page_pool_module(page_size: int) -> Module:
    args = make_cpp_args(page_size)
    return load_jit(
        "free_page_pool",
        *args,
        cuda_files=["free_page_pool.cuh"],
        cuda_wrappers=[("free_dual_pool", f"free_dual_pool<{args}>")],
    )


def free_dual_pool(
    free_index: torch.Tensor,
    self_epoch: torch.Tensor,
    self_cur_epoch: int,
    self_ring: torch.Tensor,
    self_cap: int,
    self_tail: torch.Tensor,
    mark_self: bool,
    mapping: torch.Tensor,
    swa_epoch: torch.Tensor,
    swa_cur_epoch: int,
    swa_ring: torch.Tensor,
    swa_cap: int,
    swa_tail: torch.Tensor,
    scan_swa: bool,
    page_size: int,
) -> None:
    """Single-launch page-granular dedup free into device-resident rings.

    See csrc/free_page_pool.cuh. With scan_swa the winners also translate
    their page through `mapping` (full_to_swa), free live swa pages, and zero
    the mapping. Sync-free: no data-dependent shape ever reaches the host.
    """
    module = _jit_free_page_pool_module(page_size)
    module.free_dual_pool(
        free_index,
        self_epoch,
        self_cur_epoch,
        self_ring,
        self_cap,
        self_tail,
        int(mark_self),
        mapping,
        swa_epoch,
        swa_cur_epoch,
        swa_ring,
        swa_cap,
        swa_tail,
        int(scan_swa),
    )
