# SPDX-License-Identifier: Apache-2.0
"""Opaque page-row transfers between pinned host memory and paged device pools."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_hip_runtime,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# Sixteen 16 B loads in flight per lane measured fastest on B300 PCIe for MSA
# rows: ~51 GB/s H2D and ~52 GB/s D2H, vs 46 / 35 GB/s with 8 / 32 per lane.
_VECS_PER_LANE = 16
_BLOCK_SIZE = 512
# Two CTAs saturate B300 PCIe; MI350X needs 16 or more (measured on gfx950).
_DEFAULT_BLOCK_QUOTA = 32 if is_hip_runtime() else 2


@cache_once
def _jit_page_row_module(*, load: bool) -> Module:
    args = make_cpp_args(load, _VECS_PER_LANE, _BLOCK_SIZE)
    return load_jit(
        "page_row",
        *args,
        cuda_files=["kvcacheio/page_row.cuh"],
        cuda_wrappers=[
            ("run", f"&PageRowKernel<{args}>::run"),
            ("host_link_coherent", "&page_row_host_link_coherent"),
            ("host_pointer_usable", "&page_row_host_pointer_usable"),
        ],
    )


def warmup_page_row_transfer() -> None:
    """Compile both directions before requests can submit transfers, so compile
    errors surface at pool construction rather than mid-serving."""
    _jit_page_row_module(load=True)
    _jit_page_row_module(load=False)


def page_row_host_link_coherent(device_id: int) -> bool:
    """Whether the GPU reaches pageable host memory through the host page tables,
    as on a coherent NVLink-C2C link."""
    return bool(_jit_page_row_module(load=True).host_link_coherent(device_id))


def page_row_host_pointer_usable(device_id: int) -> bool:
    """Whether kernels may use the host address of cudaHostRegister-ed memory
    (host pointer tables hold host addresses)."""
    return bool(_jit_page_row_module(load=True).host_pointer_usable(device_id))


def transfer_page_rows(
    host_ptrs: torch.Tensor,
    device_ptrs: torch.Tensor,
    host_pages: torch.Tensor,
    device_pages: torch.Tensor,
    *,
    host_stride: int,
    device_stride: int,
    row_bytes: int,
    load: bool,
    index_stride: int = 1,
    block_quota: int = _DEFAULT_BLOCK_QUOTA,
) -> None:
    """Copy ``row_bytes`` per (layer, page) on the current stream.

    ``host_ptrs`` / ``device_ptrs`` are CUDA uint64 ``[layers]`` row-0 addresses
    (host addresses are mapped pinned memory); page ``p`` of a layer starts at
    ``ptr + p * stride``. ``host_pages`` / ``device_pages`` are validated,
    same-device CUDA int32/int64 indices whose page ``i`` is
    ``pages[i * index_stride] // index_stride``: 1 for page ids, the page size for
    page-aligned token slots. ``block_quota`` caps the grid; 0 sizes it to the work.
    Bases, strides and ``row_bytes`` must be 128-byte aligned. The caller owns
    buffers and indices until its stream completion event fires.
    """
    if host_pages.numel() == 0 or host_ptrs.numel() == 0:
        return
    _jit_page_row_module(load=load).run(
        host_ptrs,
        device_ptrs,
        host_pages,
        device_pages,
        host_stride,
        device_stride,
        row_bytes,
        index_stride,
        block_quota,
    )
