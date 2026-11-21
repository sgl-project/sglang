# SPDX-License-Identifier: Apache-2.0
"""Page gather out of a peer GPU's mapped KV pool (NVLink), on the page-row kernel."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# Measured on GB300 NVLink with GLM-5.2 pages (36864 / 8448 B): 4, 8 and 16 vectors
# per lane tie; 64 blocks reach 95% of a contiguous copy.
_VECS_PER_LANE = 16
_BLOCK_SIZE = 512


@cache_once
def _jit_peer_page_gather_module(*, num_blocks: int) -> Module:
    # page_row's "host" side is just an address table; here it is the peer mapping.
    args = make_cpp_args(True, _VECS_PER_LANE, _BLOCK_SIZE, num_blocks)
    return load_jit(
        "page_row",
        *args,
        cuda_files=["kvcacheio/page_row.cuh"],
        cuda_wrappers=[("run", f"&PageRowKernel<{args}>::run")],
    )


def gather_pages(
    *,
    src_ptrs: torch.Tensor,
    dst_ptrs: torch.Tensor,
    src_pages: torch.Tensor,
    dst_pages: torch.Tensor,
    page_bytes: int,
    num_blocks: int,
) -> None:
    """Copy page ``src_pages[i]`` of every buffer to page ``dst_pages[i]`` on the current stream.

    ``src_ptrs`` / ``dst_ptrs`` are CUDA uint64 ``[buffers]`` page-0 addresses; page
    ``p`` of a buffer starts at ``ptr + p * page_bytes``. Source addresses may be a
    peer mapping. Pages are validated, same-device CUDA int32/int64 ids. Bases and
    ``page_bytes`` must be 128-byte aligned. ``num_blocks`` caps the CTAs taken
    from other work; each value compiles its own module.
    """
    if src_pages.numel() == 0 or src_ptrs.numel() == 0:
        return
    _jit_peer_page_gather_module(num_blocks=num_blocks).run(
        src_ptrs,
        dst_ptrs,
        src_pages,
        dst_pages,
        page_bytes,
        page_bytes,
        page_bytes,
    )
