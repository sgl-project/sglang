"""Adler-32 checksum for GPU tensors.

adler32_strided_checksum(data_ptrs, strides, indices) -> int checksums selected
items across multiple tensors in a single kernel launch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_adler32_module() -> Module:
    return load_jit(
        "adler32_checksum",
        cuda_files=["elementwise/adler32_checksum.cuh"],
        cuda_wrappers=[
            ("adler32_strided", "Adler32StridedKernel::run"),
            ("adler32_reduce", "Adler32ReduceKernel::run"),
        ],
    )


_BLOCK_SIZE = 256


def _alloc_states(num_blocks: int, device: torch.device) -> torch.Tensor:
    state_bytes = num_blocks * (4 + 4 + 8)
    return torch.empty((state_bytes + 7) // 8, dtype=torch.int64, device=device)


def _reduce_states(module: Module, states: torch.Tensor, num_blocks: int) -> int:
    result = torch.empty(1, dtype=torch.int64, device=states.device)
    module.adler32_reduce(states, result, num_blocks)
    return result.item()


def adler32_strided_checksum(
    data_ptrs: list[int],
    strides: list[int],
    indices: list[torch.Tensor],
) -> int:
    """Compute Adler-32 checksum of selected items across multiple tensors.

    Args:
        data_ptrs: list of raw device pointers (int), one per tensor
        strides: list of ints, bytes per item for each tensor
        indices: list of 1D index tensors (one per tensor, on GPU)
    """
    module = _jit_adler32_module()
    device = indices[0].device

    all_ptrs = []
    all_lens = []
    total_items = 0
    for base, idx, stride in zip(data_ptrs, indices, strides):
        assert idx.is_contiguous(), "index tensor must be contiguous"
        if idx.numel() == 0:
            continue
        offsets = idx.to(torch.int64) * stride + base
        all_ptrs.append(offsets)
        all_lens.append(
            torch.full((idx.numel(),), stride, dtype=torch.int32, device=device)
        )
        total_items += idx.numel()

    if total_items == 0:
        return 1

    ptrs = torch.cat(all_ptrs)
    lens = torch.cat(all_lens)

    num_blocks = (total_items + _BLOCK_SIZE - 1) // _BLOCK_SIZE
    states = _alloc_states(num_blocks, device)

    module.adler32_strided(ptrs, lens, states, total_items)
    return _reduce_states(module, states, num_blocks)
