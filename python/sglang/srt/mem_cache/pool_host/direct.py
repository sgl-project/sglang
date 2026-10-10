"""Direct-backend DMA for indexed, contiguous rows with independent slot strides."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.srt.utils.nvtx_utils import profile_range

if TYPE_CHECKING:
    pass


@cache_once
def load_block_dma_module():
    return load_jit(
        "hicache_block_dma",
        cuda_files=["kvcacheio/block_dma.cuh"],
        cuda_wrappers=[
            ("transfer", "sglang::transfer_row_blocks"),
            ("registration_bounds", "sglang::host_registration_bounds"),
            ("page_indices", "sglang::dma_page_indices"),
        ],
        extra_ldflags=["-lcuda"],
    )


def _registration_bounds(buffer: torch.Tensor) -> Sequence[Sequence[int]]:
    return load_block_dma_module().registration_bounds(buffer)


def transfer_layer_block(
    src_layers: list[torch.Tensor],
    dst_layers: list[torch.Tensor],
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    host_buffer: torch.Tensor,
):
    """Coalesce adjacent rows only when both source and destination have no gaps."""
    host_to_device = not src_layers or not src_layers[0].is_cuda
    rows = (
        (
            src_layers if host_to_device else dst_layers,
            dst_layers if host_to_device else src_layers,
            host_buffer,
            *_registration_bounds(host_buffer),
        ),
        src_indices if host_to_device else dst_indices,
        dst_indices if host_to_device else src_indices,
    )
    with profile_range("hicache.kv_dma_cpp_submit"):
        return load_block_dma_module().transfer(
            [[rows]], host_to_device, torch.cuda.current_stream().cuda_stream, None
        )


def _mamba_rows(
    host: torch.Tensor, device: torch.Tensor, layers: Sequence[int]
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    layers = sorted(layers)
    # Group adjacent layers so the native planner can coalesce their payloads.
    ranges: list[tuple[int, int]] = []
    for layer in layers:
        if ranges and ranges[-1][1] == layer:
            ranges[-1] = (ranges[-1][0], layer + 1)
        else:
            ranges.append((layer, layer + 1))
    return (
        [host[:, first:end, 0] for first, end in ranges],
        # Page-first storage still exposes [layer, slot, ...] views.
        # Transpose to [slot, layer, ...] for DMA without copying.
        [device[first:end].transpose(0, 1) for first, end in ranges],
    )


def transfer_mamba_block(
    host_buffers: list[torch.Tensor],
    device_buffers: list[torch.Tensor],
    host_indices: torch.Tensor,
    device_indices: torch.Tensor,
    layer_ids: list[int],
    host_to_device: bool,
):
    rows = [
        (
            (*_mamba_rows(host, device, layer_ids), host, *_registration_bounds(host)),
            host_indices,
            device_indices,
        )
        for host, device in zip(host_buffers, device_buffers, strict=True)
    ]
    with profile_range("hicache.mamba_dma_cpp_submit"):
        return load_block_dma_module().transfer(
            [rows], host_to_device, torch.cuda.current_stream().cuda_stream, None
        )
