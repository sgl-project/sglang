"""Direct-backend DMA for indexed, contiguous rows with independent slot strides."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from tvm_ffi import Array, convert

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.srt.utils.nvtx_utils import profile_range

if TYPE_CHECKING:
    from sglang.srt.mem_cache.l2_transfer import L2Transfer


def can_batch_direct(host_pool) -> bool:
    """Select the static-pool path once, before any transfers are submitted."""
    from sglang.srt.mem_cache.memory_pool import (
        MambaPool,
        MHATokenToKVPool,
        MHATokenToKVPoolMXFP8,
    )
    from sglang.srt.mem_cache.pool_host.group import HostPoolGroup
    from sglang.srt.mem_cache.pool_host.mamba import MambaPoolHost
    from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
    from sglang.srt.mem_cache.pool_host.mha_mxfp8 import MHATokenToKVPoolMXFP8Host

    pools = (
        [entry.host_pool for entry in host_pool.entries]
        if isinstance(host_pool, HostPoolGroup)
        else [host_pool]
    )
    for host in pools:
        if type(host) is MambaPoolHost:
            if type(host.device_pool) is not MambaPool or host.slot_state_buffers:
                return False
        elif type(host) in (MHATokenToKVPoolHost, MHATokenToKVPoolMXFP8Host):
            device = host.device_pool
            if (
                type(device) not in (MHATokenToKVPool, MHATokenToKVPoolMXFP8)
                or host.layout != "page_first_direct"
                or host.mtp_draft_device_pools
                or device.layer_shard_enabled
                or device.kv_cache_layout != "nhd"
            ):
                return False
        else:
            return False
    return True


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


@dataclass(frozen=True)
class _DirectRows:
    pool_index: int
    layout: Array

    @classmethod
    def create(
        cls,
        pool_index: int,
        host_rows: list[torch.Tensor],
        device_rows: list[torch.Tensor],
        host_buffer: torch.Tensor,
    ) -> _DirectRows:
        # Cache FFI metadata with the views; neither changes per copy.
        # The array retains the views until layout invalidation.
        layout = convert(
            (host_rows, device_rows, host_buffer, *_registration_bounds(host_buffer))
        )
        return cls(pool_index, layout)


@dataclass(frozen=True)
class DirectTransfer:
    blocks: list[list[_DirectRows]]
    block_ends: tuple[int, ...] | None
    page_pools: tuple[int, ...]

    def submit(
        self,
        transfers: Sequence[L2Transfer],
        device_indices: Sequence[torch.Tensor],
        stream: torch.cuda.Stream,
        on_layer_done: Callable[[int], None] | None = None,
    ) -> None:
        module = load_block_dma_module()
        indices = [
            (t.host_indices, ids)
            for t, ids in zip(transfers, device_indices, strict=True)
        ]
        # KV payloads and scales share page IDs; Mamba keeps state-slot IDs.
        if self.page_pools:
            pages = module.page_indices(
                [
                    (ids, transfers[i].host_pool.page_size)
                    for i in self.page_pools
                    for ids in indices[i]
                ]
            )
            for pair, i in enumerate(self.page_pools):
                indices[i] = (pages[2 * pair], pages[2 * pair + 1])
        blocks = [
            [(rows.layout, *indices[rows.pool_index]) for rows in block]
            for block in self.blocks
        ]
        block_ends = self.block_ends
        host_to_device = block_ends is not None
        name = "restore" if host_to_device else "backup"
        complete: Callable[[int], None] | None = None
        if on_layer_done is not None:
            assert block_ends is not None
            complete = lambda block: on_layer_done(block_ends[block] - 1)
        with profile_range(f"hicache.{name}_dma_cpp_submit"):
            module.transfer(blocks, host_to_device, stream.cuda_stream, complete)


def _direct_layout_key(
    transfers: Sequence[L2Transfer], block_ends: tuple[int, ...] | None
) -> tuple:
    from sglang.srt.mem_cache.pool_host.mamba import MambaPoolHost
    from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost

    pools = []
    for transfer in transfers:
        host, device = transfer.host_pool, transfer.device_pool
        if isinstance(host, MambaPoolHost):
            buffers = (id(host.kv_buffer), id(device.mamba_cache))
        else:
            assert isinstance(host, MHATokenToKVPoolHost)
            buffers = (id(host.kv_buffer), id(device.k_buffer), id(device.v_buffer))
        pools.append((host, device, transfer.layer_mapper, transfer.is_draft, buffers))
    return block_ends, tuple(pools)


def prepare_direct_transfer(
    transfers: Sequence[L2Transfer], block_ends: tuple[int, ...] | None
) -> DirectTransfer:
    from sglang.srt.mem_cache.l2_transfer import local_layers_for_block
    from sglang.srt.mem_cache.pool_host.mamba import MambaPoolHost
    from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
    from sglang.srt.mem_cache.pool_host.mha_mxfp8 import MHATokenToKVPoolMXFP8Host

    num_blocks = 1 if block_ends is None else len(block_ends)
    blocks: list[list[_DirectRows]] = [[] for _ in range(num_blocks)]
    page_pools = []
    for pool_index, transfer in enumerate(transfers):
        host, device = transfer.host_pool, transfer.device_pool
        is_mamba = isinstance(host, MambaPoolHost)
        assert is_mamba or type(host) in (
            MHATokenToKVPoolHost,
            MHATokenToKVPoolMXFP8Host,
        )
        assert not transfer.is_draft
        if not is_mamba:
            assert not host.mtp_draft_device_pools
        if block_ends is None:
            # Backups include every local layer, independent of the restore schedule.
            layer_count = host.num_mamba_layers if is_mamba else host.layer_num
            layer_blocks = [list(range(layer_count))]
        else:
            layer_blocks = [
                local_layers_for_block(transfer, first, end, block_ends[-1])
                for first, end in zip((0, *block_ends), block_ends)
            ]
        if is_mamba:
            buffers = list(zip(host.conv_buffer, device.mamba_cache.conv, strict=True))
            if host.temporal_state_elem_size > 0:
                buffers.append((host.temporal_buffer, device.mamba_cache.temporal))
            for buffer, gpu in buffers:
                for block, layers in enumerate(layer_blocks):
                    if layers:
                        blocks[block].append(
                            _DirectRows.create(
                                pool_index, *_mamba_rows(buffer, gpu, layers), buffer
                            )
                        )
        else:
            assert host.layout == "page_first_direct"
            page_pools.append(pool_index)
            # View layer-first GPU KV as contiguous layer-page units without copying.
            device_kv_rows = tuple(
                [buf.unflatten(0, (-1, host.page_size)) for buf in layers]
                for layers in (device.k_buffer, device.v_buffer)
            )
            buffers = [
                (
                    host.kv_buffer,
                    (host.k_data_refs, host.v_data_refs),
                    device_kv_rows,
                )
            ]
            if isinstance(host, MHATokenToKVPoolMXFP8Host):
                buffers.extend(
                    [
                        (
                            host.k_scale_host,
                            (host.k_scale_host_layers,),
                            (host.k_scale_device_rows,),
                        ),
                        (
                            host.v_scale_host,
                            (host.v_scale_host_layers,),
                            (host.v_scale_device_rows,),
                        ),
                    ]
                )
            for buffer, host_layers, device_layers in buffers:
                for block, layers in enumerate(layer_blocks):
                    if layers:
                        blocks[block].append(
                            _DirectRows.create(
                                pool_index,
                                [buf[layer] for layer in layers for buf in host_layers],
                                [
                                    buf[layer]
                                    for layer in layers
                                    for buf in device_layers
                                ],
                                buffer,
                            )
                        )
    return DirectTransfer(blocks, block_ends, tuple(page_pools))
