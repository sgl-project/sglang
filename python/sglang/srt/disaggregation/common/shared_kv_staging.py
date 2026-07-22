# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch

from sglang.srt.disaggregation.common.utils import group_concurrent_contiguous
from sglang.srt.mem_cache.shared_kv.transfer import OwnerShardedTransferBuffer


@dataclass(frozen=True)
class _PackedOwnerDescriptor:
    dst_ptr_index: int
    staging_offset: int
    item_bytes: int
    owner_page_bytes: int
    geometry_key: tuple[int, int, int]
    item_positions: npt.NDArray[np.int64]
    subpages: npt.NDArray[np.int64]


@dataclass(frozen=True)
class _OwnerRowsToPack:
    rows: torch.Tensor
    source_rows: npt.NDArray[np.int64]
    staging_offset: int
    owner_page_bytes: int
    rank_stride_owner_pages: int | None


@dataclass
class OwnerShardedStagingCache:
    """Request-local packed owner pages retained for destination fan-out."""

    populated: bool = False
    staging_ptr: int | None = None
    source_indices: tuple[int, ...] = ()
    cp_rank: int = -1
    cp_size: int = 0
    descriptor_count: int = 0
    packed_bytes: int = 0
    descriptors: tuple[_PackedOwnerDescriptor, ...] = ()


def _source_owner_page_rows(
    descriptor: OwnerShardedTransferBuffer,
    logical_pages: npt.NDArray[np.int64],
    *,
    cp_rank: int,
    cp_size: int,
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.int64]]:
    rank_stride = descriptor.rank_stride_owner_pages
    if rank_stride is None:
        selected = logical_pages % cp_size == cp_rank
        return selected, logical_pages[selected] // cp_size

    selected = np.ones(logical_pages.shape, dtype=np.bool_)
    owners = logical_pages % cp_size
    owner_rows = logical_pages // cp_size
    return selected, owners * rank_stride + owner_rows


def _cached_transfer_blocks(
    cache: OwnerShardedStagingCache,
    dst_ptrs: list[int],
    dst_indices: npt.NDArray[np.int64],
) -> list[tuple[int, int, int]]:
    transfer_blocks: list[tuple[int, int, int]] = []
    geometry_runs: dict[tuple[int, int, int], tuple[tuple[int, int, int], ...]] = {}
    assert cache.staging_ptr is not None
    for plan in cache.descriptors:
        runs = geometry_runs.get(plan.geometry_key)
        if runs is None:
            dst_byte_offsets = (
                dst_indices[plan.item_positions] * plan.item_bytes
                + plan.subpages * plan.owner_page_bytes
            )
            src_groups, dst_groups = group_concurrent_contiguous(
                np.arange(plan.item_positions.size, dtype=np.int64),
                dst_byte_offsets,
            )
            runs = tuple(
                (
                    int(src_group[0]) * plan.owner_page_bytes,
                    int(dst_group[0]),
                    len(src_group) * plan.owner_page_bytes,
                )
                for src_group, dst_group in zip(src_groups, dst_groups)
            )
            geometry_runs[plan.geometry_key] = runs
        transfer_blocks.extend(
            (
                cache.staging_ptr + plan.staging_offset + src_byte_offset,
                int(dst_ptrs[plan.dst_ptr_index]) + dst_byte_offset,
                size,
            )
            for src_byte_offset, dst_byte_offset, size in runs
        )
    return transfer_blocks


def _pack_owner_rows(
    rows_to_pack: list[_OwnerRowsToPack],
    staging_buffer,
    *,
    cp_size: int,
) -> None:
    group_start = 0
    while group_start < len(rows_to_pack):
        first = rows_to_pack[group_start]
        rank_stride = first.rank_stride_owner_pages
        layer_rows = (
            first.rows.shape[0] - (cp_size - 1) * rank_stride
            if rank_stride is not None
            else 0
        )
        per_layer_bytes = first.source_rows.size * first.owner_page_bytes
        group_end = group_start + 1
        if layer_rows > 0:
            storage_ptr = first.rows.untyped_storage().data_ptr()
            while group_end < len(rows_to_pack):
                candidate = rows_to_pack[group_end]
                layer_index = group_end - group_start
                if (
                    candidate.source_rows is not first.source_rows
                    or candidate.owner_page_bytes != first.owner_page_bytes
                    or candidate.rank_stride_owner_pages != rank_stride
                    or candidate.rows.shape != first.rows.shape
                    or candidate.rows.dtype != first.rows.dtype
                    or candidate.rows.device != first.rows.device
                    or candidate.rows.untyped_storage().data_ptr() != storage_ptr
                    or candidate.rows.data_ptr()
                    != first.rows.data_ptr()
                    + layer_index * layer_rows * first.owner_page_bytes
                    or candidate.staging_offset
                    != first.staging_offset + layer_index * per_layer_bytes
                ):
                    break
                group_end += 1

        layer_count = group_end - group_start
        index = torch.as_tensor(
            first.source_rows, dtype=torch.long, device=first.rows.device
        )
        if layer_count > 1:
            layered_rows = first.rows.as_strided(
                (layer_count, first.rows.shape[0], first.owner_page_bytes),
                (layer_rows * first.owner_page_bytes, first.owner_page_bytes, 1),
            )
            packed = staging_buffer.buffer[
                first.staging_offset : first.staging_offset
                + layer_count * per_layer_bytes
            ].view(layer_count, first.source_rows.size, first.owner_page_bytes)
            torch.index_select(layered_rows, 1, index, out=packed)
        else:
            packed = staging_buffer.buffer[
                first.staging_offset : first.staging_offset + per_layer_bytes
            ].view(first.source_rows.size, first.owner_page_bytes)
            torch.index_select(first.rows, 0, index, out=packed)
        group_start = group_end


def send_owner_sharded_staged(
    transfer: Callable[[str, list[tuple[int, int, int]]], int],
    session_id: str,
    src_buffers: list[OwnerShardedTransferBuffer],
    logical_src_indices: npt.NDArray[np.int32],
    dst_ptrs: list[int],
    logical_dst_indices: npt.NDArray[np.int32],
    cp_rank: int,
    cp_size: int,
    staging_buffer,
    cache: OwnerShardedStagingCache | None = None,
) -> int:
    """Pack this CP rank's owner pages and transfer them into ordinary KV items."""
    if staging_buffer is None:
        raise RuntimeError("shared PD transfer staging buffer is unavailable")
    if cp_size <= 0 or cp_rank < 0 or cp_rank >= cp_size:
        raise ValueError(f"invalid CP rank {cp_rank} for CP size {cp_size}")

    src_indices = np.asarray(logical_src_indices, dtype=np.int64)
    dst_indices = np.asarray(logical_dst_indices, dtype=np.int64)
    if src_indices.size != dst_indices.size:
        raise ValueError(
            "shared PD transfer requires equal source and destination index counts"
        )
    if len(src_buffers) != len(dst_ptrs):
        raise ValueError("shared PD transfer buffer metadata is inconsistent")
    if src_indices.size == 0:
        return 0

    staging_size = staging_buffer.get_size()
    staging_ptr = staging_buffer.get_ptr()

    if cache is not None and cache.populated:
        if cache.staging_ptr != staging_ptr:
            raise RuntimeError("shared PD staging cache belongs to another buffer")
        if cache.source_indices != tuple(int(index) for index in src_indices):
            raise RuntimeError("shared PD staging cache source indices changed")
        if cache.cp_rank != cp_rank or cache.cp_size != cp_size:
            raise RuntimeError("shared PD staging cache CP geometry changed")
        if cache.descriptor_count != len(src_buffers):
            raise RuntimeError("shared PD staging cache descriptor count changed")
        transfer_blocks = _cached_transfer_blocks(cache, dst_ptrs, dst_indices)
        return transfer(session_id, transfer_blocks)

    if cache is not None:
        packed_descriptors: list[_PackedOwnerDescriptor] = []
        rows_to_pack: list[_OwnerRowsToPack] = []
        packed_bytes = 0
        item_positions = np.arange(src_indices.size, dtype=np.int64)
        source_plans = {}
        for descriptor_index, descriptor in enumerate(src_buffers):
            source_geometry = (
                descriptor.owner_pages_per_item,
                descriptor.rank_stride_owner_pages,
            )
            source_plan = source_plans.get(source_geometry)
            if source_plan is None:
                subpages = np.arange(descriptor.owner_pages_per_item, dtype=np.int64)
                logical_pages = (
                    src_indices[:, None] * descriptor.owner_pages_per_item + subpages
                ).reshape(-1)
                expanded_item_positions = np.repeat(
                    item_positions, descriptor.owner_pages_per_item
                )
                expanded_subpages = np.tile(subpages, src_indices.size)
                selected, source_rows = _source_owner_page_rows(
                    descriptor,
                    logical_pages,
                    cp_rank=cp_rank,
                    cp_size=cp_size,
                )
                source_plan = (
                    expanded_item_positions[selected],
                    expanded_subpages[selected],
                    source_rows,
                )
                source_plans[source_geometry] = source_plan
            selected_item_positions, selected_subpages, source_rows = source_plan
            if source_rows.size == 0:
                continue
            rows = descriptor.owner_page_rows()
            if int(source_rows.max()) >= rows.shape[0]:
                raise IndexError(
                    "shared PD owner-local row exceeds the source tensor geometry"
                )
            descriptor_bytes = source_rows.size * descriptor.owner_page_bytes
            if packed_bytes + descriptor_bytes > staging_size:
                break
            packed_descriptors.append(
                _PackedOwnerDescriptor(
                    dst_ptr_index=descriptor_index,
                    staging_offset=packed_bytes,
                    item_bytes=descriptor.item_bytes,
                    owner_page_bytes=descriptor.owner_page_bytes,
                    geometry_key=(
                        descriptor.owner_pages_per_item,
                        descriptor.item_bytes,
                        descriptor.owner_page_bytes,
                    ),
                    item_positions=selected_item_positions,
                    subpages=selected_subpages,
                )
            )
            rows_to_pack.append(
                _OwnerRowsToPack(
                    rows=rows,
                    source_rows=source_rows,
                    staging_offset=packed_bytes,
                    owner_page_bytes=descriptor.owner_page_bytes,
                    rank_stride_owner_pages=descriptor.rank_stride_owner_pages,
                )
            )
            packed_bytes += descriptor_bytes
        else:
            _pack_owner_rows(rows_to_pack, staging_buffer, cp_size=cp_size)

            if staging_buffer.buffer.is_cuda:
                torch.cuda.current_stream(staging_buffer.buffer.device).synchronize()

            cache.populated = True
            cache.staging_ptr = staging_ptr
            cache.source_indices = tuple(int(index) for index in src_indices)
            cache.cp_rank = cp_rank
            cache.cp_size = cp_size
            cache.descriptor_count = len(src_buffers)
            cache.packed_bytes = packed_bytes
            cache.descriptors = tuple(packed_descriptors)
            transfer_blocks = _cached_transfer_blocks(cache, dst_ptrs, dst_indices)
            return transfer(session_id, transfer_blocks)

    staging_cursor = 0
    transfer_blocks: list[tuple[int, int, int]] = []

    def flush_staging() -> int:
        nonlocal staging_cursor
        if not transfer_blocks:
            return 0
        if staging_buffer.buffer.is_cuda:
            torch.cuda.current_stream(staging_buffer.buffer.device).synchronize()
        status = transfer(session_id, transfer_blocks)
        transfer_blocks.clear()
        staging_cursor = 0
        return status

    for descriptor, dst_ptr in zip(src_buffers, dst_ptrs):
        subpages = np.arange(descriptor.owner_pages_per_item, dtype=np.int64)
        logical_pages = (
            src_indices[:, None] * descriptor.owner_pages_per_item + subpages
        ).reshape(-1)
        expanded_dst_items = np.repeat(dst_indices, descriptor.owner_pages_per_item)
        expanded_subpages = np.tile(subpages, src_indices.size)
        selected, source_rows = _source_owner_page_rows(
            descriptor,
            logical_pages,
            cp_rank=cp_rank,
            cp_size=cp_size,
        )
        dst_byte_offsets = (
            expanded_dst_items[selected] * descriptor.item_bytes
            + expanded_subpages[selected] * descriptor.owner_page_bytes
        )
        if source_rows.size == 0:
            continue

        rows = descriptor.owner_page_rows()
        if int(source_rows.max()) >= rows.shape[0]:
            raise IndexError(
                "shared PD owner-local row exceeds the source tensor geometry"
            )
        if staging_size < descriptor.owner_page_bytes:
            raise RuntimeError(
                "shared PD staging buffer is smaller than one owner page "
                f"({descriptor.owner_page_bytes} bytes)"
            )

        start = 0
        while start < source_rows.size:
            rows_per_chunk = (
                staging_size - staging_cursor
            ) // descriptor.owner_page_bytes
            if rows_per_chunk == 0:
                status = flush_staging()
                if status != 0:
                    return status
                continue

            end = min(start + rows_per_chunk, source_rows.size)
            chunk_rows = source_rows[start:end]
            chunk_dst_offsets = dst_byte_offsets[start:end]
            index = torch.as_tensor(chunk_rows, dtype=torch.long, device=rows.device)
            packed = staging_buffer.buffer[
                staging_cursor : staging_cursor
                + (end - start) * descriptor.owner_page_bytes
            ].view(end - start, descriptor.owner_page_bytes)
            torch.index_select(rows, 0, index, out=packed)

            src_groups, dst_groups = group_concurrent_contiguous(
                np.arange(end - start, dtype=np.int64),
                chunk_dst_offsets,
            )
            transfer_blocks.extend(
                (
                    staging_buffer.get_ptr()
                    + staging_cursor
                    + int(src_group[0]) * descriptor.owner_page_bytes,
                    int(dst_ptr) + int(dst_group[0]),
                    len(src_group) * descriptor.owner_page_bytes,
                )
                for src_group, dst_group in zip(src_groups, dst_groups)
            )
            staging_cursor += (end - start) * descriptor.owner_page_bytes
            start = end

    return flush_staging()
