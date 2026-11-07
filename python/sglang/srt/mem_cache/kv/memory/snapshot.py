"""Retraction copies preserve every part of the plan's entries and fixed rows."""

from itertools import product
from typing import Optional

import msgspec
import torch

from sglang.srt.platforms import current_platform


class BandSnapshot(msgspec.Struct, kw_only=True):
    positions: torch.Tensor
    entries: torch.Tensor
    page_positions: Optional[torch.Tensor] = None
    page_parts: dict[str, torch.Tensor] = {}


class StorageSnapshot(msgspec.Struct, kw_only=True):
    bands: dict[str, BandSnapshot]
    request_parts: dict[str, torch.Tensor] = {}


def _physical(band, indices, *, state):
    indices = indices.to(torch.int64).reshape(-1)
    return band.ids.translate(indices) if state else band.ids.translate_write(indices)


def _page_positions(indices, page_size):
    # Remember positions in this request, not its old virtual ids: a restore
    # may own different virtual pages as well as different physical pages.
    _, inverse = torch.unique(indices // page_size, sorted=True, return_inverse=True)
    count = 0 if inverse.numel() == 0 else int(inverse.max().item()) + 1
    first = torch.full(
        (count,), indices.numel(), dtype=torch.int64, device=indices.device
    )
    return first.scatter_reduce_(
        0, inverse, torch.arange(indices.numel(), device=indices.device), reduce="amin"
    )


# Retraction runs under memory pressure; a temporary never exceeds this bound.
# This is a workspace policy, independent of token counts and entry geometry.
_COPY_CHUNK_BYTES = 16 << 20


def _copy_tiles(shape, element_size):
    remaining = max(1, _COPY_CHUNK_BYTES // element_size)
    steps = [1] * len(shape)
    for axis in reversed(range(len(shape))):
        steps[axis] = max(1, min(shape[axis], remaining))
        remaining = max(1, remaining // steps[axis])
    for starts in product(*(range(0, n, step) for n, step in zip(shape, steps))):
        yield tuple(
            slice(start, min(start + step, n))
            for start, step, n in zip(starts, steps, shape)
        )


def _gather_to_cpu(tensor, rows, row_axis):
    shape = list(tensor.shape)
    shape[row_axis] = rows.numel()
    result = torch.empty(shape, dtype=tensor.dtype, device="cpu")
    for tile in _copy_tiles(shape, tensor.element_size()):
        source = list(tile)
        source[row_axis] = slice(None)
        values = tensor[tuple(source)].index_select(row_axis, rows[tile[row_axis]])
        result[tile].copy_(values, non_blocking=True)
    return result


def _scatter_from_cpu(tensor, rows, values, value_rows, row_axis):
    shape = list(values.shape)
    shape[row_axis] = rows.numel()
    for tile in _copy_tiles(shape, tensor.element_size()):
        columns = list(tile)
        columns[row_axis] = slice(None)
        source = values[tuple(columns)].index_select(
            row_axis, value_rows[tile[row_axis]]
        )
        tensor[tuple(columns)].index_copy_(
            row_axis,
            rows[tile[row_axis]],
            source.to(tensor.device, non_blocking=True),
        )


def copy_to_cpu(allocator, indices, *, state_indices, request_row):
    current_platform.synchronize()
    bands = {}
    for band in allocator.members:
        state = band.sub_pool_name == allocator.roles.state
        ids = state_indices if state else indices
        if ids is None:
            continue
        ids = ids.reshape(-1)
        physical = _physical(band, ids, state=state)
        positions = torch.nonzero(physical > 0, as_tuple=True)[0]
        snapshot = BandSnapshot(
            positions=positions.cpu(),
            entries=_gather_to_cpu(
                allocator.arena.entries(band.sub_pool_name), physical[positions], 0
            ),
        )
        memory = allocator.arena.page_parts.get(band.sub_pool_name)
        if memory is not None:
            reps = _page_positions(ids, band.page_size)
            pages = band.ids.virtual_to_physical[ids[reps] // band.page_size]
            valid = pages > 0
            snapshot.page_positions = reps[valid].cpu()
            for part in memory.parts:
                snapshot.page_parts[part.name] = _gather_to_cpu(
                    memory.tensors[part.name], pages[valid], part.row_axis
                )
        bands[band.sub_pool_name] = snapshot
    request = {}
    storage = allocator.storage
    if storage is not None and request_row is not None:
        memory = storage.request_memory
        if memory is not None:
            for part in memory.parts:
                tensor = memory.tensors[part.name]
                row = torch.tensor([request_row], device=tensor.device)
                request[part.name] = _gather_to_cpu(tensor, row, part.row_axis).select(
                    part.row_axis, 0
                )
    current_platform.synchronize()
    return StorageSnapshot(bands=bands, request_parts=request)


def restore_from_cpu(allocator, snapshot, indices, *, state_indices, request_row):
    current_platform.synchronize()
    for band in allocator.members:
        saved = snapshot.bands.get(band.sub_pool_name)
        if saved is None:
            continue
        state = band.sub_pool_name == allocator.roles.state
        ids = state_indices if state else indices
        if ids is None:
            raise ValueError("restoring recurrent state requires its new slot ids")
        physical = _physical(band, ids, state=state)
        positions = saved.positions.to(device=physical.device)
        destinations = physical[positions]
        valid = destinations > 0
        # A window may retain less history by restore time. Only its still-bound
        # entries are restored; another band's bytes must never be addressed.
        _scatter_from_cpu(
            allocator.arena.entries(band.sub_pool_name),
            destinations[valid],
            saved.entries,
            torch.nonzero(valid.cpu(), as_tuple=True)[0],
            0,
        )
        memory = allocator.arena.page_parts.get(band.sub_pool_name)
        if memory is not None and saved.page_positions is not None:
            reps = saved.page_positions.to(physical.device)
            pages = band.ids.virtual_to_physical[
                ids.reshape(-1)[reps].to(torch.int64) // band.page_size
            ]
            live = pages > 0
            for part in memory.parts:
                _scatter_from_cpu(
                    memory.tensors[part.name],
                    pages[live],
                    saved.page_parts[part.name],
                    torch.nonzero(live.cpu(), as_tuple=True)[0],
                    part.row_axis,
                )
    if snapshot.request_parts:
        if request_row is None:
            raise ValueError("restoring request parts requires the new request row")
        memory = allocator.storage.request_memory
        for part in memory.parts:
            tensor = memory.tensors[part.name]
            _scatter_from_cpu(
                tensor,
                torch.tensor([request_row], device=tensor.device),
                snapshot.request_parts[part.name].unsqueeze(part.row_axis),
                torch.tensor([0]),
                part.row_axis,
            )
    current_platform.synchronize()
