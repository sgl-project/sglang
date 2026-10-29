"""Retraction copies preserve every part of the plan's entries and fixed rows."""

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
        entries = allocator.arena.entries(band.sub_pool_name).index_select(
            0, physical[positions]
        )
        snapshot = BandSnapshot(
            positions=positions.cpu(), entries=entries.to("cpu", non_blocking=True)
        )
        memory = allocator.arena.page_parts.get(band.sub_pool_name)
        if memory is not None:
            reps = _page_positions(ids, band.page_size)
            pages = band.ids.virtual_to_physical[ids[reps] // band.page_size]
            valid = pages > 0
            snapshot.page_positions = reps[valid].cpu()
            for part in memory.parts:
                snapshot.page_parts[part.name] = (
                    memory.tensors[part.name]
                    .index_select(part.row_axis, pages[valid])
                    .to("cpu", non_blocking=True)
                )
        bands[band.sub_pool_name] = snapshot
    request = {}
    storage = allocator.storage
    if storage is not None and request_row is not None:
        memory = storage.request_memory
        if memory is not None:
            for part in memory.parts:
                request[part.name] = (
                    memory.tensors[part.name]
                    .select(part.row_axis, request_row)
                    .clone()
                    .to("cpu", non_blocking=True)
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
        allocator.arena.entries(band.sub_pool_name).index_copy_(
            0,
            destinations[valid],
            saved.entries[valid.cpu()].to(physical.device, non_blocking=True),
        )
        memory = allocator.arena.page_parts.get(band.sub_pool_name)
        if memory is not None and saved.page_positions is not None:
            reps = saved.page_positions.to(physical.device)
            pages = band.ids.virtual_to_physical[
                ids.reshape(-1)[reps].to(torch.int64) // band.page_size
            ]
            live = pages > 0
            for part in memory.parts:
                values = saved.page_parts[part.name].index_select(
                    part.row_axis, torch.nonzero(live.cpu(), as_tuple=True)[0]
                )
                memory.tensors[part.name].index_copy_(
                    part.row_axis,
                    pages[live],
                    values.to(physical.device, non_blocking=True),
                )
    if snapshot.request_parts:
        if request_row is None:
            raise ValueError("restoring request parts requires the new request row")
        memory = allocator.storage.request_memory
        for part in memory.parts:
            memory.tensors[part.name].select(part.row_axis, request_row).copy_(
                snapshot.request_parts[part.name], non_blocking=True
            )
    current_platform.synchronize()
