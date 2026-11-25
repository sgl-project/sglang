"""CPU exchange of the routing rows selected by the Ascend EPLB planner."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from sglang.srt.hardware_backend.npu.eplb.placement import MigrationPlacement


class PlacementRows(NamedTuple):
    """One row per changed layer, in installation order on this rank."""

    layers: tuple[int, ...]
    physical_to_logical: torch.Tensor
    logical_to_physical: torch.Tensor
    local_dispatch: torch.Tensor | None


def populated_replica_width(logical: torch.Tensor) -> int:
    # Keep any holes in the table and trim only columns that are all padding.
    occupied = (logical != -1).any(dim=(0, 1)).nonzero()
    return int(occupied[-1, 0]) + 1 if occupied.numel() else 0


def _pack_placement(
    placement: MigrationPlacement,
    physical_shape: torch.Size,
    num_logical: int,
    num_ranks: int | None,
) -> tuple[torch.Tensor, int, list[torch.Tensor] | None]:
    layers = placement.layers
    physical = placement.physical_to_logical
    logical = placement.logical_to_physical
    static_dispatch = placement.static_dispatch
    num_layers, num_slots = physical_shape
    count = len(layers)
    if (
        physical.shape != (count, num_slots)
        or logical.shape[:-1] != (count, num_logical)
        or any(layer < 0 or layer >= num_layers for layer in layers)
    ):
        raise ValueError("EPLB placement changed the routing geometry")

    ids = torch.tensor(layers, dtype=torch.int64, device="cpu")
    width = populated_replica_width(logical)
    fields = (ids, physical, logical[..., :width])
    payload = torch.cat([field.reshape(-1) for field in fields]).long()
    static_rows_by_rank = None
    if static_dispatch is not None:
        if static_dispatch.shape != (count, num_logical, num_ranks):
            raise ValueError("EPLB placement changed the static routing geometry")
        static_rows_by_rank = list(
            static_dispatch.permute(2, 0, 1).long().contiguous().unbind()
        )
    return payload, width, static_rows_by_rank


def allocate_plan_header() -> torch.Tensor:
    """Allocate the reusable CPU header before startup agreement."""
    return torch.empty(3, dtype=torch.int64, device="cpu")


def broadcast_placement(
    placement: MigrationPlacement | None,
    error: str | None,
    rank: int,
    group: dist.ProcessGroup,
    physical_shape: torch.Size,
    num_logical: int,
    *,
    header: torch.Tensor,
) -> PlacementRows | None:
    """Broadcast changed rows; scatter only the static picks each rank uses."""
    # Header: [changed layers (-1 on failure), replica width, static dispatch flag].
    static_rows_by_rank = None
    if rank == 0:
        header.zero_()
        if error is None and placement is not None:
            try:
                num_ranks = (
                    dist.get_world_size(group)
                    if placement.static_dispatch is not None
                    else None
                )
                payload, width, static_rows_by_rank = _pack_placement(
                    placement, physical_shape, num_logical, num_ranks
                )
                header[0] = len(placement.layers)
                header[1] = width
                header[2] = int(static_rows_by_rank is not None)
            except Exception as exc:
                error = str(exc)
        if error is not None:
            header[0] = -1
    dist.broadcast(header, src=0, group=group)
    layer_count, replica_width, has_static_dispatch = header.tolist()
    if layer_count < 0:
        message = [error]
        dist.broadcast_object_list(message, src=0, group=group)
        raise RuntimeError(f"EPLB CPU planner failed: {message[0]}")
    if layer_count == 0:
        return None

    num_slots = physical_shape[1]
    field_sizes = (
        layer_count,
        layer_count * num_slots,
        layer_count * num_logical * replica_width,
    )
    local_dispatch = None
    receive_error = None
    try:
        if rank != 0:
            payload = torch.empty(sum(field_sizes), dtype=torch.int64, device="cpu")
        if has_static_dispatch:
            local_dispatch = torch.empty(
                (layer_count, num_logical), dtype=torch.int64, device="cpu"
            )
    except Exception as exc:
        receive_error = exc

    # The header was decoded above. Reuse its storage to agree on both
    # allocations before any rank enters the payload broadcast or scatter.
    failed_rank = header[:1]
    failed_rank.fill_(rank if receive_error is not None else -1)
    dist.all_reduce(failed_rank, op=dist.ReduceOp.MAX, group=group)
    failed = int(failed_rank.item())
    if failed >= 0:
        raise RuntimeError(
            f"Async EPLB rank {failed} could not allocate placement receive buffers"
        ) from receive_error

    dist.broadcast(payload, src=0, group=group)
    if has_static_dispatch:
        assert local_dispatch is not None
        dist.scatter(local_dispatch, static_rows_by_rank, src=0, group=group)
    layer_ids, physical, logical = payload.split(field_sizes)
    return PlacementRows(
        layers=tuple(layer_ids.tolist()),
        physical_to_logical=physical.view(layer_count, num_slots),
        logical_to_physical=logical.view(layer_count, num_logical, replica_width),
        local_dispatch=local_dispatch,
    )
