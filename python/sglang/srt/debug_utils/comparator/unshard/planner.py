from collections import defaultdict
from typing import NamedTuple

from sglang.srt.debug_utils.comparator.dims import DimSpec, Ordering, ParallelAxis
from sglang.srt.debug_utils.comparator.unshard.types import (
    AxisInfo,
    ConcatParams,
    UnshardParams,
    UnshardPlan,
)

# _CoordsList[tensor_index][axis] =
#     the axis_rank (shard position) of the tensor_index-th tensor along `axis`
#     (e.g. coords[2] = {TP: 3} means tensor 2 is the 3rd shard in TP axis)
_CoordsList = list[dict[ParallelAxis, int]]


class _GroupResult(NamedTuple):
    groups: list[list[int]]
    projected_coords: _CoordsList


def compute_unshard_plan(
    dim_specs: list[DimSpec],
    parallel_infos: list[dict[ParallelAxis, AxisInfo]],
) -> list[UnshardPlan]:
    if not parallel_infos:
        raise ValueError("parallel_infos must not be empty")

    sharded_axis_infos: dict[ParallelAxis, tuple[int, DimSpec]] = {
        spec.parallel: (dim_idx, spec)
        for dim_idx, spec in enumerate(dim_specs)
        if spec.parallel is not None
    }
    if not sharded_axis_infos:
        return []

    _validate(sharded_axes=set(sharded_axis_infos), parallel_infos=parallel_infos)

    current_coords: _CoordsList = [
        {axis: info[axis].axis_rank for axis in sharded_axis_infos}
        for info in parallel_infos
    ]

    plans: list[UnshardPlan] = []
    for axis, (dim_index, spec) in sharded_axis_infos.items():
        result = _group_and_project(
            current_coords=current_coords,
            target_axis=axis,
        )

        plans.append(
            UnshardPlan(
                axis=axis,
                params=_resolve_unshard_params(spec=spec, dim_index=dim_index),
                groups=result.groups,
            )
        )

        current_coords = result.projected_coords

    return plans


def _validate(
    *,
    sharded_axes: set[ParallelAxis],
    parallel_infos: list[dict[ParallelAxis, AxisInfo]],
) -> None:
    """Check that every rank has all sharded axes, sizes are consistent, and ranks are complete."""
    axis_sizes: dict[ParallelAxis, int] = {}

    for world_rank, parallel_info in enumerate(parallel_infos):
        for axis in sharded_axes:
            if axis not in parallel_info:
                raise ValueError(
                    f"world_rank={world_rank} missing parallel_info for "
                    f"sharded axis {axis.value!r}"
                )

            axis_info = parallel_info[axis]
            if axis not in axis_sizes:
                axis_sizes[axis] = axis_info.axis_size
            elif axis_info.axis_size != axis_sizes[axis]:
                raise ValueError(
                    f"Inconsistent axis_size for {axis.value}: "
                    f"expected {axis_sizes[axis]}, got {axis_info.axis_size} "
                    f"at world_rank={world_rank}"
                )

    for axis, expected_size in axis_sizes.items():
        seen_ranks = {info[axis].axis_rank for info in parallel_infos}
        if seen_ranks != set(range(expected_size)):
            raise ValueError(
                f"axis_rank coverage for {axis.value} is incomplete: "
                f"got {sorted(seen_ranks)}, expected 0..{expected_size - 1}"
            )


def _group_and_project(
    *,
    current_coords: _CoordsList,
    target_axis: ParallelAxis,
) -> _GroupResult:
    """Group tensors by other-axes coords, sort within group by target_axis rank."""
    # buckets[coords_excluding_target] = [(axis_rank, tensor_index), ...]
    # e.g. when target_axis=CP: buckets[{(TP,0)}] = [(0, 1), (1, 3)]
    #   means tensor 1 (CP rank 0) and tensor 3 (CP rank 1) share TP rank 0
    buckets: dict[frozenset, list[tuple[int, int]]] = defaultdict(list)

    for idx, coords in enumerate(current_coords):
        key = frozenset((k, v) for k, v in coords.items() if k != target_axis)
        buckets[key].append((coords[target_axis], idx))

    groups: list[list[int]] = []
    projected: _CoordsList = []
    for key in sorted(buckets, key=lambda k: sorted((a.value, v) for a, v in k)):
        entries = sorted(buckets[key])
        groups.append([idx for _, idx in entries])
        projected.append(dict(key))

    return _GroupResult(groups=groups, projected_coords=projected)


def _resolve_unshard_params(*, spec: DimSpec, dim_index: int) -> UnshardParams:
    if spec.reduction is not None:
        raise NotImplementedError(
            f"Unshard for reduction={spec.reduction} not yet implemented (Phase 2)"
        )
    if spec.ordering is not None and spec.ordering != Ordering.NATURAL:
        raise NotImplementedError(
            f"Unshard for ordering={spec.ordering} not yet implemented (Phase 2)"
        )
    return ConcatParams(dim=dim_index)
