from collections import defaultdict

from sglang.srt.debug_utils.comparator.dims import DimSpec, Ordering, ParallelAxis
from sglang.srt.debug_utils.comparator.unshard.types import (
    AxisInfo,
    ConcatParams,
    UnshardParams,
    UnshardPlan,
)


def compute_unshard_plan(
    dim_specs: list[DimSpec],
    parallel_infos: list[dict[ParallelAxis, AxisInfo]],
) -> list[UnshardPlan]:
    if not parallel_infos:
        raise ValueError("parallel_infos must not be empty")

    sharded_axes: dict[ParallelAxis, tuple[int, DimSpec]] = {}
    for dim_idx, spec in enumerate(dim_specs):
        if spec.parallel is not None:
            sharded_axes[spec.parallel] = (dim_idx, spec)

    if not sharded_axes:
        return []

    coord_map: dict[int, dict[ParallelAxis, int]] = {}
    axis_sizes: dict[ParallelAxis, int] = {}

    for world_rank, pinfo in enumerate(parallel_infos):
        coords: dict[ParallelAxis, int] = {}
        for axis_name in sharded_axes:
            if axis_name not in pinfo:
                raise ValueError(
                    f"world_rank={world_rank} missing parallel_info for "
                    f"sharded axis {axis_name.value!r}"
                )

            ainfo = pinfo[axis_name]
            coords[axis_name] = ainfo.axis_rank

            if axis_name not in axis_sizes:
                axis_sizes[axis_name] = ainfo.axis_size
            elif ainfo.axis_size != axis_sizes[axis_name]:
                raise ValueError(
                    f"Inconsistent axis_size for {axis_name.value}: "
                    f"expected {axis_sizes[axis_name]}, got {ainfo.axis_size} "
                    f"at world_rank={world_rank}"
                )

        coord_map[world_rank] = coords

    for axis_name, expected_size in axis_sizes.items():
        seen_ranks = {coords[axis_name] for coords in coord_map.values()}
        if seen_ranks != set(range(expected_size)):
            raise ValueError(
                f"axis_rank coverage for {axis_name.value} is incomplete: "
                f"got {sorted(seen_ranks)}, expected 0..{expected_size - 1}"
            )

    current_indices = list(range(len(parallel_infos)))
    current_coords: dict[int, dict[ParallelAxis, int]] = {
        idx: dict(coord_map[idx]) for idx in current_indices
    }
    remaining_axes = list(sharded_axes.keys())

    plans: list[UnshardPlan] = []
    for axis_name in remaining_axes:
        dim_idx, spec = sharded_axes[axis_name]
        groups = _compute_groups(
            current_indices=current_indices,
            current_coords=current_coords,
            target_axis=axis_name,
        )

        plans.append(UnshardPlan(
            axis=spec.parallel,
            params=_resolve_unshard_params(spec=spec, dim_index=dim_idx),
            groups=groups,
        ))

        new_indices: list[int] = []
        new_coords: dict[int, dict[ParallelAxis, int]] = {}
        for new_idx, group in enumerate(groups):
            representative = group[0]
            remaining = {
                k: v for k, v in current_coords[representative].items()
                if k != axis_name
            }
            new_coords[new_idx] = remaining
            new_indices.append(new_idx)

        current_indices = new_indices
        current_coords = new_coords

    return plans


def _compute_groups(
    current_indices: list[int],
    current_coords: dict[int, dict[ParallelAxis, int]],
    target_axis: ParallelAxis,
) -> list[list[int]]:
    buckets: dict[tuple[tuple[ParallelAxis, int], ...], list[tuple[int, int]]] = defaultdict(list)

    for idx in current_indices:
        coords = current_coords[idx]
        other_key = tuple(
            sorted(
                ((k, v) for k, v in coords.items() if k != target_axis),
                key=lambda pair: pair[0].value,
            )
        )
        axis_rank = coords[target_axis]
        buckets[other_key].append((axis_rank, idx))

    groups: list[list[int]] = []
    for key in sorted(buckets.keys()):
        entries = buckets[key]
        entries.sort(key=lambda pair: pair[0])
        groups.append([idx for _, idx in entries])

    return groups


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
