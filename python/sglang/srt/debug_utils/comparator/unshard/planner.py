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

    sharded_axes = _extract_sharded_axes(dim_specs)
    if not sharded_axes:
        return []

    current_coords = _build_and_validate_coords(
        sharded_axes=sharded_axes,
        parallel_infos=parallel_infos,
    )

    plans: list[UnshardPlan] = []
    for axis_name, (dim_idx, spec) in sharded_axes.items():
        groups = _compute_groups(
            current_coords=current_coords,
            target_axis=axis_name,
        )

        plans.append(UnshardPlan(
            axis=spec.parallel,
            params=_resolve_unshard_params(spec=spec, dim_index=dim_idx),
            groups=groups,
        ))

        current_coords = _project_coords(
            groups=groups,
            current_coords=current_coords,
            removed_axis=axis_name,
        )

    return plans


def _extract_sharded_axes(
    dim_specs: list[DimSpec],
) -> dict[ParallelAxis, tuple[int, DimSpec]]:
    result: dict[ParallelAxis, tuple[int, DimSpec]] = {}
    for dim_idx, spec in enumerate(dim_specs):
        if spec.parallel is not None:
            result[spec.parallel] = (dim_idx, spec)
    return result


def _build_and_validate_coords(
    *,
    sharded_axes: dict[ParallelAxis, tuple[int, DimSpec]],
    parallel_infos: list[dict[ParallelAxis, AxisInfo]],
) -> dict[int, dict[ParallelAxis, int]]:
    coords_by_index: dict[int, dict[ParallelAxis, int]] = {}
    axis_sizes: dict[ParallelAxis, int] = {}

    for world_rank, parallel_info in enumerate(parallel_infos):
        coords: dict[ParallelAxis, int] = {}
        for axis_name in sharded_axes:
            if axis_name not in parallel_info:
                raise ValueError(
                    f"world_rank={world_rank} missing parallel_info for "
                    f"sharded axis {axis_name.value!r}"
                )

            axis_info = parallel_info[axis_name]
            coords[axis_name] = axis_info.axis_rank

            if axis_name not in axis_sizes:
                axis_sizes[axis_name] = axis_info.axis_size
            elif axis_info.axis_size != axis_sizes[axis_name]:
                raise ValueError(
                    f"Inconsistent axis_size for {axis_name.value}: "
                    f"expected {axis_sizes[axis_name]}, got {axis_info.axis_size} "
                    f"at world_rank={world_rank}"
                )

        coords_by_index[world_rank] = coords

    for axis_name, expected_size in axis_sizes.items():
        seen_ranks = {coords[axis_name] for coords in coords_by_index.values()}
        if seen_ranks != set(range(expected_size)):
            raise ValueError(
                f"axis_rank coverage for {axis_name.value} is incomplete: "
                f"got {sorted(seen_ranks)}, expected 0..{expected_size - 1}"
            )

    return coords_by_index


def _compute_groups(
    *,
    current_coords: dict[int, dict[ParallelAxis, int]],
    target_axis: ParallelAxis,
) -> list[list[int]]:
    buckets: dict[tuple[tuple[ParallelAxis, int], ...], list[tuple[int, int]]] = defaultdict(list)

    for idx, coords in current_coords.items():
        grouping_key = _coords_key_excluding(coords, excluded_axis=target_axis)
        axis_rank = coords[target_axis]
        buckets[grouping_key].append((axis_rank, idx))

    groups: list[list[int]] = []
    for key in sorted(buckets.keys()):
        entries = buckets[key]
        entries.sort(key=lambda pair: pair[0])
        groups.append([idx for _, idx in entries])

    return groups


def _coords_key_excluding(
    coords: dict[ParallelAxis, int],
    *,
    excluded_axis: ParallelAxis,
) -> tuple[tuple[ParallelAxis, int], ...]:
    return tuple(sorted(
        ((k, v) for k, v in coords.items() if k != excluded_axis),
        key=lambda pair: pair[0].value,
    ))


def _project_coords(
    *,
    groups: list[list[int]],
    current_coords: dict[int, dict[ParallelAxis, int]],
    removed_axis: ParallelAxis,
) -> dict[int, dict[ParallelAxis, int]]:
    return {
        new_idx: {
            k: v for k, v in current_coords[group[0]].items()
            if k != removed_axis
        }
        for new_idx, group in enumerate(groups)
    }


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
