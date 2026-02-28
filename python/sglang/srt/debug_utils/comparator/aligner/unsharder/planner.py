from collections import defaultdict
from typing import NamedTuple, Optional

from sglang.srt.debug_utils.comparator.aligner.unsharder.types import (
    AxisInfo,
    ConcatParams,
    CpThdConcatParams,
    PickParams,
    UnsharderParams,
    UnsharderPlan,
)
from sglang.srt.debug_utils.comparator.dims import (
    TOKEN_DIM_NAME,
    DimSpec,
    ParallelAxis,
)

# _CoordsList[tensor_index][axis] =
#     the axis_rank (shard position) of the tensor_index-th tensor along `axis`
#     (e.g. coords[2] = {TP: 3} means tensor 2 is the 3rd shard in TP axis)
_CoordsList = list[dict[ParallelAxis, int]]


class _GroupResult(NamedTuple):
    groups: list[list[int]]
    projected_coords: _CoordsList


def compute_unsharder_plan(
    dim_specs: list[DimSpec],
    parallel_infos: list[dict[ParallelAxis, AxisInfo]],
    *,
    thd_global_seq_lens: Optional[list[int]] = None,
) -> list[UnsharderPlan]:
    if not parallel_infos:
        raise ValueError("parallel_infos must not be empty")

    sharded_axis_infos: dict[ParallelAxis, DimSpec] = {
        spec.parallel: spec for spec in dim_specs if spec.parallel is not None
    }
    sharded_axes_raw: set[ParallelAxis] = set(sharded_axis_infos)

    all_axes: set[ParallelAxis] = {axis for info in parallel_infos for axis in info}

    # axis annotated in dims but absent from all parallel_infos -> axis_size=1, skip
    sharded_axes: set[ParallelAxis] = sharded_axes_raw & all_axes
    sharded_axis_infos = {
        k: v for k, v in sharded_axis_infos.items() if k in sharded_axes
    }
    replicated_axes: set[ParallelAxis] = all_axes - sharded_axes

    if not sharded_axes and not replicated_axes:
        return []

    _validate(
        axes_to_validate=sharded_axes | replicated_axes,
        parallel_infos=parallel_infos,
    )

    current_coords: _CoordsList = [
        {axis: info[axis].axis_rank for axis in sharded_axes | replicated_axes}
        for info in parallel_infos
    ]

    axis_and_params: list[tuple[ParallelAxis, UnsharderParams]] = [
        (axis, PickParams()) for axis in sorted(replicated_axes, key=lambda a: a.value)
    ] + [
        (
            axis,
            _resolve_unshard_params(
                spec=spec,
                parallel_infos=parallel_infos,
                thd_global_seq_lens=thd_global_seq_lens,
            ),
        )
        for axis, spec in sharded_axis_infos.items()
    ]

    plans: list[UnsharderPlan] = []
    for axis, params in axis_and_params:
        result = _group_and_project(
            current_coords=current_coords,
            target_axis=axis,
        )
        plans.append(UnsharderPlan(axis=axis, params=params, groups=result.groups))
        current_coords = result.projected_coords

    return plans


def _validate(
    *,
    axes_to_validate: set[ParallelAxis],
    parallel_infos: list[dict[ParallelAxis, AxisInfo]],
) -> None:
    """Check that every rank has all axes, sizes are consistent, and ranks are complete."""
    axis_sizes: dict[ParallelAxis, int] = {}

    for world_rank, parallel_info in enumerate(parallel_infos):
        for axis in axes_to_validate:
            if axis not in parallel_info:
                raise ValueError(
                    f"world_rank={world_rank} missing parallel_info for "
                    f"axis {axis.value!r}"
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


def _resolve_unshard_params(
    *,
    spec: DimSpec,
    parallel_infos: list[dict[ParallelAxis, AxisInfo]],
    thd_global_seq_lens: Optional[list[int]] = None,
) -> UnsharderParams:
    if spec.reduction is not None:
        raise NotImplementedError(
            f"Unshard for reduction={spec.reduction} not yet implemented (Phase 2)"
        )

    if (
        spec.name == TOKEN_DIM_NAME
        and spec.parallel == ParallelAxis.CP
        and thd_global_seq_lens is not None
    ):
        if spec.parallel is None:
            raise ValueError(
                f"THD unshard requires a parallel axis on dim '{spec.name}', but got None"
            )
        axis_size: int = parallel_infos[0][spec.parallel].axis_size
        for s in thd_global_seq_lens:
            if s % axis_size != 0:
                raise ValueError(
                    f"THD seq_len {s} is not divisible by cp_size {axis_size}. "
                    f"Sequences must be padded to a multiple of cp_size for CP zigzag."
                )
        seq_lens_per_rank: list[int] = [s // axis_size for s in thd_global_seq_lens]
        return CpThdConcatParams(
            dim_name=spec.name, seq_lens_per_rank=seq_lens_per_rank
        )

    return ConcatParams(dim_name=spec.name)
