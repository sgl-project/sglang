from typing import Optional

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
) -> Optional[UnshardPlan]:
    if not parallel_infos:
        raise ValueError("parallel_infos must not be empty")

    sharded_axes: dict[ParallelAxis, tuple[int, DimSpec]] = {}
    for dim_idx, spec in enumerate(dim_specs):
        if spec.parallel is not None:
            sharded_axes[spec.parallel] = (dim_idx, spec)

    if len(sharded_axes) > 1:
        raise NotImplementedError(
            f"Multi-axis unshard is not supported. "
            f"Got {len(sharded_axes)} sharded axes: {sorted(a.value for a in sharded_axes)}"
        )

    if not sharded_axes:
        return None

    axis_name, (dim_idx, spec) = next(iter(sharded_axes.items()))

    expected_size: Optional[int] = None
    rank_to_world: dict[int, int] = {}

    for world_rank, pinfo in enumerate(parallel_infos):
        if axis_name not in pinfo:
            continue

        ainfo = pinfo[axis_name]

        if expected_size is None:
            expected_size = ainfo.axis_size
        elif ainfo.axis_size != expected_size:
            raise ValueError(
                f"Inconsistent axis_size for {axis_name.value}: "
                f"expected {expected_size}, got {ainfo.axis_size} "
                f"at world_rank={world_rank}"
            )

        rank_to_world.setdefault(ainfo.axis_rank, world_rank)

    if expected_size is None:
        raise ValueError(f"No parallel_info found for sharded axis {axis_name.value!r}")

    if set(rank_to_world.keys()) != set(range(expected_size)):
        raise ValueError(
            f"axis_rank coverage for {axis_name.value} is incomplete: "
            f"got {sorted(rank_to_world.keys())}, expected 0..{expected_size - 1}"
        )

    return UnshardPlan(
        axis=spec.parallel,
        params=_resolve_unshard_params(spec=spec, dim_index=dim_idx),
        world_ranks_by_axis_rank=[rank_to_world[i] for i in range(expected_size)],
    )


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
