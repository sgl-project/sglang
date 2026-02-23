from typing import Optional

from sglang.srt.debug_utils.comparator.dims import DimSpec, Ordering
from sglang.srt.debug_utils.comparator.unshard.types import (
    AxisInfo,
    ConcatParams,
    UnshardParams,
    UnshardPlan,
    UnshardStep,
)


def compute_unshard_plan(
    dim_specs: list[DimSpec],
    parallel_infos: list[dict[str, AxisInfo]],
    tensor_name: str = "",
    dims_str: str = "",
) -> UnshardPlan:
    """Compute an unshard plan from dim specs and per-rank parallel info.

    Pure computation — does not load tensor data.
    Validates axis_size consistency and axis_rank coverage.
    """
    if not parallel_infos:
        raise ValueError("parallel_infos must not be empty")

    sharded_axes: dict[str, tuple[int, DimSpec]] = {}
    for dim_idx, spec in enumerate(dim_specs):
        if spec.parallel is not None:
            sharded_axes[spec.parallel.value] = (dim_idx, spec)

    all_axis_names = {k for pinfo in parallel_infos for k in pinfo}

    replicated_axes: dict[str, AxisInfo] = {}
    for axis_name in all_axis_names:
        if axis_name not in sharded_axes:
            for pinfo in parallel_infos:
                if axis_name in pinfo:
                    replicated_axes[axis_name] = AxisInfo(
                        axis_rank=0, axis_size=pinfo[axis_name].axis_size
                    )
                    break

    pick = set(range(len(parallel_infos)))
    for axis_name in replicated_axes:
        pick = {
            wr
            for wr in pick
            if axis_name not in parallel_infos[wr]
            or parallel_infos[wr][axis_name].axis_rank == 0
        }

    steps: list[UnshardStep] = []
    for axis_name, (dim_idx, spec) in sharded_axes.items():
        expected_size: Optional[int] = None
        rank_to_world: dict[int, int] = {}

        for world_rank, pinfo in enumerate(parallel_infos):
            if world_rank not in pick:
                continue
            if axis_name not in pinfo:
                continue

            ainfo = pinfo[axis_name]

            if expected_size is None:
                expected_size = ainfo.axis_size
            elif ainfo.axis_size != expected_size:
                raise ValueError(
                    f"Inconsistent axis_size for {axis_name}: "
                    f"expected {expected_size}, got {ainfo.axis_size} "
                    f"at world_rank={world_rank}"
                )

            rank_to_world.setdefault(ainfo.axis_rank, world_rank)

        if expected_size is None:
            raise ValueError(
                f"No parallel_info found for sharded axis {axis_name!r}"
            )

        if set(rank_to_world.keys()) != set(range(expected_size)):
            raise ValueError(
                f"axis_rank coverage for {axis_name} is incomplete: "
                f"got {sorted(rank_to_world.keys())}, expected 0..{expected_size - 1}"
            )

        steps.append(
            UnshardStep(
                axis=spec.parallel,
                params=_resolve_unshard_params(spec=spec, dim_index=dim_idx),
                world_ranks_by_axis_rank=[
                    rank_to_world[i] for i in range(expected_size)
                ],
            )
        )

    return UnshardPlan(
        tensor_name=tensor_name,
        dims_str=dims_str,
        replicated_axes=replicated_axes,
        steps=steps,
        pick_world_ranks=frozenset(pick),
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
