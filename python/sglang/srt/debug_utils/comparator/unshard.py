from dataclasses import dataclass, field
from typing import Callable, NamedTuple, Optional

import torch

from sglang.srt.debug_utils.comparator.dims import DimSpec, Ordering, ParallelAxis


class AxisInfo(NamedTuple):
    axis_rank: int
    axis_size: int


_PARALLEL_INFO_KEYS = ("sglang_parallel_info", "megatron_parallel_info")

_AXIS_PREFIXES = [e.value for e in ParallelAxis]


def normalize_parallel_info(meta: dict) -> dict[str, AxisInfo]:
    """Extract unified parallel info from dump meta.

    Looks for sglang_parallel_info or megatron_parallel_info in meta.
    Returns e.g. {"tp": AxisInfo(axis_rank=0, axis_size=4), ...}
    for every axis whose size > 1.
    """
    found: list[tuple[str, dict]] = []
    for key in _PARALLEL_INFO_KEYS:
        if key in meta and isinstance(meta[key], dict) and meta[key]:
            found.append((key, meta[key]))

    if len(found) > 1:
        raise ValueError(
            f"Meta contains multiple parallel_info keys: {[k for k, _ in found]}"
        )

    if not found:
        return {}

    _source_key, info = found[0]
    result: dict[str, AxisInfo] = {}
    for prefix in _AXIS_PREFIXES:
        rank_key = f"{prefix}_rank"
        size_key = f"{prefix}_size"
        if rank_key in info and size_key in info:
            axis_size = info[size_key]
            if axis_size > 1:
                result[prefix] = AxisInfo(
                    axis_rank=info[rank_key],
                    axis_size=axis_size,
                )

    return result


@dataclass(frozen=True)
class UnshardStep:
    axis: ParallelAxis
    dim_index: int
    fn: Callable[[list[torch.Tensor], int], torch.Tensor]
    world_ranks_by_axis_rank: list[int]


@dataclass(frozen=True)
class UnshardPlan:
    tensor_name: str
    dims_str: str
    replicated_axes: dict[str, AxisInfo] = field(default_factory=dict)
    steps: list[UnshardStep] = field(default_factory=list)


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

    sharded_axes: dict[str, tuple[int, Optional[DimSpec]]] = {}
    for dim_idx, spec in enumerate(dim_specs):
        if spec.parallel is not None:
            sharded_axes[spec.parallel.value] = (dim_idx, spec)

    all_axis_names: set[str] = set()
    for pinfo in parallel_infos:
        all_axis_names.update(pinfo.keys())

    replicated_axes: dict[str, AxisInfo] = {}
    for axis_name in all_axis_names:
        if axis_name not in sharded_axes:
            for pinfo in parallel_infos:
                if axis_name in pinfo:
                    replicated_axes[axis_name] = AxisInfo(
                        axis_rank=0, axis_size=pinfo[axis_name].axis_size
                    )
                    break

    steps: list[UnshardStep] = []
    for axis_name, (dim_idx, spec) in sharded_axes.items():
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
                    f"Inconsistent axis_size for {axis_name}: "
                    f"expected {expected_size}, got {ainfo.axis_size} "
                    f"at world_rank={world_rank}"
                )

            if ainfo.axis_rank in rank_to_world:
                pass
            else:
                rank_to_world[ainfo.axis_rank] = world_rank

        if expected_size is None:
            raise ValueError(f"No parallel_info found for sharded axis {axis_name!r}")

        if set(rank_to_world.keys()) != set(range(expected_size)):
            raise ValueError(
                f"axis_rank coverage for {axis_name} is incomplete: "
                f"got {sorted(rank_to_world.keys())}, expected 0..{expected_size - 1}"
            )

        world_ranks_by_axis_rank = [rank_to_world[i] for i in range(expected_size)]

        fn = _get_unshard_fn(spec)

        steps.append(
            UnshardStep(
                axis=spec.parallel,
                dim_index=dim_idx,
                fn=fn,
                world_ranks_by_axis_rank=world_ranks_by_axis_rank,
            )
        )

    return UnshardPlan(
        tensor_name=tensor_name,
        dims_str=dims_str,
        replicated_axes=replicated_axes,
        steps=steps,
    )


def execute_unshard_plan(
    plan: UnshardPlan,
    tensors_by_world_rank: dict[int, torch.Tensor],
) -> torch.Tensor:
    """Execute an unshard plan on actual tensor data.

    Concatenation order is strictly by axis_rank (not world_rank).
    """
    if not plan.steps:
        if len(tensors_by_world_rank) == 1:
            return next(iter(tensors_by_world_rank.values()))
        raise ValueError(
            f"No unshard steps but got {len(tensors_by_world_rank)} tensors"
        )

    current_tensors = dict(tensors_by_world_rank)

    for step in plan.steps:
        new_tensors: dict[int, torch.Tensor] = {}
        processed_axis_ranks: set[int] = set()

        groups: dict[tuple, list[tuple[int, torch.Tensor]]] = {}
        for world_rank, tensor in current_tensors.items():
            group_key_parts = []
            for other_step in plan.steps:
                if other_step is step:
                    continue
                if world_rank in other_step.world_ranks_by_axis_rank:
                    idx = other_step.world_ranks_by_axis_rank.index(world_rank)
                    group_key_parts.append((other_step.axis.value, idx))
                else:
                    group_key_parts.append((other_step.axis.value, -1))
            group_key = tuple(sorted(group_key_parts))

            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append((world_rank, tensor))

        for group_key, members in groups.items():
            ordered: list[tuple[int, torch.Tensor]] = []
            for axis_rank, target_world_rank in enumerate(
                step.world_ranks_by_axis_rank
            ):
                for world_rank, tensor in members:
                    if world_rank == target_world_rank:
                        ordered.append((world_rank, tensor))
                        break

            if not ordered:
                continue

            tensors_ordered = [t for _, t in ordered]
            merged = step.fn(tensors_ordered, step.dim_index)
            representative_rank = ordered[0][0]
            new_tensors[representative_rank] = merged

        current_tensors = new_tensors

    if len(current_tensors) != 1:
        raise ValueError(f"Expected 1 tensor after unshard, got {len(current_tensors)}")

    return next(iter(current_tensors.values()))


def unshard_concat(tensors: list[torch.Tensor], dim: int) -> torch.Tensor:
    return torch.cat(tensors, dim=dim)


def _get_unshard_fn(
    spec: DimSpec,
) -> Callable[[list[torch.Tensor], int], torch.Tensor]:
    if spec.reduction is not None:
        raise NotImplementedError(
            f"Unshard for reduction={spec.reduction} not yet implemented (Phase 2)"
        )
    if spec.ordering is not None and spec.ordering != Ordering.NATURAL:
        raise NotImplementedError(
            f"Unshard for ordering={spec.ordering} not yet implemented (Phase 2)"
        )
    return unshard_concat
