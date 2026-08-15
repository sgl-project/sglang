from collections import defaultdict
from typing import NamedTuple, Optional

from sglang.srt.debug_utils.comparator.aligner.unsharder.types import (
    AxisInfo,
    ConcatParams,
    CpThdConcatParams,
    PickParams,
    ReduceSumParams,
    UnsharderParams,
    UnsharderPlan,
)
from sglang.srt.debug_utils.comparator.dims_spec import (
    TOKEN_DIM_NAME,
    DimSpec,
    ParallelAxis,
    ParallelModifier,
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
    explicit_replicated_axes: frozenset[ParallelAxis] = frozenset(),
    thd_global_seq_lens: Optional[list[int]] = None,
    dp_filtered_axis: Optional[ParallelAxis] = None,
) -> list[UnsharderPlan]:
    if not parallel_infos:
        raise ValueError("parallel_infos must not be empty")

    # Within each dim spec, reverse modifier order: innermost shard (rightmost) unshards first.
    reversed_sharded_modifiers: list[tuple[str, ParallelModifier]] = [
        (spec.sanitized_name, m)
        for spec in dim_specs
        for m in reversed(spec.parallel_modifiers)
    ]

    sharded_axes_raw: set[ParallelAxis] = {
        m.axis for _, m in reversed_sharded_modifiers
    }
    all_axes: set[ParallelAxis] = {axis for info in parallel_infos for axis in info}

    # axis annotated in dims but absent from all parallel_infos -> axis_size=1, skip
    sharded_axes: set[ParallelAxis] = sharded_axes_raw & all_axes
    reversed_sharded_modifiers = [
        (name, m) for name, m in reversed_sharded_modifiers if m.axis in sharded_axes
    ]

    # RECOMPUTE_PSEUDO is always implicitly replicated (system-injected, not user-facing)
    auto_replicated: frozenset[ParallelAxis] = frozenset(
        {ParallelAxis.RECOMPUTE_PSEUDO} & all_axes
    )
    effective_replicated: frozenset[ParallelAxis] = (
        explicit_replicated_axes | auto_replicated
    )

    _validate_explicit_replicated(
        explicit_replicated_axes=effective_replicated,
        sharded_axes=sharded_axes,
        all_axes=all_axes,
        parallel_infos=parallel_infos,
        dp_filtered_axis=dp_filtered_axis,
    )
    replicated_axes: frozenset[ParallelAxis] = effective_replicated

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
            modifier.axis,
            _resolve_unshard_params(
                modifier=modifier,
                dim_name=dim_name,
                parallel_infos=parallel_infos,
                thd_global_seq_lens=thd_global_seq_lens,
            ),
        )
        for dim_name, modifier in reversed_sharded_modifiers
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


def _validate_explicit_replicated(
    *,
    explicit_replicated_axes: frozenset[ParallelAxis],
    sharded_axes: set[ParallelAxis],
    all_axes: set[ParallelAxis],
    parallel_infos: list[dict[ParallelAxis, AxisInfo]],
    dp_filtered_axis: Optional[ParallelAxis] = None,
) -> None:
    """Validate explicit replicated declarations against sharded axes and parallel_infos."""
    invalid: frozenset[ParallelAxis] = explicit_replicated_axes - all_axes
    if invalid:
        invalid_names: str = ", ".join(sorted(a.value for a in invalid))
        raise ValueError(
            f"Declared replicated axes {{{invalid_names}}} not found in parallel_infos "
            f"(active axes: {{{', '.join(sorted(a.value for a in all_axes))}}})"
        )

    conflict: set[ParallelAxis] = explicit_replicated_axes & sharded_axes
    if conflict:
        conflict_names: str = ", ".join(sorted(a.value for a in conflict))
        raise ValueError(
            f"Axes {{{conflict_names}}} declared as both sharded and replicated"
        )

    _validate_replicated_axes_orthogonal(
        explicit_replicated_axes=explicit_replicated_axes,
        parallel_infos=parallel_infos,
    )

    candidate_axes: set[ParallelAxis] = (
        all_axes - sharded_axes - explicit_replicated_axes
    )
    implicitly_replicated: frozenset[ParallelAxis] = _compute_dependent_axes(
        parent_axes=explicit_replicated_axes,
        candidate_axes=candidate_axes,
        parallel_infos=parallel_infos,
    )
    implicitly_sharded: frozenset[ParallelAxis] = _compute_dependent_axes(
        parent_axes=sharded_axes,
        candidate_axes=candidate_axes - implicitly_replicated,
        parallel_infos=parallel_infos,
    )

    declared_axes: frozenset[ParallelAxis] = frozenset(
        sharded_axes
        | explicit_replicated_axes
        | implicitly_replicated
        | implicitly_sharded
        | ({dp_filtered_axis} if dp_filtered_axis is not None else set())
    )
    undeclared: set[ParallelAxis] = all_axes - declared_axes

    jointly_determined: frozenset[ParallelAxis] = frozenset(
        child
        for child in undeclared
        if _is_jointly_determined(
            parallel_infos, parent_axes=declared_axes, child=child
        )
    )
    undeclared -= jointly_determined

    if undeclared:
        undeclared_names: str = ", ".join(sorted(a.value for a in undeclared))
        raise ValueError(
            f"Axes {{{undeclared_names}}} are active (axis_size > 1) but not declared "
            f"in dims. Annotate as sharded in dim spec or as '# axis:replicated'."
        )


def _validate_replicated_axes_orthogonal(
    *,
    explicit_replicated_axes: frozenset[ParallelAxis],
    parallel_infos: list[dict[ParallelAxis, AxisInfo]],
) -> None:
    """Every pair of explicitly replicated axes must be fully orthogonal (no dependency)."""
    axes: list[ParallelAxis] = sorted(explicit_replicated_axes, key=lambda a: a.value)
    if len(axes) < 2:
        return

    violations: list[str] = []
    for i, axis_a in enumerate(axes):
        for axis_b in axes[i + 1 :]:
            for parent, child in [(axis_a, axis_b), (axis_b, axis_a)]:
                if _is_dependent_axis(parallel_infos, parent=parent, child=child):
                    violations.append(
                        f"'{parent.value}' determines '{child.value}' — "
                        f"remove '{child.value}:replicated'"
                    )

    if violations:
        details = "; ".join(violations)
        raise ValueError(
            f"Explicitly-replicated axes overlap (not orthogonal): {details}"
        )


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


def _compute_dependent_axes(
    parent_axes: set[ParallelAxis] | frozenset[ParallelAxis],
    candidate_axes: set[ParallelAxis],
    parallel_infos: list[dict[ParallelAxis, AxisInfo]],
) -> frozenset[ParallelAxis]:
    """Return candidate axes whose rank is uniquely determined by some parent axis."""
    return frozenset(
        child
        for child in candidate_axes
        if any(
            _is_dependent_axis(parallel_infos, parent=parent, child=child)
            for parent in parent_axes
        )
    )


def _is_jointly_determined(
    parallel_infos: list[dict[ParallelAxis, AxisInfo]],
    *,
    parent_axes: frozenset[ParallelAxis],
    child: ParallelAxis,
) -> bool:
    """True if child's rank is uniquely determined by the joint tuple of parent ranks.

    Unlike ``_is_dependent_axis`` which checks single-parent dependency, this
    checks whether the *combination* of all parent axes jointly determines the
    child.  For example, ``edp_rank`` may not be a function of ``tp_rank`` alone
    or ``cp_rank`` alone, but it *is* a function of ``(tp_rank, cp_rank)``.

    Parent axes that are absent from *every* info are ignored (they carry no
    information — e.g. DP with size 1 filtered by ``normalize_parallel_info``).
    However, a parent axis present in *some* infos but missing from an info
    that contains the child makes the determination incomplete → ``False``.
    """
    if not parent_axes:
        return False

    active_parents: frozenset[ParallelAxis] = frozenset(
        ax for ax in parent_axes if any(ax in info for info in parallel_infos)
    )
    if not active_parents:
        return False

    mapping: dict[frozenset, int] = {}
    for info in parallel_infos:
        if child not in info:
            continue
        if not active_parents.issubset(info):
            return False
        parent_key = frozenset((ax, info[ax].axis_rank) for ax in active_parents)
        child_rank: int = info[child].axis_rank
        if mapping.setdefault(parent_key, child_rank) != child_rank:
            return False

    return bool(mapping)


def _is_dependent_axis(
    parallel_infos: list[dict[ParallelAxis, AxisInfo]],
    *,
    parent: ParallelAxis,
    child: ParallelAxis,
) -> bool:
    """True if child's rank is uniquely determined by parent's rank."""
    parent_rank_to_child_rank: dict[int, int] = {}
    for info in parallel_infos:
        if parent not in info or child not in info:
            continue
        parent_rank = info[parent].axis_rank
        child_rank = info[child].axis_rank
        if parent_rank_to_child_rank.setdefault(parent_rank, child_rank) != child_rank:
            return False
    return True


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
    modifier: ParallelModifier,
    dim_name: str,
    parallel_infos: list[dict[ParallelAxis, AxisInfo]],
    thd_global_seq_lens: Optional[list[int]] = None,
) -> UnsharderParams:
    if modifier.reduction is not None:
        return ReduceSumParams()

    if (
        dim_name == TOKEN_DIM_NAME
        and modifier.axis == ParallelAxis.CP
        and thd_global_seq_lens is not None
    ):
        axis_size: int = parallel_infos[0][modifier.axis].axis_size
        for s in thd_global_seq_lens:
            if s % axis_size != 0:
                raise ValueError(
                    f"THD seq_len {s} is not divisible by cp_size {axis_size}. "
                    f"Sequences must be padded to a multiple of cp_size for CP zigzag."
                )
        seq_lens_per_rank: list[int] = [s // axis_size for s in thd_global_seq_lens]
        return CpThdConcatParams(dim_name=dim_name, seq_lens_per_rank=seq_lens_per_rank)

    return ConcatParams(dim_name=dim_name)
