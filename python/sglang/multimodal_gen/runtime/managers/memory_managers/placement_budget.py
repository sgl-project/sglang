# SPDX-License-Identifier: Apache-2.0
"""Joint placement optimization across every constrained memory resource."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import msgspec


class NoFeasiblePlacementError(ValueError):
    pass


class PlacementOption(msgspec.Struct, frozen=True):
    """One alternative placement for a component or layer group.

    Resource costs are deltas from the currently active placement. This is the
    normalized form of absolute placement costs: the current placement is the
    implicit zero option, positive values consume headroom, and negative values
    release a resource. Keeping one resource map lets a single decision satisfy
    rank/phase VRAM and node-scoped pinned-host constraints. A lifecycle option
    may describe different load and serving placements by including load,
    transition, and runtime resource dimensions in this same vector. The caller
    must construct only compatible lifecycle combinations; independently solved
    load and serving choices cannot be composed safely after the fact.

    ``estimated_latency_savings`` is an integer supplied by the caller. It can
    be transfer nanoseconds when measured bandwidth is available, or avoided
    transfer bytes when only a stable ordering is known.
    """

    group_key: str
    option_key: str
    resource_delta_bytes: dict[str, int]
    estimated_latency_savings: int
    # Lexicographic soft cost used only after latency-equivalent placements
    # have been identified. Resource budgets remain the hard safety limits.
    placement_cost_bytes: tuple[int, ...] = ()


class PlacementPlan(msgspec.Struct, frozen=True):
    resource_delta_bytes: dict[str, int] = {}
    estimated_latency_savings: int = 0
    placement_cost_bytes: tuple[int, ...] = ()
    selections: list[PlacementOption] = []


# Resource deltas, utility, soft placement cost, selected options.
_State = tuple[
    tuple[int, ...],
    int,
    tuple[int, ...],
    tuple[PlacementOption, ...],
]


def _selection_key(state: _State) -> tuple[str, ...]:
    return tuple(option.option_key for option in state[3])


def _better_same_cost(candidate: _State, incumbent: _State) -> bool:
    if candidate[1] != incumbent[1]:
        return candidate[1] > incumbent[1]
    return _selection_key(candidate) < _selection_key(incumbent)


def _dominates(candidate: _State, other: _State) -> bool:
    """Whether candidate is at least as useful and no more expensive."""
    return (
        candidate[1] >= other[1]
        and all(left <= right for left, right in zip(candidate[0], other[0]))
        and all(left <= right for left, right in zip(candidate[2], other[2]))
    )


def _pareto_prune(states: Iterable[_State]) -> list[_State]:
    """Drop states dominated across every constrained resource."""
    best_at_cost: dict[tuple[tuple[int, ...], tuple[int, ...]], _State] = {}
    for state in states:
        cost_key = (state[0], state[2])
        incumbent = best_at_cost.get(cost_key)
        if incumbent is None or _better_same_cost(state, incumbent):
            best_at_cost[cost_key] = state

    ordered = sorted(
        best_at_cost.values(),
        key=lambda state: (
            sum(state[0]),
            state[0],
            state[2],
            -state[1],
            _selection_key(state),
        ),
    )
    frontier: list[_State] = []
    for state in ordered:
        if any(_dominates(existing, state) for existing in frontier):
            continue
        frontier = [
            existing for existing in frontier if not _dominates(state, existing)
        ]
        frontier.append(state)
    return frontier


def optimize_placement(
    options: Iterable[PlacementOption],
    *,
    resource_budget_bytes: Mapping[str, int],
    estimated_latency_tolerance: int = 0,
    require_selection_from_every_group: bool = False,
) -> PlacementPlan:
    """Choose at most one alternative per group under every resource budget.

    Not selecting an alternative keeps that group at its current placement.
    Callers that provide complete target states may instead require exactly
    one option per group, allowing the solver to replace or demote a current
    placement rather than only add to it.
    All alternatives are evaluated with the same selection vector, so every
    constraint describing this placement state sees the same decision. A
    choice may trade one resource for another by using a negative delta. The
    result is exact for the supplied option frontier, resource model and
    utility values. Distinct load and serving states may be encoded in one
    option, but the result does not imply that the supplied frontier contains
    every physically possible lifecycle plan.
    """
    if estimated_latency_tolerance < 0:
        raise ValueError("estimated_latency_tolerance must be non-negative")

    options = list(options)
    placement_cost_dimensions = max(
        (len(option.placement_cost_bytes) for option in options), default=0
    )
    for option in options:
        if option.placement_cost_bytes and (
            len(option.placement_cost_bytes) != placement_cost_dimensions
        ):
            raise ValueError(
                "all non-empty placement costs must have the same dimensions"
            )

    grouped: dict[str, list[PlacementOption]] = {}
    option_keys: set[str] = set()
    all_resource_names = tuple(sorted(resource_budget_bytes))
    for option in options:
        if option.option_key in option_keys:
            raise ValueError(f"duplicate placement option key {option.option_key!r}")
        option_keys.add(option.option_key)
        unknown_resources = option.resource_delta_bytes.keys() - all_resource_names
        if unknown_resources:
            raise ValueError(
                f"placement option {option.option_key!r} uses unknown resources: "
                f"{sorted(unknown_resources)}"
            )
        grouped.setdefault(option.group_key, []).append(option)

    # A resource whose worst possible positive use still fits cannot constrain
    # the choice. Removing it before Pareto construction is exact and matters
    # for the common case where a large-host server can pin every candidate:
    # all pin-prefix alternatives then collapse to the highest-utility one for
    # each GPU placement instead of multiplying the global frontier.
    resource_names = tuple(
        resource_name
        for resource_name in all_resource_names
        if sum(
            max(
                0,
                *(
                    option.resource_delta_bytes.get(resource_name, 0)
                    for option in group_options
                ),
            )
            for group_options in grouped.values()
        )
        > resource_budget_bytes[resource_name]
    )
    resource_budgets = tuple(
        int(resource_budget_bytes[name]) for name in resource_names
    )

    ordered_groups = []
    for group_key in sorted(grouped):
        group_options = sorted(grouped[group_key], key=lambda option: option.option_key)
        local_states = []
        if not require_selection_from_every_group:
            local_states.append(
                (
                    (0,) * len(resource_names),
                    0,
                    (0,) * placement_cost_dimensions,
                    (),
                )
            )
        local_states.extend(
            (
                tuple(
                    option.resource_delta_bytes.get(resource_name, 0)
                    for resource_name in resource_names
                ),
                option.estimated_latency_savings,
                option.placement_cost_bytes or (0,) * placement_cost_dimensions,
                (option,),
            )
            for option in group_options
        )
        ordered_groups.append(
            [state[3][0] for state in _pareto_prune(local_states) if state[3]]
        )
    minimum_remaining: list[tuple[int, ...]] = [
        (0,) * len(resource_names) for _ in range(len(ordered_groups) + 1)
    ]
    for group_index in range(len(ordered_groups) - 1, -1, -1):
        group_options = ordered_groups[group_index]
        next_minimum = minimum_remaining[group_index + 1]
        minimum_remaining[group_index] = tuple(
            next_delta
            + (
                min(
                    option.resource_delta_bytes.get(resource_name, 0)
                    for option in group_options
                )
                if require_selection_from_every_group
                else min(
                    0,
                    *(
                        option.resource_delta_bytes.get(resource_name, 0)
                        for option in group_options
                    ),
                )
            )
            for next_delta, resource_name in zip(next_minimum, resource_names)
        )

    frontier: list[_State] = [
        (
            (0,) * len(resource_names),
            0,
            (0,) * placement_cost_dimensions,
            (),
        )
    ]
    for group_index, group_options in enumerate(ordered_groups):
        candidates = [] if require_selection_from_every_group else list(frontier)
        for state in frontier:
            for option in group_options:
                resource_deltas = tuple(
                    current + option.resource_delta_bytes.get(resource_name, 0)
                    for current, resource_name in zip(state[0], resource_names)
                )
                candidates.append(
                    (
                        resource_deltas,
                        state[1] + option.estimated_latency_savings,
                        tuple(
                            current + delta
                            for current, delta in zip(
                                state[2],
                                option.placement_cost_bytes
                                or (0,) * placement_cost_dimensions,
                            )
                        ),
                        state[3] + (option,),
                    )
                )
        remaining = minimum_remaining[group_index + 1]
        reachable = [
            state
            for state in candidates
            if all(
                used + releasable <= budget
                for used, releasable, budget in zip(
                    state[0], remaining, resource_budgets
                )
            )
        ]
        frontier = _pareto_prune(reachable)

    frontier = [
        state
        for state in frontier
        if all(used <= budget for used, budget in zip(state[0], resource_budgets))
    ]
    if not frontier:
        raise NoFeasiblePlacementError("no placement satisfies all resource budgets")

    best_utility = max(state[1] for state in frontier)
    latency_equivalent = [
        state
        for state in frontier
        if state[1] >= best_utility - estimated_latency_tolerance
    ]
    best = min(
        latency_equivalent,
        key=lambda state: (
            state[2],
            sum(state[0]),
            state[0],
            -state[1],
            _selection_key(state),
        ),
    )
    full_resource_deltas = {
        resource_name: sum(
            option.resource_delta_bytes.get(resource_name, 0) for option in best[3]
        )
        for resource_name in all_resource_names
    }
    return PlacementPlan(
        resource_delta_bytes=full_resource_deltas,
        estimated_latency_savings=best[1],
        placement_cost_bytes=best[2],
        selections=list(best[3]),
    )
