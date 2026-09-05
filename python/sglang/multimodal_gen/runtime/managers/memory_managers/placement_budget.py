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

_UnconstrainedState = tuple[
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
        # soft costs are summed and compared lexicographically by the final
        # objective; this order is translation invariant across suffixes
        and candidate[2] <= other[2]
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
    if ordered and not ordered[0][0] and len(ordered[0][2]) == 2:
        # with no hard resource dimension, local dominance is a 2D soft-cost
        # skyline; a Fenwick sweep avoids the quadratic layer frontier scan
        second_costs = sorted({state[2][1] for state in ordered})
        second_cost_indices = {
            cost: index + 1 for index, cost in enumerate(second_costs)
        }
        best_utility: list[int | None] = [None] * (len(second_costs) + 1)

        def prefix_max(index: int) -> int | None:
            result = None
            while index > 0:
                value = best_utility[index]
                if value is not None and (result is None or value > result):
                    result = value
                index -= index & -index
            return result

        def update(index: int, utility: int) -> None:
            while index < len(best_utility):
                value = best_utility[index]
                if value is None or utility > value:
                    best_utility[index] = utility
                index += index & -index

        frontier = []
        for state in sorted(
            ordered,
            key=lambda item: (
                item[2][0],
                item[2][1],
                -item[1],
                _selection_key(item),
            ),
        ):
            index = second_cost_indices[state[2][1]]
            dominating_utility = prefix_max(index)
            if dominating_utility is not None and dominating_utility >= state[1]:
                continue
            frontier.append(state)
            update(index, state[1])
        return frontier

    frontier: list[_State] = []
    for state in ordered:
        if any(_dominates(existing, state) for existing in frontier):
            continue
        frontier = [
            existing for existing in frontier if not _dominates(state, existing)
        ]
        frontier.append(state)
    return frontier


def _drop_locally_latency_ineligible_options(
    options: Iterable[PlacementOption],
    *,
    resource_names: tuple[str, ...],
    estimated_latency_tolerance: int,
) -> list[PlacementOption]:
    """Drop an option that alone spends more than the global latency slack.

    When two options have the same constrained-resource vector, replacing the
    slower one cannot make any complete plan infeasible. If that replacement
    gains more than the final plan's entire latency tolerance, the slower
    option cannot belong to a latency-equivalent optimum regardless of the
    choices made by other groups. This matters when an unconstrained HostPin
    dimension leaves many otherwise identical pin prefixes in the frontier.
    """
    options = list(options)
    best_utility_by_resources: dict[tuple[int, ...], int] = {}
    resource_vector_by_key: dict[str, tuple[int, ...]] = {}
    for option in options:
        resources = tuple(
            option.resource_delta_bytes.get(resource_name, 0)
            for resource_name in resource_names
        )
        resource_vector_by_key[option.option_key] = resources
        best_utility_by_resources[resources] = max(
            best_utility_by_resources.get(resources, option.estimated_latency_savings),
            option.estimated_latency_savings,
        )
    return [
        option
        for option in options
        if option.estimated_latency_savings
        >= best_utility_by_resources[resource_vector_by_key[option.option_key]]
        - estimated_latency_tolerance
    ]


def _prune_unconstrained_states(
    states: Iterable[_UnconstrainedState],
) -> list[_UnconstrainedState]:
    """Keep the exact latency-deficit versus lexicographic-cost frontier."""
    best_by_deficit: dict[int, _UnconstrainedState] = {}
    for state in states:
        incumbent = best_by_deficit.get(state[0])
        if incumbent is None or (
            state[1],
            tuple(option.option_key for option in state[2]),
        ) < (incumbent[1], tuple(option.option_key for option in incumbent[2])):
            best_by_deficit[state[0]] = state

    frontier = []
    best_cost: tuple[int, ...] | None = None
    for state in sorted(
        best_by_deficit.values(),
        key=lambda item: (
            item[0],
            item[1],
            tuple(option.option_key for option in item[2]),
        ),
    ):
        if best_cost is not None and best_cost <= state[1]:
            continue
        frontier.append(state)
        best_cost = state[1]
    return frontier


def _optimize_unconstrained(
    grouped: Mapping[str, list[PlacementOption]],
    *,
    placement_cost_dimensions: int,
    estimated_latency_tolerance: int,
    require_selection_from_every_group: bool,
) -> _State:
    """Solve the exact global latency window after every budget is non-binding."""
    zero_cost = (0,) * placement_cost_dimensions
    frontier: list[_UnconstrainedState] = [(0, zero_cost, ())]
    maximum_utility = 0

    for group_key in sorted(grouped):
        alternatives = list(grouped[group_key])
        group_maximum = max(
            [option.estimated_latency_savings for option in alternatives]
            + ([] if require_selection_from_every_group else [0])
        )
        maximum_utility += group_maximum
        group_states = [
            (
                group_maximum - option.estimated_latency_savings,
                option.placement_cost_bytes or zero_cost,
                (option,),
            )
            for option in alternatives
            if group_maximum - option.estimated_latency_savings
            <= estimated_latency_tolerance
        ]
        if (
            not require_selection_from_every_group
            and group_maximum <= estimated_latency_tolerance
        ):
            group_states.append((group_maximum, zero_cost, ()))
        group_frontier = _prune_unconstrained_states(group_states)
        frontier = _prune_unconstrained_states(
            (
                accumulated[0] + choice[0],
                tuple(left + right for left, right in zip(accumulated[1], choice[1])),
                accumulated[2] + choice[2],
            )
            for accumulated in frontier
            for choice in group_frontier
            if accumulated[0] + choice[0] <= estimated_latency_tolerance
        )

    best = min(
        frontier,
        key=lambda state: (
            state[1],
            state[0],
            tuple(option.option_key for option in state[2]),
        ),
    )
    return (), maximum_utility - best[0], best[1], best[2]


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

    if not resource_names:
        best = _optimize_unconstrained(
            grouped,
            placement_cost_dimensions=placement_cost_dimensions,
            estimated_latency_tolerance=estimated_latency_tolerance,
            require_selection_from_every_group=require_selection_from_every_group,
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

    # Each local frontier includes the implicit "keep current placement" choice
    # when selection is optional. Keep the branch representation compact so the
    # global solve does not materialize and repeatedly Pareto-scan a large
    # Cartesian product of layer-residency and HostPin prefixes.
    ordered_groups = []
    for group_key in sorted(grouped):
        group_options = sorted(
            _drop_locally_latency_ineligible_options(
                grouped[group_key],
                resource_names=resource_names,
                estimated_latency_tolerance=estimated_latency_tolerance,
            ),
            key=lambda option: option.option_key,
        )
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
        ordered_groups.append(_pareto_prune(local_states))

    def _suffix_bounds(groups):
        minimum_resources = [(0,) * len(resource_names) for _ in range(len(groups) + 1)]
        maximum_utility = [0] * (len(groups) + 1)
        minimum_cost = [
            (0,) * placement_cost_dimensions for _ in range(len(groups) + 1)
        ]
        for group_index in range(len(groups) - 1, -1, -1):
            group = groups[group_index]
            minimum_resources[group_index] = tuple(
                following + min(choice[0][resource_index] for choice in group)
                for resource_index, following in enumerate(
                    minimum_resources[group_index + 1]
                )
            )
            maximum_utility[group_index] = maximum_utility[group_index + 1] + max(
                choice[1] for choice in group
            )
            minimum_cost[group_index] = tuple(
                following + min(choice[2][cost_index] for choice in group)
                for cost_index, following in enumerate(minimum_cost[group_index + 1])
            )
        return minimum_resources, maximum_utility, minimum_cost

    # First maximize utility. Choices with the same constrained-resource vector
    # collapse to the fastest one; HostPin-only soft-cost alternatives therefore
    # do not multiply this pass when HostPin is non-binding.
    utility_groups = []
    for group in ordered_groups:
        best_by_resources: dict[tuple[int, ...], _State] = {}
        for choice in group:
            incumbent = best_by_resources.get(choice[0])
            if incumbent is None or (-choice[1], choice[2], _selection_key(choice)) < (
                -incumbent[1],
                incumbent[2],
                _selection_key(incumbent),
            ):
                best_by_resources[choice[0]] = choice
        utility_groups.append(
            sorted(
                best_by_resources.values(),
                key=lambda choice: (-choice[1], choice[0], _selection_key(choice)),
            )
        )

    minimum_resources, maximum_utility, _ = _suffix_bounds(utility_groups)
    best_utility: int | None = None

    def _maximize_utility(
        group_index: int, resources: tuple[int, ...], utility: int
    ) -> None:
        nonlocal best_utility
        if (
            best_utility is not None
            and utility + maximum_utility[group_index] <= best_utility
        ):
            return
        if any(
            used + releasable > budget
            for used, releasable, budget in zip(
                resources, minimum_resources[group_index], resource_budgets
            )
        ):
            return
        if group_index == len(utility_groups):
            best_utility = utility
            return
        for choice in utility_groups[group_index]:
            _maximize_utility(
                group_index + 1,
                tuple(left + right for left, right in zip(resources, choice[0])),
                utility + choice[1],
            )

    _maximize_utility(0, (0,) * len(resource_names), 0)
    if best_utility is None:
        raise NoFeasiblePlacementError("no placement satisfies all resource budgets")

    # Then minimize the lexicographic soft placement cost among every plan in
    # the single global latency-equivalence window. Depth-first bounds avoid the
    # quadratic global Pareto scans that large layerwise frontiers otherwise
    # trigger while preserving the exact final ordering.
    minimum_resources, maximum_utility, minimum_cost = _suffix_bounds(ordered_groups)
    utility_floor = best_utility - estimated_latency_tolerance
    best: _State | None = None

    def _minimize_cost(
        group_index: int,
        resources: tuple[int, ...],
        utility: int,
        cost: tuple[int, ...],
        selections: tuple[PlacementOption, ...],
    ) -> None:
        nonlocal best
        if utility + maximum_utility[group_index] < utility_floor:
            return
        if any(
            used + releasable > budget
            for used, releasable, budget in zip(
                resources, minimum_resources[group_index], resource_budgets
            )
        ):
            return
        if best is not None:
            lower_cost = tuple(
                current + remaining
                for current, remaining in zip(cost, minimum_cost[group_index])
            )
            if lower_cost > best[2]:
                return
        if group_index == len(ordered_groups):
            candidate = (resources, utility, cost, selections)
            if best is None or (
                candidate[2],
                sum(candidate[0]),
                candidate[0],
                -candidate[1],
                _selection_key(candidate),
            ) < (
                best[2],
                sum(best[0]),
                best[0],
                -best[1],
                _selection_key(best),
            ):
                best = candidate
            return
        for choice in ordered_groups[group_index]:
            _minimize_cost(
                group_index + 1,
                tuple(left + right for left, right in zip(resources, choice[0])),
                utility + choice[1],
                tuple(left + right for left, right in zip(cost, choice[2])),
                selections + choice[3],
            )

    _minimize_cost(
        0,
        (0,) * len(resource_names),
        0,
        (0,) * placement_cost_dimensions,
        (),
    )
    assert best is not None
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
