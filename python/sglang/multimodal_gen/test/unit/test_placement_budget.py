# SPDX-License-Identifier: Apache-2.0

import itertools
import random

import pytest

from sglang.multimodal_gen.runtime.managers.memory_managers.placement_budget import (
    NoFeasiblePlacementError,
    PlacementOption,
    optimize_placement,
)


def _option(
    group: str,
    option: str,
    *,
    device: int = 0,
    pinned: int = 0,
    savings: int = 0,
) -> PlacementOption:
    return PlacementOption(
        group_key=group,
        option_key=option,
        resource_delta_bytes={
            "gpu:rank0:runtime": device,
            "hostpin:node0": pinned,
        },
        estimated_latency_savings=savings,
    )


def _exhaustive_plan_key(
    options: list[PlacementOption],
    *,
    resource_budget_bytes: dict[str, int],
    estimated_latency_tolerance: int,
    require_selection_from_every_group: bool,
):
    resource_names = tuple(sorted(resource_budget_bytes))
    groups: dict[str, list[PlacementOption | None]] = {}
    for option in options:
        groups.setdefault(option.group_key, []).append(option)
    choices = []
    for group_key in sorted(groups):
        group = sorted(groups[group_key], key=lambda option: option.option_key)
        if not require_selection_from_every_group:
            group = [None, *group]
        choices.append(group)

    feasible = []
    for selections in itertools.product(*choices):
        selected = tuple(option for option in selections if option is not None)
        resources = tuple(
            sum(option.resource_delta_bytes.get(name, 0) for option in selected)
            for name in resource_names
        )
        if any(
            used > resource_budget_bytes[name]
            for name, used in zip(resource_names, resources)
        ):
            continue
        utility = sum(option.estimated_latency_savings for option in selected)
        cost_dimensions = max(
            (len(option.preference_cost) for option in options), default=0
        )
        cost = tuple(
            sum(
                (option.preference_cost[index] if option.preference_cost else 0)
                for option in selected
            )
            for index in range(cost_dimensions)
        )
        feasible.append((selected, resources, utility, cost))

    if not feasible:
        return None
    best_utility = max(state[2] for state in feasible)
    utility_floor = best_utility - estimated_latency_tolerance
    selected, resources, utility, cost = min(
        (state for state in feasible if state[2] >= utility_floor),
        key=lambda state: (
            state[3],
            sum(state[1]),
            state[1],
            -state[2],
            tuple(option.option_key for option in state[0]),
        ),
    )
    return (
        tuple(option.option_key for option in selected),
        resources,
        utility,
        cost,
    )


def test_joint_plan_beats_independent_greedy_allocations():
    plan = optimize_placement(
        [
            _option("a", "a-resident", device=10, savings=10),
            _option("a", "a-pinned", pinned=10, savings=9),
            _option("b", "b-resident", device=10, savings=100),
            _option("b", "b-pinned", pinned=10, savings=1),
        ],
        resource_budget_bytes={
            "gpu:rank0:runtime": 10,
            "hostpin:node0": 10,
        },
    )

    assert [option.option_key for option in plan.selections] == [
        "a-pinned",
        "b-resident",
    ]
    assert plan.estimated_latency_savings == 109


def test_one_selection_vector_obeys_load_runtime_and_hostpin_budgets():
    plan = optimize_placement(
        [
            PlacementOption(
                group_key="text_encoder",
                option_key="text-encoder-resident",
                resource_delta_bytes={"gpu:load": 8, "gpu:runtime": 2},
                estimated_latency_savings=60,
            ),
            PlacementOption(
                group_key="text_encoder",
                option_key="text-encoder-pinned",
                resource_delta_bytes={"gpu:load": 1, "hostpin": 6},
                estimated_latency_savings=50,
            ),
            PlacementOption(
                group_key="transformer",
                option_key="transformer-resident",
                resource_delta_bytes={"gpu:load": 2, "gpu:runtime": 8},
                estimated_latency_savings=100,
            ),
            PlacementOption(
                group_key="transformer",
                option_key="transformer-pinned",
                resource_delta_bytes={"gpu:runtime": 1, "hostpin": 6},
                estimated_latency_savings=20,
            ),
        ],
        resource_budget_bytes={
            "gpu:load": 9,
            "gpu:runtime": 8,
            "hostpin": 6,
        },
        require_selection_from_every_group=True,
    )

    assert [option.option_key for option in plan.selections] == [
        "text-encoder-pinned",
        "transformer-resident",
    ]
    assert plan.resource_delta_bytes == {
        "gpu:load": 3,
        "gpu:runtime": 8,
        "hostpin": 6,
    }


def test_selects_at_most_one_option_per_group():
    plan = optimize_placement(
        [
            _option("dit", "half", device=5, savings=8),
            _option("dit", "full", device=10, savings=20),
        ],
        resource_budget_bytes={
            "gpu:rank0:runtime": 20,
            "hostpin:node0": 0,
        },
    )

    assert [option.option_key for option in plan.selections] == ["full"]


def test_exact_optimizer_can_choose_multiple_smaller_components():
    plan = optimize_placement(
        [
            _option("large", "large", device=10, savings=100),
            _option("small-a", "small-a", device=6, savings=70),
            _option("small-b", "small-b", device=4, savings=40),
        ],
        resource_budget_bytes={
            "gpu:rank0:runtime": 10,
            "hostpin:node0": 0,
        },
    )

    assert [option.option_key for option in plan.selections] == [
        "small-a",
        "small-b",
    ]
    assert plan.estimated_latency_savings == 110


def test_ties_prefer_lower_resource_cost_then_stable_key():
    plan = optimize_placement(
        [
            _option("a", "higher-cost", device=2, savings=10),
            _option("a", "z", device=1, savings=10),
            _option("a", "a", device=1, savings=10),
        ],
        resource_budget_bytes={
            "gpu:rank0:runtime": 2,
            "hostpin:node0": 0,
        },
    )

    assert [option.option_key for option in plan.selections] == ["a"]
    assert plan.resource_delta_bytes == {
        "gpu:rank0:runtime": 1,
        "hostpin:node0": 0,
    }


def test_complete_state_selection_can_replace_current_placement():
    plan = optimize_placement(
        [
            PlacementOption(
                group_key="transformer",
                option_key="transformer:resident",
                resource_delta_bytes={"vram": 0},
                estimated_latency_savings=100,
            ),
            PlacementOption(
                group_key="transformer",
                option_key="transformer:offload",
                resource_delta_bytes={"vram": -40},
                estimated_latency_savings=0,
            ),
            PlacementOption(
                group_key="text_encoder",
                option_key="text_encoder:resident",
                resource_delta_bytes={"vram": 30},
                estimated_latency_savings=200,
            ),
            PlacementOption(
                group_key="text_encoder",
                option_key="text_encoder:offload",
                resource_delta_bytes={"vram": 0},
                estimated_latency_savings=0,
            ),
        ],
        resource_budget_bytes={"vram": 0},
        require_selection_from_every_group=True,
    )

    assert {option.option_key for option in plan.selections} == {
        "transformer:offload",
        "text_encoder:resident",
    }


def test_latency_equivalent_plan_prefers_lower_soft_memory_cost():
    plan = optimize_placement(
        [
            PlacementOption(
                group_key="transformer",
                option_key="transformer-resident",
                resource_delta_bytes={"gpu:rank0:runtime": 40},
                estimated_latency_savings=1_000,
                preference_cost=(40, 0),
            ),
            PlacementOption(
                group_key="text_encoder",
                option_key="text-encoder-resident",
                resource_delta_bytes={"gpu:rank0:runtime": 10},
                estimated_latency_savings=60,
                preference_cost=(10, 0),
            ),
        ],
        resource_budget_bytes={"gpu:rank0:runtime": 100},
        estimated_latency_tolerance=100,
    )

    assert [option.option_key for option in plan.selections] == ["transformer-resident"]
    assert plan.estimated_latency_savings == 1_000
    assert plan.preference_cost == (40, 0)


def test_latency_tolerance_is_global_not_per_option():
    plan = optimize_placement(
        [
            PlacementOption(
                group_key="a",
                option_key="a-resident",
                resource_delta_bytes={},
                estimated_latency_savings=60,
                preference_cost=(1,),
            ),
            PlacementOption(
                group_key="b",
                option_key="b-resident",
                resource_delta_bytes={},
                estimated_latency_savings=60,
                preference_cost=(1,),
            ),
        ],
        resource_budget_bytes={},
        estimated_latency_tolerance=100,
    )

    # Neither 60-unit option would survive a per-option 100-unit cutoff. Their
    # combined benefit first establishes a non-zero global utility floor; the
    # least costly placement within that floor then keeps one of them.
    assert [option.option_key for option in plan.selections] == ["a-resident"]
    assert plan.estimated_latency_savings == 60


def test_lexicographic_preference_prunes_irrelevant_later_dimension():
    plan = optimize_placement(
        [
            PlacementOption(
                group_key="dit",
                option_key="partial-pin",
                resource_delta_bytes={"gpu": 1},
                estimated_latency_savings=90,
                preference_cost=(0, -90, 5),
            ),
            PlacementOption(
                group_key="dit",
                option_key="full-pin",
                resource_delta_bytes={"gpu": 1},
                estimated_latency_savings=100,
                preference_cost=(0, -100, 10),
            ),
        ],
        resource_budget_bytes={"gpu": 1},
        estimated_latency_tolerance=20,
        require_selection_from_every_group=True,
    )

    assert [option.option_key for option in plan.selections] == ["full-pin"]


def test_one_static_placement_satisfies_all_observed_phase_constraints():
    plan = optimize_placement(
        [
            PlacementOption(
                group_key="a",
                option_key="a-resident",
                resource_delta_bytes={
                    "gpu:rank0:encode": 8,
                    "gpu:rank0:denoise": 2,
                    "hostpin:node0": 0,
                },
                estimated_latency_savings=9,
            ),
            PlacementOption(
                group_key="a",
                option_key="a-pinned",
                resource_delta_bytes={
                    "gpu:rank0:encode": 1,
                    "gpu:rank0:denoise": 1,
                    "hostpin:node0": 8,
                },
                estimated_latency_savings=5,
            ),
            PlacementOption(
                group_key="b",
                option_key="b-resident",
                resource_delta_bytes={
                    "gpu:rank0:encode": 2,
                    "gpu:rank0:denoise": 8,
                    "hostpin:node0": 0,
                },
                estimated_latency_savings=10,
            ),
            PlacementOption(
                group_key="b",
                option_key="b-pinned",
                resource_delta_bytes={
                    "gpu:rank0:encode": 1,
                    "gpu:rank0:denoise": 1,
                    "hostpin:node0": 8,
                },
                estimated_latency_savings=4,
            ),
        ],
        resource_budget_bytes={
            "gpu:rank0:encode": 10,
            "gpu:rank0:denoise": 10,
            "hostpin:node0": 8,
        },
    )

    assert [option.option_key for option in plan.selections] == [
        "a-resident",
        "b-resident",
    ]
    assert plan.resource_delta_bytes == {
        "gpu:rank0:encode": 10,
        "gpu:rank0:denoise": 10,
        "hostpin:node0": 0,
    }


def test_lifecycle_plan_jointly_constrains_load_transition_runtime_and_hostpin():
    plan = optimize_placement(
        [
            PlacementOption(
                group_key="dit-lifecycle",
                option_key="dit-gpu-load-to-resident",
                resource_delta_bytes={
                    "gpu:rank0:load": 8,
                    "gpu:rank0:transition": 8,
                    "gpu:rank0:runtime": 6,
                    "hostpin:node0": 0,
                },
                estimated_latency_savings=100,
            ),
            PlacementOption(
                group_key="dit-lifecycle",
                option_key="dit-rank-local-load-to-layerwise",
                resource_delta_bytes={
                    "gpu:rank0:load": 4,
                    "gpu:rank0:transition": 3,
                    "gpu:rank0:runtime": 3,
                    "hostpin:node0": 8,
                },
                estimated_latency_savings=80,
            ),
            PlacementOption(
                group_key="encoder-lifecycle",
                option_key="encoder-gpu-load-to-resident",
                resource_delta_bytes={
                    "gpu:rank0:load": 2,
                    "gpu:rank0:transition": 4,
                    "gpu:rank0:runtime": 4,
                    "hostpin:node0": 0,
                },
                estimated_latency_savings=50,
            ),
            PlacementOption(
                group_key="encoder-lifecycle",
                option_key="encoder-cpu-load-to-pinned",
                resource_delta_bytes={
                    "gpu:rank0:load": 0,
                    "gpu:rank0:transition": 1,
                    "gpu:rank0:runtime": 1,
                    "hostpin:node0": 4,
                },
                estimated_latency_savings=20,
            ),
        ],
        resource_budget_bytes={
            "gpu:rank0:load": 10,
            "gpu:rank0:transition": 10,
            "gpu:rank0:runtime": 10,
            "hostpin:node0": 8,
        },
    )

    # Independently optimizing load and runtime would pick both resident
    # choices, but their transition uses 12 bytes and is infeasible.
    assert [option.option_key for option in plan.selections] == [
        "dit-rank-local-load-to-layerwise",
        "encoder-gpu-load-to-resident",
    ]
    assert plan.resource_delta_bytes == {
        "gpu:rank0:load": 6,
        "gpu:rank0:transition": 7,
        "gpu:rank0:runtime": 7,
        "hostpin:node0": 8,
    }


def test_negative_headroom_forces_the_plan_to_release_a_resource():
    plan = optimize_placement(
        [
            PlacementOption(
                group_key="z-cold-encoder",
                option_key="cold-encoder-pageable",
                resource_delta_bytes={"hostpin:node0": -8},
                estimated_latency_savings=-1,
            )
        ],
        resource_budget_bytes={"hostpin:node0": -4},
    )

    assert [option.option_key for option in plan.selections] == [
        "cold-encoder-pageable"
    ]


def test_negative_headroom_without_a_release_is_infeasible():
    with pytest.raises(ValueError, match="no placement satisfies"):
        optimize_placement(
            [],
            resource_budget_bytes={"hostpin:node0": -1},
        )


def test_unknown_resource_is_rejected():
    with pytest.raises(ValueError, match="unknown resources"):
        optimize_placement(
            [
                PlacementOption(
                    group_key="dit",
                    option_key="resident",
                    resource_delta_bytes={"gpu:rank0:denoise": 1},
                    estimated_latency_savings=1,
                )
            ],
            resource_budget_bytes={"gpu:rank0:load": 1},
        )


def test_signed_deltas_trade_pinned_host_for_device_memory():
    plan = optimize_placement(
        [
            PlacementOption(
                group_key="dit",
                option_key="dit-resident",
                resource_delta_bytes={
                    "gpu:rank0:runtime": 8,
                    "hostpin:node0": -8,
                },
                estimated_latency_savings=20,
            ),
            PlacementOption(
                group_key="encoder",
                option_key="encoder-pinned",
                resource_delta_bytes={
                    "gpu:rank0:runtime": 0,
                    "hostpin:node0": 8,
                },
                estimated_latency_savings=10,
            ),
        ],
        resource_budget_bytes={
            "gpu:rank0:runtime": 8,
            "hostpin:node0": 0,
        },
    )

    assert [option.option_key for option in plan.selections] == [
        "dit-resident",
        "encoder-pinned",
    ]
    assert plan.resource_delta_bytes["hostpin:node0"] == 0


def test_joint_plan_can_accept_a_local_regression_to_reassign_hostpin():
    plan = optimize_placement(
        [
            PlacementOption(
                group_key="cold-encoder",
                option_key="cold-encoder-pageable",
                resource_delta_bytes={"hostpin:node0": -10},
                estimated_latency_savings=-1,
            ),
            PlacementOption(
                group_key="a-hot-dit",
                option_key="hot-dit-pinned",
                resource_delta_bytes={"hostpin:node0": 10},
                estimated_latency_savings=100,
            ),
        ],
        resource_budget_bytes={"hostpin:node0": 0},
    )

    assert [option.option_key for option in plan.selections] == [
        "hot-dit-pinned",
        "cold-encoder-pageable",
    ]
    assert plan.estimated_latency_savings == 99


def test_nonbinding_resource_is_removed_without_losing_full_accounting():
    plan = optimize_placement(
        [
            _option("dit", "dit-low", device=5, pinned=20, savings=10),
            _option("dit", "dit-high", device=5, pinned=80, savings=20),
        ],
        resource_budget_bytes={
            "gpu:rank0:runtime": 5,
            "hostpin:node0": 100,
        },
    )

    assert [option.option_key for option in plan.selections] == ["dit-high"]
    assert plan.resource_delta_bytes == {
        "gpu:rank0:runtime": 5,
        "hostpin:node0": 80,
    }


def test_nonbinding_pin_prefixes_keep_only_globally_latency_eligible_states():
    plan = optimize_placement(
        [
            PlacementOption(
                group_key="dit",
                option_key="dit-pin-none",
                resource_delta_bytes={"gpu": 5, "hostpin": 0},
                estimated_latency_savings=800,
                preference_cost=(5, 0),
            ),
            PlacementOption(
                group_key="dit",
                option_key="dit-pin-most",
                resource_delta_bytes={"gpu": 5, "hostpin": 80},
                estimated_latency_savings=950,
                preference_cost=(5, 80),
            ),
            PlacementOption(
                group_key="dit",
                option_key="dit-pin-all",
                resource_delta_bytes={"gpu": 5, "hostpin": 100},
                estimated_latency_savings=1_000,
                preference_cost=(5, 100),
            ),
        ],
        resource_budget_bytes={"gpu": 5, "hostpin": 100},
        estimated_latency_tolerance=100,
    )

    # No-pin is over the global tolerance and cannot be selected. Pin-most is
    # still latency-equivalent, so the soft memory tie-break chooses it.
    assert [option.option_key for option in plan.selections] == ["dit-pin-most"]


def test_branch_and_bound_matches_exhaustive_multiresource_search():
    rng = random.Random(20260827)
    resource_names = ("load-vram", "runtime-vram", "hostpin")

    for _ in range(200):
        options = []
        for group_index in range(rng.randint(1, 5)):
            for option_index in range(rng.randint(1, 4)):
                options.append(
                    PlacementOption(
                        group_key=f"group-{group_index}",
                        option_key=f"group-{group_index}:option-{option_index}",
                        resource_delta_bytes={
                            name: rng.randint(-5, 8) for name in resource_names
                        },
                        estimated_latency_savings=rng.randint(-5, 20),
                        preference_cost=tuple(rng.randint(-10, 10) for _ in range(5)),
                    )
                )
        budgets = {name: rng.randint(-2, 12) for name in resource_names}
        tolerance = rng.randint(0, 5)
        require_every_group = rng.choice((False, True))
        expected = _exhaustive_plan_key(
            options,
            resource_budget_bytes=budgets,
            estimated_latency_tolerance=tolerance,
            require_selection_from_every_group=require_every_group,
        )

        if expected is None:
            with pytest.raises(NoFeasiblePlacementError):
                optimize_placement(
                    options,
                    resource_budget_bytes=budgets,
                    estimated_latency_tolerance=tolerance,
                    require_selection_from_every_group=require_every_group,
                )
            continue

        plan = optimize_placement(
            options,
            resource_budget_bytes=budgets,
            estimated_latency_tolerance=tolerance,
            require_selection_from_every_group=require_every_group,
        )
        expected_keys, expected_resources, expected_utility, expected_cost = expected
        assert tuple(option.option_key for option in plan.selections) == expected_keys
        assert (
            tuple(plan.resource_delta_bytes[name] for name in sorted(resource_names))
            == expected_resources
        )
        assert plan.estimated_latency_savings == expected_utility
        assert plan.preference_cost == expected_cost


def test_identical_resource_columns_keep_only_the_strictest_budget():
    options = [
        PlacementOption(
            group_key="dit",
            option_key="dit:small",
            resource_delta_bytes={"phase-a": 4, "phase-b": 4},
            estimated_latency_savings=4,
        ),
        PlacementOption(
            group_key="dit",
            option_key="dit:large",
            resource_delta_bytes={"phase-a": 8, "phase-b": 8},
            estimated_latency_savings=8,
        ),
    ]

    plan = optimize_placement(
        options,
        resource_budget_bytes={"phase-a": 10, "phase-b": 6},
    )

    assert [option.option_key for option in plan.selections] == ["dit:small"]
    assert plan.resource_delta_bytes == {"phase-a": 4, "phase-b": 4}


def test_nonbinding_preference_sweep_matches_exhaustive_search():
    rng = random.Random(20260828)

    for _ in range(100):
        options = []
        for group_index in range(rng.randint(1, 5)):
            for option_index in range(rng.randint(1, 6)):
                options.append(
                    PlacementOption(
                        group_key=f"group-{group_index}",
                        option_key=f"group-{group_index}:option-{option_index}",
                        resource_delta_bytes={},
                        estimated_latency_savings=rng.randint(0, 50),
                        preference_cost=(
                            rng.randint(0, 20),
                            rng.randint(0, 20),
                            rng.randint(0, 20),
                        ),
                    )
                )
        tolerance = rng.randint(0, 10)
        require_every_group = rng.choice((False, True))
        expected = _exhaustive_plan_key(
            options,
            resource_budget_bytes={},
            estimated_latency_tolerance=tolerance,
            require_selection_from_every_group=require_every_group,
        )

        plan = optimize_placement(
            options,
            resource_budget_bytes={},
            estimated_latency_tolerance=tolerance,
            require_selection_from_every_group=require_every_group,
        )
        assert expected is not None
        expected_keys, _, expected_utility, expected_cost = expected
        assert tuple(option.option_key for option in plan.selections) == expected_keys
        assert plan.estimated_latency_savings == expected_utility
        assert plan.preference_cost == expected_cost
