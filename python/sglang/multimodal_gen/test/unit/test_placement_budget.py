# SPDX-License-Identifier: Apache-2.0

import pytest

from sglang.multimodal_gen.runtime.managers.memory_managers.placement_budget import (
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
