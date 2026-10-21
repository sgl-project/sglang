# SPDX-License-Identifier: Apache-2.0
"""Local policy settings reject invalid startup input before execution."""

import copy

import pytest

from sglang.srt.managers.kv_control.service import parse_config
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


@pytest.mark.parametrize(
    "policies",
    [
        [{"name": "same"}, {"name": "same"}],
        [{"name": str(i)} for i in range(17)],
        [{"name": "bad", "implementation": "unavailable"}],
        [{"name": "bad", "options": {"ttl_seconds": 0}}],
        [{"name": "bad", "options": {"priority": float("inf")}}],
        [{"name": "bad", "options": {"force_pin": True}}],
    ],
)
def test_invalid_startup_policy_configuration_is_rejected(policies):
    with pytest.raises(ValueError):
        parse_config({"namespace": "ns", "policies": policies})


def test_policies_are_absent_by_default_and_options_are_typed():
    assert parse_config({"namespace": "ns"}).policies == ()
    config = parse_config(
        {
            "namespace": "ns",
            "policies": [
                {"name": "retention", "options": {"priority": -7, "ttl_seconds": 3}}
            ],
        }
    )
    assert config.policies[0].options.priority == -7
    assert config.policies[0].options.ttl_seconds == 3


def test_quota_tag_and_share_mapping_are_typed_without_mutating_input():
    raw = {
        "namespace": "ns",
        "policies": [
            {"name": "retain"},
            {
                "name": "quota",
                "implementation": "quota",
                "options": {"shares": {"A": 2, "B": 1}},
            },
        ],
    }
    before = copy.deepcopy(raw)
    config = parse_config(raw)
    assert raw == before
    assert [
        (share.session_id, share.weight) for share in config.policies[1].options.shares
    ] == [("A", 2), ("B", 1)]


@pytest.mark.parametrize(
    "options",
    [
        {"aging_seconds": 0},
        {"aging_seconds": 1e-300},
        {"reuse_score_weight": float("nan")},
        {"shares": {"A": float("nan")}},
        {"shares": {"A": -1}},
        {
            "shares": [
                {"session_id": "A", "weight": 1},
                {"session_id": "A", "weight": 2},
            ]
        },
        {"priority": 30},
    ],
)
def test_invalid_quota_options_fail_before_scheduler_use(options):
    with pytest.raises(ValueError):
        parse_config(
            {
                "namespace": "ns",
                "policies": [
                    {"name": "q", "implementation": "quota", "options": options}
                ],
            }
        )


@pytest.mark.parametrize(
    "options",
    [
        {"tier": "unavailable"},
        {"min_pressure": -1},
        {"min_pressure": float("nan")},
        {"initial_estimated_transfer_seconds": 0},
        {"lead_seconds": float("inf")},
        {"max_contexts": 0},
        {"force_pin": True},
    ],
)
def test_invalid_tool_options_fail_before_execution(options):
    with pytest.raises(ValueError):
        parse_config(
            {
                "namespace": "ns",
                "policies": [
                    {"name": "tools", "implementation": "tool_wait", "options": options}
                ],
            }
        )
