# SPDX-License-Identifier: Apache-2.0
"""Local policy settings reject invalid startup input before execution."""

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
