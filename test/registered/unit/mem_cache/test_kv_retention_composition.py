# SPDX-License-Identifier: Apache-2.0
"""Two configured policies share real copies with independent TTLs and limits."""

import pytest
import test_kv_control_cache as cache_fixtures
import test_kv_control_transfers as fixtures
import test_kv_policy_runtime as runtime_tests

from sglang.srt.managers.kv_control.contracts import Status
from sglang.srt.managers.kv_control.policy_runtime import (
    PolicyDefinition,
    RetentionOptions,
)
from sglang.srt.managers.kv_control.service import CacheLimits, ServiceConfig
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")
owner = fixtures.owner


@pytest.mark.parametrize(
    "owner",
    [
        ServiceConfig(
            namespace="ns",
            engine_id="engine",
            cache_limits=CacheLimits(max_policy_claims_per_copy=limit),
            policies=(
                PolicyDefinition(
                    name="long", options=RetentionOptions(priority=20, ttl_seconds=10)
                ),
                PolicyDefinition(
                    name="short", options=RetentionOptions(priority=90, ttl_seconds=1)
                ),
            ),
        )
        for limit in (1, 4)
    ],
    indirect=True,
)
def test_composed_policy_results_expiry_and_capacity_are_independent(owner):
    case, cache, service, dma = owner
    fixtures.seed(cache)
    parent = runtime_tests.submit(
        service, target=cache_fixtures.target(), signal="resume"
    ).results[0]
    assert parent.status == Status.COMPLETED
    children = [
        service.control.registry.get(ref) for ref in parent.update.derived_operations
    ]
    assert len(children) == 2
    limited = service.config.cache_limits.max_policy_claims_per_copy == 1
    assert [child.status for child in children] == [
        Status.COMPLETED,
        Status.REJECTED if limited else Status.COMPLETED,
    ]
    assert runtime_tests.observed(service) == [20 if limited else 90]
    case.clock.now += 2
    assert runtime_tests.observed(service) == [20]
    case.clock.now += 9
    assert runtime_tests.observed(service) == [0]
    acknowledgement = service.control.registry.acknowledge(
        children[0].ref.context, children[0].ref.message_id
    )
    assert [summary.status for summary in acknowledgement.summaries] == [
        child.status for child in children
    ]
    assert not service.has_pending and not dma.pending
