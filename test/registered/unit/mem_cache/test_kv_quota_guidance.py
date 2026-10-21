# SPDX-License-Identifier: Apache-2.0
"""Real scheduler match/budget boundaries behind read-only quota advice."""

from array import array
from types import SimpleNamespace
from unittest.mock import patch

import msgspec
import pytest
import test_kv_control_cache as cache_fixtures
import test_kv_control_transfers as transfers

from sglang.srt.managers.kv_control.contracts import Associate, Status
from sglang.srt.managers.kv_control.policy_runtime import QuotaDefinition, QuotaOptions
from sglang.srt.managers.kv_control.service import ServiceConfig
from sglang.srt.managers.kv_hints import KvHintAction, KvHintsEnvelope
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sglang.srt.mem_cache.common import match_kv_cache
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")
owner = transfers.owner
CONFIG = ServiceConfig(
    namespace="ns",
    engine_id="engine",
    policies=(
        QuotaDefinition(name="quota", options=QuotaOptions(borrow_idle_capacity=False)),
    ),
)
with_quota = pytest.mark.parametrize("owner", [CONFIG], indirect=True)


def task(service, name):
    service.control.index.associate(
        Associate(
            session_id=name,
            continuation_id="c",
            expected_revision=0,
            target=cache_fixtures.target(),
        )
    )


def request(service, name, tokens=(1, 2, 3, 4, 9), *, output=1):
    req = Req(
        rid=name,
        origin_input_text=None,
        origin_input_ids=array("q", tokens),
        sampling_params=SamplingParams(max_new_tokens=output, ignore_eos=True),
        vocab_size=128,
    )
    service.prepare_request(req)
    envelope = KvHintsEnvelope(
        protocol_version="0.1",
        message_id="bind-" + name,
        actions=[
            KvHintAction(
                action_id="resume",
                action_type="sglang.signal",
                action_version="1.0",
                payload={
                    "target": {
                        "kind": "session",
                        "namespace": "ns",
                        "session_id": name,
                    },
                    "signal": "resume",
                },
            )
        ],
    )
    service.attach(req, envelope)
    return req


def observe(cache, service, requests, running=()):
    for req in requests:
        match_kv_cache(req, cache)
    batch = SimpleNamespace(reqs=list(running))
    scheduler = SimpleNamespace(
        waiting_queue=list(requests),
        chunked_req=None,
        new_token_ratio_tracker=SimpleNamespace(current=1),
    )
    service.update_policy_guidance(scheduler, batch)
    return service.policy_runtime._view, scheduler, batch


@with_quota
def test_fresh_match_observation_counts_shared_unreferenced_prefix_once(owner):
    _, cache, service, dma = owner
    transfers.seed(cache)
    for name in ("A", "B"):
        task(service, name)
    a, b = request(service, "A"), request(service, "B")
    view, _, _ = observe(cache, service, (a, b))
    assert [r.cached_tokens for r in view.requests] == [4, 4]
    assert [r.estimated_incremental_tokens for r in view.requests] == [7, 7]
    assert view.estimated_waiting_tokens == 10
    assert all(r.task is not None for r in view.requests)
    assert not service.policy_runtime._binding_tasks
    assert not dma.pending


@with_quota
def test_policy_deferral_aging_and_native_priority_order(owner):
    case, cache, service, dma = owner
    transfers.seed(cache)
    for name in ("A", "B"):
        task(service, name)
    a, b = request(service, "A", output=30), request(service, "B", (7, 8, 9), output=1)
    observe(cache, service, (a, b))
    assert not service.policy_allows_request(a) and service.policy_allows_request(b)
    assert service.policy_request_key(b, 1, 0) < service.policy_request_key(a, 0, 0)
    a.priority, b.priority = 10, 0
    assert service.policy_request_key(a, 0, -1) < service.policy_request_key(b, 1, -1)
    case.clock.now += 2
    assert service.policy_allows_request(a)
    # Expired requests use arrival order rather than an unbounded advisory score.
    assert service.policy_request_key(a, 0, 0) < service.policy_request_key(b, 1, 0)


@with_quota
def test_advisory_allowance_cannot_spend_stale_space_before_real_prefill(owner):
    _, cache, service, dma = owner
    transfers.seed(cache)
    task(service, "A")
    req = request(service, "A")
    _, _, batch = observe(cache, service, (req,))
    assert service.policy_allows_request(req)
    other = cache.token_to_kv_pool_allocator.alloc(60)
    try:
        req.init_next_round_input(cache)
        adder = PrefillAdder(
            1, cache, cache.token_to_kv_pool_allocator, batch, 1, 64, None
        )
        result = adder.add_one_req(
            req, has_chunked_req=False, truncation_align_size=None
        )
        assert result in (AddReqResult.NO_TOKEN, AddReqResult.OTHER)
        assert not adder.can_run_list
        assert cache.token_to_kv_pool_allocator.available_size() == 0
    finally:
        cache.token_to_kv_pool_allocator.free(other)


@pytest.mark.parametrize(
    "owner", [msgspec.structs.replace(CONFIG, max_policy_requests=1)], indirect=True
)
def test_truncated_request_view_falls_back_without_zero_demand(owner):
    _, cache, service, dma = owner
    task(service, "A")
    task(service, "B")
    a, b = request(service, "A"), request(service, "B")
    view, _, _ = observe(cache, service, (a, b))
    assert not view.requests_complete and view.estimated_waiting_tokens is None
    assert service.policy_allows_request(a) and service.policy_allows_request(b)


@with_quota
def test_token_signal_is_unconsumed_and_session_signal_does_not_scan_requests(owner):
    _, cache, service, dma = owner
    envelope = KvHintsEnvelope(
        protocol_version="0.1",
        message_id="token",
        actions=[
            KvHintAction(
                action_id="a",
                action_type="sglang.signal",
                action_version="1.0",
                payload={
                    "target": msgspec.json.decode(
                        msgspec.json.encode(cache_fixtures.target())
                    ),
                    "signal": "importance",
                    "importance": 90,
                },
            )
        ],
    )
    with patch.object(
        service.control.index, "resolve", side_effect=AssertionError("history")
    ):
        assert (
            service.control.submit(envelope, service.control.context("router"))
            .results[0]
            .status
            == Status.UNCONSUMED
        )
    task(service, "A")
    with patch(
        "sglang.srt.managers.kv_control.policy_observer.observe_scheduler",
        side_effect=AssertionError("request scan"),
    ):
        req = request(service, "A")
    assert req.kv_control_policy_tasks
    assert not service.policy_runtime._binding_tasks and not dma.pending
