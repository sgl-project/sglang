"""Admission behavior of opt-in completion of a restored short tail."""

from unittest.mock import MagicMock

import pytest
import test_prefill_adder as fixtures

from sglang.srt.managers.schedule_policy import AddReqResult
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


@pytest.fixture
def factory():
    case = fixtures.TestPrefillAdder()
    case.setUp()
    try:
        yield case
    finally:
        case.doCleanups()


def setup_admission(
    factory, *, initial=8, remaining=4, tail=8, host=4, existing=True, enabled=True
):
    cache = factory.mock_tree_cache
    cache.supports_mamba.return_value = False
    cache.supports_swa.return_value = False
    cache.window_sub_pools.return_value = ()
    cache.window_reprefill_tail_tokens.return_value = 0
    cache.prefer_complete_restored_requests = enabled
    allocator = factory.mock_token_allocator
    allocator.available_size.return_value = 10000
    allocator.full_available_size.return_value = 10000
    req = factory.create_mock_req("restore", 0, 1)
    req.full_untruncated_fill_ids = list(range(host + tail))
    req.best_match_node = req.last_node
    req.host_hit_length = host
    req.needs_host_load_back.return_value = host > 0
    cache.init_load_back.return_value = (host, req.last_node)
    cache.load_back_is_cache_owned.return_value = False
    adder = factory.create_adder(
        factory.create_running_batch(), rem_chunk_tokens=initial, page_size=1
    )
    adder.rem_chunk_tokens = remaining
    if existing:
        adder.can_run_list = [object()]
    return adder, req, cache


def test_restored_short_tail_waits_before_restore_and_vote(factory):
    adder, req, cache = setup_admission(factory)
    adder.prefill_delayer_single_pass = MagicMock()
    assert adder.add_one_req(req, False, None) is AddReqResult.OTHER
    assert req not in adder.can_run_list
    cache.init_load_back.assert_not_called()
    adder.prefill_delayer_single_pass.negotiate_should_allow_prefill.assert_not_called()


@pytest.mark.parametrize(
    "options", [dict(existing=False), dict(enabled=False), dict(tail=12), dict(host=0)]
)
def test_empty_batch_and_nonmatching_cases_continue_to_progress(factory, options):
    adder, req, cache = setup_admission(factory, **options)
    adder.add_one_req(req, False, None)
    assert req in adder.can_run_list
    assert req.extend_len > 0


def test_zero_chunk_budget_rejects_before_storage_restore(factory):
    adder, req, cache = setup_admission(factory, remaining=0)
    adder.add_one_req(req, False, None)
    assert req not in adder.can_run_list
    cache.init_load_back.assert_not_called()
