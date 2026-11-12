"""Read-only FCFS observations and their actual load-reporting consumer."""

from array import array
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch

from sglang.srt.disaggregation.mode import DisaggregationMode
from sglang.srt.managers.scheduler_components.load_inquirer import SchedulerLoadInquirer
from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.srt.mem_cache.storage.flexkv.flexkv_cache_lifecycle import (
    FlexKVCacheLifecycleMixin,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class ObservedCache(FlexKVCacheLifecycleMixin, RadixCache):
    pass


def case(gpu=4, host=4, *, key=None, salt=None, supported=True):
    base = RadixCache.create_simulated(page_size=4)
    cache = ObservedCache.__new__(ObservedCache)
    cache.__dict__.update(base.__dict__)
    if gpu:
        cache.insert(
            InsertParams(
                key=RadixKey(array("q", range(gpu)), key, cache_salt=salt),
                value=torch.arange(gpu),
            )
        )
    connector = NS(
        supports_cache_namespace=supported,
        pending_probe_plan=Mock(return_value={"observed_at": 100.0, "indices": [0]}),
        sync_prefix_length=Mock(side_effect=lambda length: length),
        peek_cpu_prefix_tokens=Mock(return_value=host),
    )
    cache.flexkv_connector = connector
    req = NS(
        rid="waiting",
        extra_key=key,
        cache_salt=salt,
        full_untruncated_fill_ids=array("q", range(9)),
        positional_embed_overrides=None,
        _compute_max_prefix_len=lambda length: length - 1,
        prefix_len=0,
        num_matched_prefix_tokens=0,
        seqlen=9,
        last_node=cache.root_node,
    )
    return cache, req, connector


def tree_state(cache):
    def walk(node):
        return (
            id(node),
            node.last_access_time,
            node.lock_ref,
            tuple(torch.as_tensor(node.value).tolist())
            if node.value is not None
            else None,
            tuple((key, walk(child)) for key, child in node.children.items()),
        )

    return walk(cache.root_node), cache.evictable_size(), cache.protected_size()


@pytest.mark.parametrize("gpu", [0, 4, 8])
@pytest.mark.parametrize("host", [0, 4, None])
@pytest.mark.parametrize("namespace", [(None, None), ("lora", "tenant")])
def test_observation_is_read_only_and_distinguishes_unavailable(gpu, host, namespace):
    cache, req, connector = case(gpu, host, key=namespace[0], salt=namespace[1])
    before = tree_state(cache)
    cache.refresh_pending_prefix_metrics([req])
    assert tree_state(cache) == before
    assert req.prefix_len == req.num_matched_prefix_tokens == 0
    expected = gpu if gpu == 8 else None if host is None else min(8, gpu + host)
    assert req._pending_prefix_observed_tokens == expected
    assert req._pending_prefix_probe_valid is (expected is not None)
    assert req._pending_prefix_observed_at == 100.0
    if gpu == 8:
        connector.peek_cpu_prefix_tokens.assert_not_called()
    else:
        args, kwargs = connector.peek_cpu_prefix_tokens.call_args
        assert list(args[0]) == list(range(8))
        assert args[1].tolist() == [False] * gpu + [True] * (8 - gpu)
        assert ("namespace" in kwargs) is (namespace != (None, None))


@pytest.mark.parametrize(
    "capability", ["pending_probe_plan", "peek_cpu_prefix_tokens", "sync_prefix_length"]
)
def test_missing_capability_invalidates_stale_observation(capability):
    cache, req, connector = case()
    req._pending_prefix_probe_valid = True
    delattr(connector, capability)
    cache.refresh_pending_prefix_metrics([req])
    assert req._pending_prefix_probe_valid is False


def test_namespace_without_support_is_unknown_without_unscoped_query():
    cache, req, connector = case(key="lora", supported=False)
    cache.refresh_pending_prefix_metrics([req])
    assert req._pending_prefix_probe_valid is False
    connector.peek_cpu_prefix_tokens.assert_not_called()


@pytest.mark.parametrize("guard", ["disable", "embed", "empty"])
def test_non_reusable_prefix_is_known_zero(guard):
    cache, req, connector = case()
    if guard == "disable":
        cache.disable = True
    elif guard == "embed":
        req.positional_embed_overrides = object()
    else:
        req._compute_max_prefix_len = lambda length: 0
    cache.refresh_pending_prefix_metrics([req])
    assert req._pending_prefix_observed_tokens == 0
    assert req._pending_prefix_probe_valid is True
    connector.peek_cpu_prefix_tokens.assert_not_called()


def test_refreshes_only_leader_selected_requests():
    cache, first, connector = case()
    second = NS(**vars(first))
    second._pending_prefix_observed_at = 99.5
    cache.refresh_pending_prefix_metrics([first, second])
    connector.pending_probe_plan.assert_called_once_with([0.0, 99.5])
    assert not hasattr(second, "_pending_prefix_observed_tokens")


def test_device_consensus_controls_whether_host_collective_runs():
    cache, req, connector = case(gpu=8, host=4)
    connector.sync_prefix_length.side_effect = lambda local: 4
    cache.refresh_pending_prefix_metrics([req])
    connector.peek_cpu_prefix_tokens.assert_called_once()
    assert req._pending_prefix_observed_tokens == 8


@pytest.mark.parametrize("matched_policy", [False, True])
@pytest.mark.parametrize(
    "observed,valid,expected",
    [(8, True, 1), (0, True, 9), (None, False, None), (8, False, None)],
)
def test_load_consumer_prefers_valid_observation_and_preserves_unknown_fallback(
    matched_policy, observed, valid, expected
):
    req = NS(
        seqlen=9,
        num_matched_prefix_tokens=4,
        _pending_prefix_observed_tokens=observed,
        _pending_prefix_probe_valid=valid,
    )
    inquirer = NS(
        disaggregation_mode=DisaggregationMode.NULL,
        waiting_queue_prefix_matched=lambda: matched_policy,
        get_recent_cache_hit_rate=lambda: 0.5,
        get_waiting_queue=lambda: [req],
        get_chunked_req=lambda: None,
        get_waiting_storage_span=None,
    )
    if matched_policy:
        expected = 5
    elif expected is None:
        expected = 4
    assert SchedulerLoadInquirer.get_num_waiting_uncached_tokens(inquirer) == expected


def test_peek_of_partial_node_does_not_split_or_touch_lru():
    cache, _, _ = case(gpu=8)
    before = tree_state(cache)
    assert (
        cache.peek_prefix_length(MatchPrefixParams(key=RadixKey(array("q", range(4)))))
        == 4
    )
    assert tree_state(cache) == before
