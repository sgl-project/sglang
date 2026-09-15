import ast
import contextlib
import importlib.util
import sys
import threading
from array import array
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest
import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InitLoadBackParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _load_flexkv_radix_cache_class():
    """Load the cache without requiring the optional FlexKV package."""
    module_name = "_flexkv_radix_cache_load_back_under_test"
    connector_name = "flexkv.integration.sglang.connector"
    connector_stub = ModuleType(connector_name)
    connector_stub.FlexKVConnector = object
    connector_stub.FlexKVHostReleaseShim = object
    module_path = (
        Path(__file__).resolve().parents[4]
        / "python/sglang/srt/mem_cache/storage/flexkv/flexkv_radix_cache.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        with patch.dict(sys.modules, {connector_name: connector_stub}):
            spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
    return module.FlexKVRadixCache


FlexKVRadixCache = _load_flexkv_radix_cache_class()


def _make_cache(page_size=4):
    allocator = MagicMock()
    allocator.device = torch.device("cpu")
    allocator.available_size.return_value = 1024
    next_slot = 100

    def alloc(size):
        nonlocal next_slot
        slots = torch.arange(next_slot, next_slot + size, dtype=torch.int64)
        next_slot += size
        return slots

    allocator.alloc.side_effect = alloc
    cache = RadixCache.create_simulated(
        mock_allocator=allocator,
        page_size=page_size,
    )
    cache.__class__ = FlexKVRadixCache
    cache._mode = FlexKVRadixCache.match_prefix.__globals__["FlexKVMode"].IP
    cache.flexkv_connector = MagicMock()
    cache.store_stream = MagicMock()
    cache._load_markers = {}
    cache._defer_duplicate_restores = False
    cache._restoring_host_prefixes = {}
    cache._restore_prefix_by_rid = {}
    cache._inflight_store_nodes = {}
    cache._pending_store_launches = {}
    cache._pending_store_copies = {}
    cache._restore_leases = {}
    cache._aborted_restore_leases = {}
    cache._restore_generation = 0
    cache._async_store_slot_mapping = False
    cache._profile_store_stages = False
    cache.flexkv_connector.is_store_sync_leader = True
    cache.flexkv_connector.sync_ready_store_rids.side_effect = lambda rids: list(rids)
    cache._node_lock = threading.Lock()
    return cache, allocator


def _load(cache, key, value_numel, uncached_len, rid):
    load_fn = MagicMock(side_effect=lambda slots: int(slots.numel()))
    result = cache._allocate_and_load(
        key=key,
        value_numel=value_numel,
        uncached_len=uncached_len,
        last_node=cache.root_node,
        tracking_rid=rid,
        sglang_req_id=rid,
        load_fn=load_fn,
    )
    assert result is not None
    return result, load_fn


def test_duplicate_restore_reuses_live_node_without_creating_stale_leaf():
    cache, allocator = _make_cache()
    key = RadixKey(array("q", range(4)))

    (first_indices, first_node), first_load = _load(cache, key, 0, 4, "first")
    (second_indices, second_node), second_load = _load(cache, key, 0, 4, "second")

    assert first_load.call_count == 1
    second_load.assert_not_called()
    cache.flexkv_connector.release_pending.assert_called_once_with("second")
    assert torch.equal(second_indices, first_indices)
    assert second_node is first_node
    assert cache.root_node.children[key.child_key(cache.page_size)] is first_node
    assert cache.evictable_leaves == {first_node}
    assert cache.evictable_size() == 4

    result = cache.evict(EvictParams(num_tokens=4))
    assert result.num_tokens_evicted == 4
    assert allocator.free.call_count + allocator.free_segment.call_count == 1
    assert cache.evictable_size() == 0


def test_evict_never_enters_a_cross_rank_store_protocol():
    """evict() fires on local allocator pressure, so it is not rank-symmetric.

    Both store-completion polls scatter across ranks, so a rank that evicted
    would block in a collective its peers never entered. Draining belongs to
    check_hicache_events, which every rank runs each scheduler tick.
    """
    cache, _allocator = _make_cache()
    # The async path is what used to drain stores from inside evict().
    cache._async_store_slot_mapping = True
    key = RadixKey(array("q", range(4)))
    _load(cache, key, 0, 4, "first")

    cache.evict(EvictParams(num_tokens=4))

    cache.flexkv_connector.sync_ready_store_rids.assert_not_called()
    cache.flexkv_connector.check_completed_stores.assert_not_called()
    # The store stream must still be synchronized: eviction frees the source
    # slots a queued D2H copy may still be reading.
    cache.store_stream.synchronize.assert_called_once()

    # The rank-symmetric hook is what actually drains.
    cache.flexkv_connector.check_completed_stores.return_value = []
    cache.check_hicache_events()
    cache.flexkv_connector.check_completed_stores.assert_called_once()
    cache.flexkv_connector.sync_ready_store_rids.assert_called_once()


def _ip_restore(cache, rid="ip-request", uncached_len=4, value_numel=0, key=None):
    """Run an IP-mode (request-owned) restore and return the fake Req."""
    req = SimpleNamespace(
        rid=rid,
        origin_input_ids=[],
        output_ids=[],
        kv=SimpleNamespace(kv_committed_len=0, cache_protected_len=0),
        _flexkv_uncached_restore=False,
        pending_restore_generation=None,
        pending_restore_slots=None,
    )
    result = cache._allocate_and_load(
        key=key if key is not None else RadixKey(array("q", range(uncached_len))),
        value_numel=value_numel,
        uncached_len=uncached_len,
        last_node=cache.root_node,
        tracking_rid=rid,
        sglang_req_id=rid,
        load_fn=MagicMock(side_effect=lambda slots: int(slots.numel())),
        request_owned_req=req,
    )
    assert result is not None
    return req, result


def test_ip_restore_is_reported_uncommitted_until_the_cache_commits_it():
    """The scheduler skips the waiting-queue rematch while this is True.

    Without it, init_next_round_input reassigns req.prefix_indices, which is
    the only handle on request-owned restored slots, and they leak.
    """
    cache, _allocator = _make_cache()
    req, _ = _ip_restore(cache)

    assert cache.has_uncommitted_restore(req) is True

    with (
        patch.object(RadixCache, "cache_finished_req", lambda *a, **k: None),
        patch.dict(
            FlexKVRadixCache.cache_finished_req.__globals__,
            {"get_spec": lambda: SimpleNamespace(speculative_eagle_topk=None)},
        ),
    ):
        cache.cache_finished_req(req, is_insert=False, kv_len_to_handle=0)

    assert cache.has_uncommitted_restore(req) is False
    assert req._flexkv_uncached_restore is False
    assert req.pending_restore_slots is None


def test_mp_restore_never_takes_a_lease():
    """MP restores are attached to the tree, so leasing them would make the
    scheduler skip every MP request forever."""
    cache, _allocator = _make_cache()
    key = RadixKey(array("q", range(4)))
    _load(cache, key, 0, 4, "mp-request")

    assert cache._restore_leases == {}

    # Positive control: leasing does happen on the IP path, so the assertion
    # above is about MP declining a lease, not about leases never being taken.
    # Use a disjoint key so this restore actually allocates instead of being
    # deduplicated against the prefix the MP restore just put in the tree.
    ip_req, _ = _ip_restore(
        cache, rid="ip-request", key=RadixKey(array("q", range(100, 104)))
    )
    assert cache.has_uncommitted_restore(ip_req) is True


def test_lease_covers_only_freshly_allocated_slots_not_the_reused_prefix():
    """Freeing a lease must never free tree-owned slots."""
    cache, _allocator = _make_cache()
    first_page = RadixKey(array("q", range(4)))
    full_key = RadixKey(array("q", range(8)))

    (reused, _node), _ = _load(cache, first_page, 0, 4, "first")
    cache.flexkv_connector.lookup_kv.return_value = (17, 4)
    req, (restored, _last) = _ip_restore(
        cache, rid="ip-second", uncached_len=8, key=full_key
    )

    # The request sees the whole prefix, but the lease owns only the new tail.
    assert restored.numel() == 8
    lease = cache._restore_leases["ip-second"]
    assert lease.device_indices.numel() == 4
    # No leased slot may appear anywhere in the tree-owned reused prefix.
    assert not bool((lease.device_indices == reused.unsqueeze(1)).any())


def test_reset_drains_flexkv_before_freeing_leased_restore_slots():
    """FlexKV still holds these slot addresses: a layerwise H2D may be writing
    into them. Draining must happen before the free."""
    cache, allocator = _make_cache()
    _ip_restore(cache, rid="in-flight")
    order = []
    cache.store_stream.synchronize.side_effect = lambda: order.append("store_stream")
    cache.flexkv_connector.reset.side_effect = lambda: order.append("connector_reset")
    allocator.free.side_effect = lambda *_a, **_k: order.append("free_slots")

    with patch.object(RadixCache, "reset", lambda _self: order.append("base_reset")):
        cache.reset()

    assert order == ["store_stream", "connector_reset", "free_slots", "base_reset"]
    assert cache._restore_leases == {}


def test_short_mp_restore_keeps_the_loaded_prefix():
    """MP retrieve_kv is synchronous, so the unused tail is idle and the
    partially loaded prefix is safe to keep."""
    cache, allocator = _make_cache()

    result = cache._allocate_and_load(
        key=RadixKey(array("q", range(8))),
        value_numel=0,
        uncached_len=8,
        last_node=cache.root_node,
        tracking_rid="short-mp",
        sglang_req_id="short-mp",
        load_fn=MagicMock(return_value=4),
    )

    assert result is not None
    restored, _node = result
    assert restored.numel() == 4
    freed = torch.cat([call.args[0] for call in allocator.free.call_args_list])
    assert freed.numel() == 4  # only the unused tail


def test_partial_duplicate_restore_relooks_up_only_missing_suffix():
    cache, _allocator = _make_cache()
    first_page = RadixKey(array("q", range(4)))
    full_key = RadixKey(array("q", range(8)))

    (first_indices, _first_node), _ = _load(cache, first_page, 0, 4, "first")
    cache.flexkv_connector.lookup_kv.return_value = (17, 4)
    (restored_indices, last_node), second_load = _load(cache, full_key, 0, 8, "second")

    cache.flexkv_connector.release_pending.assert_called_once_with("second")
    lookup = cache.flexkv_connector.lookup_kv.call_args
    assert lookup.kwargs["token_ids"] == full_key.raw_token_ids()
    assert lookup.kwargs["token_mask"].tolist() == [False] * 4 + [True] * 4
    assert second_load.call_args.args[0].numel() == 4
    assert torch.equal(restored_indices[:4], first_indices)
    assert restored_indices.numel() == 8
    assert list(last_node.key.token_ids) == list(full_key[4:].token_ids)
    assert len(cache.evictable_leaves) == 1
    assert cache.evictable_size() == 8

    match = RadixCache.match_prefix(cache, MatchPrefixParams(key=full_key))
    assert torch.equal(match.device_indices, restored_indices)


def test_partial_duplicate_restore_keeps_reused_prefix_when_alloc_fails():
    cache, allocator = _make_cache()
    first_page = RadixKey(array("q", range(4)))
    full_key = RadixKey(array("q", range(8)))

    (first_indices, first_node), _ = _load(cache, first_page, 0, 4, "first")
    cache.flexkv_connector.lookup_kv.return_value = (17, 4)
    allocator.alloc.side_effect = None
    allocator.alloc.return_value = None

    (restored_indices, last_node), second_load = _load(cache, full_key, 0, 8, "second")

    second_load.assert_not_called()
    assert torch.equal(restored_indices, first_indices)
    assert last_node is first_node
    assert cache.evictable_leaves == {first_node}
    assert cache.evictable_size() == 4


def test_ip_match_is_lookup_only_until_request_admission():
    cache, _allocator = _make_cache()
    key = RadixKey(array("q", range(4)))
    base_res = RadixCache.match_prefix(cache, MatchPrefixParams(key=key))
    cache.flexkv_connector.lookup_kv.return_value = (17, 4)
    req = SimpleNamespace(rid="ip-request")

    result = cache._ip_match_prefix(
        key,
        base_res,
        base_res.device_indices,
        base_res.last_device_node,
        req,
    )

    assert result.device_indices.numel() == 0
    assert result.host_hit_length == 4
    assert result.cache_protected_len == 0
    assert result.last_device_node is cache.root_node
    assert cache.evictable_size() == 0
    cache.flexkv_connector.start_load_kv_layerwise.assert_not_called()


def test_request_owned_restore_is_not_attached_before_cache_completion():
    cache, _allocator = _make_cache()
    key = RadixKey(array("q", range(4)))
    req = SimpleNamespace(
        rid="ip-request",
        kv=SimpleNamespace(cache_protected_len=0),
        _flexkv_uncached_restore=False,
    )
    load_fn = MagicMock(side_effect=lambda slots: int(slots.numel()))

    result = cache._allocate_and_load(
        key=key,
        value_numel=0,
        uncached_len=4,
        last_node=cache.root_node,
        tracking_rid="ip-request",
        sglang_req_id="ip-request",
        load_fn=load_fn,
        request_owned_req=req,
    )

    assert result is not None
    restored, last_node = result
    assert restored.numel() == 4
    assert last_node is cache.root_node
    assert req.kv.cache_protected_len == 0
    assert req._flexkv_uncached_restore is True
    assert req._flexkv_restore_tree_owned_len == 0
    assert cache.root_node.children == {}
    assert cache.evictable_size() == 0


def test_finished_request_restores_tree_owned_boundary_before_duplicate_cleanup():
    cache, _allocator = _make_cache()
    req = SimpleNamespace(
        rid="concurrent-restore",
        origin_input_ids=[],
        output_ids=[],
        kv=SimpleNamespace(kv_committed_len=0, cache_protected_len=4),
        _flexkv_uncached_restore=True,
        _flexkv_restore_tree_owned_len=0,
    )
    observed_protected_lengths = []

    def record_base_cleanup(_self, base_req, **_kwargs):
        observed_protected_lengths.append(base_req.kv.cache_protected_len)

    with (
        patch.object(RadixCache, "cache_finished_req", record_base_cleanup),
        patch.dict(
            FlexKVRadixCache.cache_finished_req.__globals__,
            {"get_spec": lambda: SimpleNamespace(speculative_eagle_topk=None)},
        ),
    ):
        cache.cache_finished_req(req, kv_len_to_handle=0)

    assert observed_protected_lengths == [0]
    assert req._flexkv_uncached_restore is False
    assert not hasattr(req, "_flexkv_restore_tree_owned_len")


def test_finished_store_uses_radix_owned_slots_after_request_row_is_cleared():
    cache, allocator = _make_cache(page_size=4)
    request_row = torch.tensor([[4, 5, 6, 7]], dtype=torch.int64)
    cache.req_to_token_pool = SimpleNamespace(req_to_token=request_row)

    # Model the scheduler/request-pool lifecycle that exposed the live fault:
    # after the base cache takes its own slot copy, the request row is no
    # longer a stable source for the asynchronous FlexKV store.
    allocator.free_segments.side_effect = lambda *_args, **_kwargs: request_row.zero_()
    cache.flexkv_connector.store_kv.return_value = 17
    req = SimpleNamespace(
        rid="finished-request",
        origin_input_ids=[1, 2, 3, 4],
        output_ids=[],
        kv=SimpleNamespace(
            kv_committed_len=4,
            req_pool_idx=0,
            cache_protected_len=0,
        ),
        extra_key=None,
        cache_salt=None,
        last_node=cache.root_node,
        _flexkv_uncached_restore=False,
    )

    producer_stream = MagicMock()
    with (
        patch.dict(
            FlexKVRadixCache.cache_finished_req.__globals__,
            {"get_spec": lambda: SimpleNamespace(speculative_eagle_topk=None)},
        ),
        patch("torch.cuda.current_stream", return_value=producer_stream),
        patch(
            "torch.cuda.stream", side_effect=lambda _stream: contextlib.nullcontext()
        ),
    ):
        cache.cache_finished_req(req, kv_len_to_handle=4)

    assert request_row.tolist() == [[0, 0, 0, 0]]
    stored = cache.flexkv_connector.store_kv.call_args.kwargs
    assert stored["token_ids"] == [1, 2, 3, 4]
    assert stored["kv_indices"].tolist() == [4, 5, 6, 7]
    cache.store_stream.wait_stream.assert_called_once_with(producer_stream)


def test_async_store_waits_for_event_then_uses_pinned_cpu_mapping():
    cache, allocator = _make_cache(page_size=4)
    cache._async_store_slot_mapping = True
    request_row = torch.tensor([[4, 5, 6, 7]], dtype=torch.int64)
    cache.req_to_token_pool = SimpleNamespace(req_to_token=request_row)
    allocator.free_segments.side_effect = lambda *_args, **_kwargs: request_row.zero_()
    cache.flexkv_connector.store_kv.return_value = 17
    ready_event = MagicMock()
    ready_event.query.side_effect = [False, True]
    cpu_mapping = torch.tensor([4, 5, 6, 7], dtype=torch.int64)

    def fake_stage(rid, pending):
        cache._pending_store_copies[rid] = SimpleNamespace(
            node=pending.node,
            token_ids=pending.token_ids,
            kv_indices=pending.kv_indices,
            cpu_indices=cpu_mapping,
            ready_event=ready_event,
        )

    req = SimpleNamespace(
        rid="async-store",
        origin_input_ids=[1, 2, 3, 4],
        output_ids=[],
        kv=SimpleNamespace(
            kv_committed_len=4,
            req_pool_idx=0,
            cache_protected_len=0,
        ),
        extra_key=None,
        cache_salt=None,
        last_node=cache.root_node,
        _flexkv_uncached_restore=False,
    )

    with (
        patch.dict(
            FlexKVRadixCache.cache_finished_req.__globals__,
            {"get_spec": lambda: SimpleNamespace(speculative_eagle_topk=None)},
        ),
        patch.object(cache, "_stage_store_copy", side_effect=fake_stage),
        patch(
            "torch.cuda.stream", side_effect=lambda _stream: contextlib.nullcontext()
        ),
    ):
        cache.cache_finished_req(req, kv_len_to_handle=4)
        cache.check_hicache_events()
        cache.flexkv_connector.store_kv.assert_not_called()
        assert list(cache._pending_store_copies) == ["async-store"]

        cache.check_hicache_events()

    stored = cache.flexkv_connector.store_kv.call_args.kwargs
    assert stored["kv_indices"] is cpu_mapping
    assert cache._pending_store_copies == {}
    assert "async-store" in cache._inflight_store_nodes
    cache.store_stream.wait_stream.assert_not_called()


def _prepare_shared_restore():
    cache, allocator = _make_cache()
    cache._mode = FlexKVRadixCache.should_defer_shared_restore.__globals__[
        "FlexKVMode"
    ].IP
    cache._defer_duplicate_restores = True
    key = RadixKey(array("q", range(4)))
    producer = SimpleNamespace(
        rid="producer", kv=SimpleNamespace(cache_protected_len=0, holds_mamba=False)
    )
    cache._allocate_and_load(
        key=key,
        value_numel=0,
        uncached_len=4,
        last_node=cache.root_node,
        tracking_rid=producer.rid,
        sglang_req_id=producer.rid,
        load_fn=lambda slots: int(slots.numel()),
        request_owned_req=producer,
    )
    waiter = SimpleNamespace(
        rid="waiter", host_hit_length=4, kv=SimpleNamespace(holds_mamba=False)
    )
    cache._load_markers[waiter.rid] = SimpleNamespace(key=key, value_numel=0)
    return cache, allocator, producer, waiter


def test_shared_restore_waits_without_borrowing_request_owned_slots():
    cache, allocator, producer, waiter = _prepare_shared_restore()
    assert cache.should_defer_shared_restore(waiter)
    cache.flexkv_connector.release_pending.assert_called_once_with(waiter.rid)
    assert waiter.rid not in cache._load_markers
    assert cache._restore_prefix_by_rid[producer.rid] in cache._restoring_host_prefixes
    assert allocator.alloc.call_count == 1
    assert cache.root_node.children == {}
    with patch.object(
        RadixCache, "cache_unfinished_req", lambda *_args, **_kwargs: None
    ):
        cache.cache_unfinished_req(producer)
    cache._load_markers[waiter.rid] = SimpleNamespace(
        key=RadixKey(array("q", range(4))), value_numel=0
    )
    assert not cache.should_defer_shared_restore(waiter)
    assert cache._restore_prefix_by_rid == {}
    assert cache._restoring_host_prefixes == {}


def test_aborting_restore_producer_releases_duplicate_admission():
    cache, allocator, producer, waiter = _prepare_shared_restore()
    assert cache.should_defer_shared_restore(waiter)
    cache.release_aborted_request(producer.rid)
    cache._load_markers[waiter.rid] = SimpleNamespace(
        key=RadixKey(array("q", range(4))), value_numel=0
    )
    assert not cache.should_defer_shared_restore(waiter)
    assert not cache.has_uncommitted_restore(producer)
    assert (
        cache._aborted_restore_leases[producer.pending_restore_generation].req
        is producer
    )
    assert producer._flexkv_uncached_restore
    allocator.free.assert_not_called()
    assert cache.flexkv_connector.release_pending.call_args_list == [
        call(waiter.rid),
        call(producer.rid),
    ]


def test_shared_restore_respects_tenant_identity_and_disabled_flag():
    cache, _allocator, _producer, waiter = _prepare_shared_restore()
    for key in [
        RadixKey(array("q", range(4)), extra_key="tenant"),
        RadixKey(array("q", range(4)), cache_salt="other"),
        RadixKey(array("q", range(4, 8))),
    ]:
        cache._load_markers[waiter.rid].key = key
        assert not cache.should_defer_shared_restore(waiter)
    cache._load_markers[waiter.rid].key = RadixKey(array("q", range(4)))
    cache._defer_duplicate_restores = False
    assert not cache.should_defer_shared_restore(waiter)


@pytest.mark.parametrize("guard", ["mamba", "mp", "no_host_hit", "missing_marker"])
def test_shared_restore_skips_unsupported_requests(guard):
    cache, allocator, _producer, waiter = _prepare_shared_restore()
    if guard == "mamba":
        waiter.kv.holds_mamba = True
    elif guard == "mp":
        cache._mode = FlexKVRadixCache.should_defer_shared_restore.__globals__[
            "FlexKVMode"
        ].MP
    elif guard == "no_host_hit":
        waiter.host_hit_length = 0
    else:
        cache._load_markers.pop(waiter.rid)
    assert not cache.should_defer_shared_restore(waiter)
    cache.flexkv_connector.release_pending.assert_not_called()
    assert allocator.alloc.call_count == 1


def test_shared_restore_bigram_identity_includes_boundary_token():
    cache, _allocator = _make_cache()
    raw = array("q", [1, 2, 3, 4, 5])
    key = RadixKey(raw, is_bigram=True)
    prefix = cache._restore_prefix_key(key, 3)
    assert prefix[-1] == (1, 2, 3, 4)
    assert prefix != cache._restore_prefix_key(RadixKey(raw), 3)
    assert prefix != cache._restore_prefix_key(
        RadixKey(array("q", [1, 2, 3, 9, 5]), is_bigram=True), 3
    )


def test_repeated_duplicate_deferral_releases_every_held_lookup():
    cache, allocator, producer, waiter = _prepare_shared_restore()
    for _ in range(3):
        cache._load_markers[waiter.rid] = SimpleNamespace(
            key=RadixKey(array("q", range(4))), value_numel=0
        )
        assert cache.should_defer_shared_restore(waiter)
        assert waiter.rid not in cache._load_markers
    assert (
        cache.flexkv_connector.release_pending.call_args_list == [call(waiter.rid)] * 3
    )
    assert allocator.alloc.call_count == 1
    assert producer.rid in cache._restore_prefix_by_rid


class _TestReq(SimpleNamespace):
    __hash__ = object.__hash__


def _restore_request(cache, *, rid="ip-request", length=4, load_fn=None):
    req = _TestReq(
        rid=rid,
        origin_input_ids=array("q", range(length)),
        output_ids=array("q"),
        kv=SimpleNamespace(
            kv_committed_len=length, cache_protected_len=0, req_pool_idx=0
        ),
        _flexkv_uncached_restore=False,
        req_pool_idx=0,
        extra_key=None,
        cache_salt=None,
        last_node=cache.root_node,
        get_fill_ids=lambda: array("q", range(length)),
    )
    result = cache._allocate_and_load(
        key=RadixKey(array("q", range(length))),
        value_numel=0,
        uncached_len=length,
        last_node=cache.root_node,
        tracking_rid=rid,
        sglang_req_id=rid,
        load_fn=load_fn or (lambda slots: int(slots.numel())),
        request_owned_req=req,
    )
    if result is not None:
        req.prefix_indices, req.last_node = result
        req.kv.cache_protected_len = len(req.prefix_indices)
        row = req.prefix_indices.unsqueeze(0).clone()
        cache.req_to_token_pool = SimpleNamespace(
            req_to_token=row,
            write=lambda index, value: row.__setitem__(index, value),
        )
    return req


@pytest.mark.parametrize("completion", ["insert", "discard", "chunk"])
@pytest.mark.parametrize("dedup", [False, True])
def test_ip_restore_lease_ends_after_real_cache_completion(completion, dedup):
    cache, allocator = _make_cache()
    cache._defer_duplicate_restores = dedup
    req = _restore_request(cache)
    assert cache.has_uncommitted_restore(req)
    with (
        patch.dict(
            FlexKVRadixCache.cache_finished_req.__globals__,
            {"get_spec": lambda: SimpleNamespace(speculative_eagle_topk=None)},
        ),
        patch.object(cache, "_launch_store", return_value=-1),
    ):
        if completion == "chunk":
            cache.cache_unfinished_req(req, chunked=True)
        else:
            cache.cache_finished_req(
                req, is_insert=completion == "insert", kv_len_to_handle=4
            )
    assert not cache.has_uncommitted_restore(req)
    assert req.pending_restore_slots is None
    assert req.pending_restore_generation is None
    assert not req._flexkv_uncached_restore
    assert cache._restore_prefix_by_rid == {}
    assert cache._restoring_host_prefixes == {}
    if completion == "discard":
        assert allocator.free_segments.call_args.args[0][0][0].numel() == 4
    else:
        match = RadixCache.match_prefix(
            cache, MatchPrefixParams(key=RadixKey(array("q", range(4))))
        )
        assert torch.equal(match.device_indices, req.prefix_indices)


@pytest.mark.parametrize("dedup", [False, True])
def test_abort_unblocks_rid_but_retains_slots_until_request_cleanup(dedup):
    cache, allocator = _make_cache()
    cache._defer_duplicate_restores = dedup
    req = _restore_request(cache)
    restored = req.pending_restore_slots
    cache.release_aborted_request(req.rid)
    # Abort notification is not completion of the asynchronous H2D writer.
    assert not cache.has_uncommitted_restore(req)
    assert cache._aborted_restore_leases[req.pending_restore_generation].req is req
    assert req._flexkv_uncached_restore
    assert cache._restore_prefix_by_rid == {}
    assert cache._restoring_host_prefixes == {}
    allocator.free.assert_not_called()
    cache.cache_finished_req(req, is_insert=False, kv_len_to_handle=4)
    released, start = allocator.free_segments.call_args.args[0][0]
    assert start == 0
    assert torch.equal(released, restored)
    assert not cache.has_uncommitted_restore(req)
    assert cache._aborted_restore_leases == {}


def test_mp_restore_is_tree_owned_and_ip_lease_excludes_reused_prefix():
    cache, allocator = _make_cache()
    (reused, _), _ = _load(cache, RadixKey(array("q", range(4))), 0, 4, "mp")
    assert not cache.has_uncommitted_restore(SimpleNamespace(rid="mp"))
    cache.flexkv_connector.lookup_kv.return_value = (17, 4)
    req = _restore_request(cache, length=8)
    lease = cache._restore_leases[req.rid]
    assert req.prefix_indices.numel() == 8
    assert torch.equal(req.prefix_indices[:4], reused)
    assert torch.equal(lease.device_indices, req.prefix_indices[4:])
    allocator.free.reset_mock()
    cache.reset()
    assert torch.equal(allocator.free.call_args.args[0], lease.device_indices)
    assert not cache.has_uncommitted_restore(req)


@pytest.mark.parametrize("mismatch", ["identity", "generation", "slots"])
@pytest.mark.parametrize("method", ["cache_finished_req", "cache_unfinished_req"])
@pytest.mark.parametrize("dedup", [False, True])
def test_restore_lease_mismatch_fails_before_mutating_cache(mismatch, method, dedup):
    cache, allocator = _make_cache()
    cache._defer_duplicate_restores = dedup
    req = _restore_request(cache)
    slots = req.pending_restore_slots
    live = _assert_allocator_free_once(allocator, slots)
    original_root = cache.root_node
    if mismatch == "identity":
        req = SimpleNamespace(**vars(req))
    elif mismatch == "generation":
        req.pending_restore_generation += 1
    else:
        req.pending_restore_slots = req.pending_restore_slots.clone()
    # Exercise the real base cache and track actual slot ownership: logging and
    # continuing here would free/insert these slots, then reset would free again.
    with pytest.raises(RuntimeError, match="restore lease mismatch"):
        if method == "cache_finished_req":
            cache.cache_finished_req(req, is_insert=False, kv_len_to_handle=4)
        else:
            cache.cache_unfinished_req(req)
    assert live == set(slots.tolist())
    assert cache.root_node is original_root
    assert not cache.root_node.children
    assert cache.has_uncommitted_restore(req)
    assert (req.rid in cache._restore_prefix_by_rid) is dedup

    cache.reset()
    assert not live


@pytest.mark.parametrize("entry", ["match", "init", "allocate"])
def test_duplicate_restore_cannot_replace_active_lease(entry):
    cache, allocator = _make_cache()
    req = _restore_request(cache)
    lease = cache._restore_leases[req.rid]
    allocator.alloc.reset_mock()
    cache.flexkv_connector.reset_mock()
    with pytest.raises(RuntimeError, match="before restore commit|duplicate load-back"):
        if entry == "match":
            cache.match_prefix(
                MatchPrefixParams(key=RadixKey(array("q", range(4))), req=req)
            )
        elif entry == "init":
            cache.init_load_back(
                InitLoadBackParams(
                    best_match_node=req.last_node, host_hit_length=4, req=req
                )
            )
        else:
            _restore_request(cache)
    assert cache._restore_leases[req.rid] is lease
    allocator.alloc.assert_not_called()
    assert cache.flexkv_connector.mock_calls == []


@pytest.mark.parametrize("drain_fails", [False, True])
@pytest.mark.parametrize("dedup", [False, True])
def test_restore_reset_preserves_ownership_until_transfers_drain(drain_fails, dedup):
    cache, allocator = _make_cache()
    cache._defer_duplicate_restores = dedup
    req = _restore_request(cache)
    original_root = cache.root_node
    order = []
    cache.store_stream.synchronize.side_effect = lambda: order.append("stream")

    def drain():
        order.append("connector")
        if drain_fails:
            raise RuntimeError("H2D still active")

    cache.flexkv_connector.reset.side_effect = drain
    allocator.free.side_effect = lambda *_args: order.append("free")
    if drain_fails:
        with pytest.raises(RuntimeError, match="H2D still active"):
            cache.reset()
        assert order == ["stream", "connector"]
        assert cache.root_node is original_root
        assert cache.has_uncommitted_restore(req)
        assert (req.rid in cache._restore_prefix_by_rid) is dedup
    else:
        with patch.object(
            RadixCache, "reset", side_effect=lambda: order.append("tree")
        ):
            cache.reset()
        assert order == ["stream", "connector", "free", "tree"]
        assert not cache.has_uncommitted_restore(req)
        assert cache._restore_prefix_by_rid == {}
        assert cache._restoring_host_prefixes == {}


def test_failed_launch_keeps_allocated_slots_for_reset():
    cache, allocator = _make_cache()
    with pytest.raises(RuntimeError, match="unknown launch status"):
        _restore_request(
            cache, load_fn=MagicMock(side_effect=RuntimeError("unknown launch status"))
        )
    lease = cache._restore_leases["ip-request"]
    allocator.free.assert_not_called()
    cache.reset()
    assert torch.equal(allocator.free.call_args.args[0], lease.device_indices)


def test_zero_length_restore_releases_allocation_and_lease():
    cache, allocator = _make_cache()
    req = _restore_request(cache, load_fn=lambda _slots: 0)
    assert not cache.has_uncommitted_restore(req)
    assert allocator.free.call_args.args[0].numel() == 4


def test_short_layerwise_restore_retains_the_full_allocation_until_reset():
    cache, allocator = _make_cache()
    with pytest.raises(RuntimeError, match="Unexpected layerwise restore length"):
        _restore_request(cache, length=8, load_fn=lambda _slots: 4)
    lease = cache._restore_leases["ip-request"]
    assert lease.device_indices.numel() == 8
    allocator.free.assert_not_called()
    cache.reset()
    assert torch.equal(allocator.free.call_args.args[0], lease.device_indices)
    assert cache._restore_leases == {}


def test_aborted_shared_producer_keeps_slots_while_waiter_allocates_its_own():
    cache, allocator, producer, waiter = _prepare_shared_restore()
    producer_slots = producer.pending_restore_slots
    cache.release_aborted_request(producer.rid)
    assert not cache.should_defer_shared_restore(waiter)

    restored, _ = cache._allocate_and_load(
        key=RadixKey(array("q", range(4))),
        value_numel=0,
        uncached_len=4,
        last_node=cache.root_node,
        tracking_rid=waiter.rid,
        sglang_req_id=waiter.rid,
        load_fn=lambda slots: int(slots.numel()),
        request_owned_req=waiter,
    )
    assert not cache.has_uncommitted_restore(producer)
    assert (
        cache._aborted_restore_leases[producer.pending_restore_generation].req
        is producer
    )
    assert cache.has_uncommitted_restore(waiter)
    assert not bool((restored == producer_slots.unsqueeze(1)).any())
    allocator.free.assert_not_called()
    # Late cleanup of the old producer must not clear its successor's marker.
    cache._release_restore_prefix(producer.rid)
    assert set(cache._restoring_host_prefixes.values()) == {waiter.rid}

    order = []
    cache.flexkv_connector.reset.side_effect = lambda: order.append("drain")
    allocator.free.side_effect = lambda slots: order.append(slots.tolist())
    cache.reset()
    assert order[0] == "drain"
    assert sorted(order[1:]) == sorted([producer_slots.tolist(), restored.tolist()])
    assert cache._aborted_restore_leases == {}
    assert cache._restore_leases == {}
    assert cache._restoring_host_prefixes == {}
    assert cache._restore_prefix_by_rid == {}


def _scheduler_method(name, namespace):
    path = (
        Path(__file__).resolve().parents[4] / "python/sglang/srt/managers/scheduler.py"
    )
    cls = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "Scheduler"
    )
    method = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            method,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


def _assert_allocator_free_once(allocator, slots):
    live = set(slots.tolist())

    def release(values, **_kwargs):
        for slot in values.tolist():
            assert slot in live, f"double free: {slot}"
            live.remove(slot)

    allocator.free.side_effect = release
    allocator.free_segment.side_effect = release
    allocator.free_segments.side_effect = lambda spans: [
        release(values) for values, _ in spans
    ]
    return live


@pytest.mark.parametrize("abort_path", ["limit", "timeout", "explicit"])
def test_scheduler_queue_abort_without_kv_row_retains_reclaimable_allocation(
    abort_path,
):
    cache, allocator = _make_cache()
    req = _restore_request(cache)
    slots = req.pending_restore_slots
    live = _assert_allocator_free_once(allocator, slots)
    # Layerwise allocation precedes prepare_for_extend's request-row assignment.
    req.req_pool_idx = None
    req.kv = SimpleNamespace(req_pool_idx=None, holds_kv=False, holds_mamba=False)
    req.mamba_pool_idx = None
    req.priority = 10
    req.time_stats = SimpleNamespace(wait_queue_entry_time=1, trace_ctx=MagicMock())
    scheduler = SimpleNamespace(
        tree_cache=cache,
        waiting_queue=[req],
        enable_hicache_storage=True,
        enable_hierarchical_cache=False,
        enable_priority_scheduling=True,
        schedule_low_priority_values_first=True,
        max_queued_requests=1,
        ipc_channels=SimpleNamespace(send_to_tokenizer=MagicMock()),
        chunked_req=None,
        dllm_config=None,
        disaggregation_mode=None,
        grammar_manager=MagicMock(),
        ps=SimpleNamespace(pp_size=1),
        running_batch=None,
        last_batch=None,
        is_fully_idle=lambda: True,
        req_to_token_pool=MagicMock(),
        token_to_kv_pool_allocator=allocator,
        metrics_reporter=MagicMock(),
        draft_worker=None,
        beam_coordinator=MagicMock(),
        mm_receiver=None,
    )
    release_kv = MagicMock(side_effect=AssertionError("no KV row exists"))
    namespace = {
        "AbortReq": lambda **kw: SimpleNamespace(**kw),
        "_make_abort_req": lambda req, **kw: SimpleNamespace(rid=req.rid, **kw),
        "HTTPStatus": SimpleNamespace(SERVICE_UNAVAILABLE=503),
        "envs": SimpleNamespace(
            SGLANG_REQ_WAITING_TIMEOUT=SimpleNamespace(get=lambda: 1)
        ),
        "time": SimpleNamespace(perf_counter=lambda: 10),
        "logger": MagicMock(),
        "logging": MagicMock(),
        "DisaggregationMode": SimpleNamespace(DECODE="decode", PREFILL="prefill"),
        "release_kv_cache": release_kv,
    }
    scheduler._release_aborted_request = lambda rid: _scheduler_method(
        "_release_aborted_request", namespace
    )(scheduler, rid)
    scheduler.collect_inflight_reqs = lambda: _scheduler_method(
        "collect_inflight_reqs", namespace
    )(scheduler)
    if abort_path == "limit":
        _scheduler_method("_abort_on_queued_limit", namespace)(
            scheduler, SimpleNamespace(rid="incoming", priority=0)
        )
    elif abort_path == "timeout":
        _scheduler_method("_abort_on_waiting_timeout", namespace)(scheduler)
    else:
        _scheduler_method("abort_request", namespace)(
            scheduler, SimpleNamespace(rid=req.rid, abort_all=False)
        )
    assert scheduler.waiting_queue == []
    release_kv.assert_not_called()
    assert not cache.has_uncommitted_restore(req)
    assert live == set(slots.tolist())  # Nothing was freed while H2D may run.
    assert (
        cache._aborted_restore_leases[req.pending_restore_generation].device_indices
        is slots
    )

    def fence():
        assert live == set(slots.tolist())

    cache.flexkv_connector.reset.side_effect = fence
    assert _scheduler_method("flush_cache", namespace)(scheduler, empty_cache=False)
    assert live == set()
    assert cache._aborted_restore_leases == {}


@pytest.mark.parametrize("dedup", [False, True])
def test_aborted_request_cleanup_does_not_free_or_commit_reused_rid(dedup):
    cache, allocator = _make_cache()
    cache._defer_duplicate_restores = dedup
    old = _restore_request(cache, rid="reused")
    old_slots = old.pending_restore_slots
    old_pool = cache.req_to_token_pool
    cache.release_aborted_request(old.rid)
    new = _restore_request(cache, rid="reused")
    new_slots = new.pending_restore_slots
    new_pool = cache.req_to_token_pool
    live = _assert_allocator_free_once(allocator, torch.cat([old_slots, new_slots]))
    cache.req_to_token_pool = old_pool
    cache.cache_finished_req(old, is_insert=False, kv_len_to_handle=4)
    assert live == set(new_slots.tolist())
    assert cache._restore_leases[new.rid].req is new
    assert (new.rid in cache._restore_prefix_by_rid) is dedup
    assert (new.rid in cache._restoring_host_prefixes.values()) is dedup
    assert cache._aborted_restore_leases == {}
    cache.req_to_token_pool = new_pool
    cache.cache_finished_req(new, is_insert=False, kv_len_to_handle=4)
    cache.reset()
    assert live == set()


@pytest.mark.parametrize("stale", ["generation", "slots"])
def test_reset_reclaims_ledger_even_when_request_metadata_is_stale(stale):
    cache, allocator = _make_cache()
    old = _restore_request(cache, rid="old")
    old_slots = old.pending_restore_slots
    cache.release_aborted_request(old.rid)
    active = _restore_request(cache, rid="active")
    active_slots = active.pending_restore_slots
    live = _assert_allocator_free_once(allocator, torch.cat([old_slots, active_slots]))
    for req in (old, active):
        if stale == "generation":
            req.pending_restore_generation += 100
        else:
            req.pending_restore_slots = req.pending_restore_slots.clone()
    cache.reset()
    assert live == set()
    assert cache._restore_leases == {}
    assert cache._aborted_restore_leases == {}


def test_reset_attempts_other_allocations_and_retains_failed_free():
    cache, allocator = _make_cache()
    first = _restore_request(cache, rid="first")
    second = _restore_request(cache, rid="second")
    first_slots, second_slots = (
        first.pending_restore_slots,
        second.pending_restore_slots,
    )

    def release(slots):
        if slots is first_slots:
            raise RuntimeError("allocator failure")

    allocator.free.side_effect = release
    with patch.object(RadixCache, "reset") as reset_tree:
        with pytest.raises(RuntimeError, match="failed to free restore allocations"):
            cache.reset()
        reset_tree.assert_not_called()
    assert allocator.free.call_args_list == [call(first_slots), call(second_slots)]
    assert set(cache._restore_leases) == {"first"}
    allocator.free.reset_mock(side_effect=True)
    cache.reset()
    allocator.free.assert_called_once_with(first_slots)
