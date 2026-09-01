import contextlib
import importlib.util
import sys
import threading
from array import array
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.mem_cache.base_prefix_cache import EvictParams, MatchPrefixParams
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
    cache.flexkv_connector = MagicMock()
    cache.store_stream = MagicMock()
    cache._load_markers = {}
    cache._inflight_store_nodes = {}
    cache._pending_store_launches = {}
    cache._pending_store_copies = {}
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
