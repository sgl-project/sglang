import importlib.util
import sys
from types import ModuleType, SimpleNamespace
from unittest import mock

import pytest
import torch

from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.mem_cache.base_prefix_cache import (
    CacheRequestHandle,
    IncLockRefResult,
    InsertResult,
    MatchResult,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


@pytest.fixture
def cache_class():
    # Exercise the real cache module without installing the optional MP service.
    metadata = ModuleType("lmcache.integration.sglang.lmcache_mp_metadata")
    metadata.LMCacheExternalFlow = mock.Mock()
    metadata.LMCachePendingStore = mock.Mock()
    connector = ModuleType("lmcache.integration.sglang.unified_lmcache_mp_connector")
    connector.UnifiedLMCacheMPConnector = mock.Mock()
    with mock.patch.dict(
        sys.modules, {metadata.__name__: metadata, connector.__name__: connector}
    ):
        spec = importlib.util.find_spec(
            "sglang.srt.mem_cache.storage.lmcache.lmcache_unified_radix_cache"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module.LMCacheUnifiedRadixCache


def _cache_and_request(cache_class, *, swa_enabled, swa_cursor=0):
    cache = cache_class.__new__(cache_class)
    cache.is_swa_enabled = swa_enabled
    cache.lmcache_connector = SimpleNamespace(aligned_swa_window_size=lambda: 4)
    flow = SimpleNamespace(
        load=SimpleNamespace(local_hit_tokens=2, device_indices=torch.arange(2, 8)),
        loaded_skip_tokens=0,
        total_hit=8,
        key=RadixKey(list(range(8))),
        prefix_published=False,
        mamba_value=None,
        free_mamba_after_load=False,
        request_mamba_value=None,
        allocated_request_mamba_for_load=False,
        load_req=None,
    )
    cache._external_flows = {"request": flow}
    req = SimpleNamespace(
        rid="request",
        kv=ReqKvInfo(
            req_pool_idx=0,
            kv_allocated_len=12,
            cache_protected_len=8,
            component_evicted_seqlens={ComponentType.AUXILIARY_SWA: 6},
        ),
        priority=0,
        last_node=None,
        lock=None,
    )
    if swa_enabled:
        req.kv.set_evicted_seqlen(ComponentType.SWA, swa_cursor)
    return cache, req, flow


@pytest.mark.parametrize("swa_cursor", [0, 6])
def test_external_swa_load_preserves_independent_cursors(cache_class, swa_cursor):
    cache, req, flow = _cache_and_request(
        cache_class, swa_enabled=True, swa_cursor=swa_cursor
    )

    cache._publish_external_loaded_prefix(req, token_ids_len=12)

    assert flow.prefix_published
    assert req.kv.cache_protected_len == 2
    assert req.kv.component_evicted_seqlens == {
        ComponentType.SWA: max(swa_cursor, 4),
        ComponentType.AUXILIARY_SWA: 6,
    }


@pytest.mark.parametrize("swa_enabled", [False, True])
def test_mamba_publication_snapshots_component_cursors(cache_class, swa_enabled):
    cache, req, flow = _cache_and_request(cache_class, swa_enabled=swa_enabled)
    flow.mamba_value = torch.tensor([1], dtype=torch.int64)
    cache.req_to_token_pool = mock.Mock()
    cache.req_to_token_pool.req_to_token = torch.arange(12).reshape(1, 12)
    cache.insert = mock.Mock(return_value=InsertResult(prefix_len=8))
    cache.inc_lock_ref = mock.Mock(return_value=IncLockRefResult(node_id=42))
    matched = MatchResult(
        device_prefix_len=8,
        last_device_node=42,
        last_host_node=None,
        best_match_node=None,
    )

    with (
        mock.patch.object(UnifiedRadixCache, "match_prefix", return_value=matched),
        mock.patch.object(cache, "path_device_indices", return_value=torch.arange(8)),
    ):
        cache._publish_external_loaded_prefix(req, token_ids_len=12)

    insert_params = cache.insert.call_args.args[0]
    expected = {ComponentType.AUXILIARY_SWA: 6}
    if swa_enabled:
        expected[ComponentType.SWA] = 4
    assert insert_params.component_evicted_seqlens == expected
    assert (
        insert_params.component_evicted_seqlens is not req.kv.component_evicted_seqlens
    )
    req.kv.set_evicted_seqlen(ComponentType.AUXILIARY_SWA, 99)
    assert insert_params.get_evicted_seqlen(ComponentType.AUXILIARY_SWA) == 6
    assert req.kv.cache_protected_len == 8
    assert req.lock.receipt.node_id == 42
    assert flow.prefix_published
    assert flow.mamba_value is None


def test_prefill_session_finishes_only_after_the_last_store(cache_class):
    cache = cache_class.__new__(cache_class)
    cache._external_flows = {}
    first = SimpleNamespace(request_id="prefill")
    last = SimpleNamespace(request_id="prefill")
    cache._pending_stores = [
        SimpleNamespace(operation=first, node_id=10, lock_params="first"),
        SimpleNamespace(operation=last, node_id=20, lock_params="last"),
    ]
    cache._pending_store_counts = {"prefill": 2}
    cache._session_finish_requested = set()
    cache.dec_lock_ref = mock.Mock()
    cache.lmcache_connector = mock.Mock()
    cache.lmcache_connector.ready_prefix_count.side_effect = [0, 1, 0, 1]

    # PD transfer can finish while external stores still read the same KV.
    cache._request_session_finish("prefill")
    cache.lmcache_connector.finish_request.assert_not_called()
    cache.check_hicache_events()
    cache.dec_lock_ref.assert_called_once_with(10, "first")
    cache.lmcache_connector.finish_request.assert_not_called()
    assert cache.has_pending_cache_operations()

    cache.check_hicache_events()
    cache.dec_lock_ref.assert_any_call(20, "last")
    cache.lmcache_connector.finish_request.assert_called_once_with("prefill")
    assert not cache.has_pending_cache_operations()
    assert not cache._pending_store_counts


def test_prefill_abort_defers_retrieve_cleanup_until_completion(cache_class):
    cache = cache_class.__new__(cache_class)
    handle = CacheRequestHandle("prefill", 0)
    flow = SimpleNamespace(
        load=SimpleNamespace(),
        load_completed=False,
        mamba_value=None,
        cancelled=False,
        retire_requested=False,
    )
    cache._external_flows = {"prefill": flow}
    cache._pending_stores = []
    cache._pending_store_counts = {}
    cache._session_finish_requested = set()
    cache.prefetch_loaded_tokens_by_reqid = {handle: 8}
    cache.prefetch_loaded_storage_start_by_reqid = {handle: 0}
    cache.lmcache_connector = mock.Mock()
    cache.lmcache_connector.ready_prefix_count.side_effect = [0, 0, 1, 0]
    cache._finish_failed_load = mock.Mock()

    cache.release_aborted_request(handle)
    assert flow.cancelled and flow.retire_requested
    assert handle not in cache.prefetch_loaded_tokens_by_reqid
    cache._finish_failed_load.assert_not_called()
    cache.check_hicache_events()
    cache._finish_failed_load.assert_not_called()
    cache.check_hicache_events()
    cache._finish_failed_load.assert_called_once_with(flow)
    assert flow.load_completed


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
