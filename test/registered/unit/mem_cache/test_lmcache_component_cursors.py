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


@pytest.mark.parametrize("first_completed", ["load", "store", "no_store"])
@pytest.mark.parametrize("load_success", [True, False])
def test_aborted_request_waits_for_load_and_store(
    cache_class, first_completed, load_success
):
    cache, req, flow = _cache_and_request(cache_class, swa_enabled=False)
    handle = CacheRequestHandle(rid=req.rid, attempt_id=0)
    req.cache_request_handle = handle
    flow.lookup = SimpleNamespace(request_id=req.rid)
    flow.load_req = req
    flow.load_completed = False
    flow.released_skip_tokens = 0
    flow.anchor_node = flow.anchor_lock = None
    flow.load.ready = False
    flow.load.result = None
    cache.prefetch_loaded_tokens_by_reqid = {}
    cache.prefetch_loaded_storage_start_by_reqid = {}
    cache._session_finish_requested = set()
    store = SimpleNamespace(request_id=req.rid, ready=False)
    has_store = first_completed != "no_store"
    cache._pending_stores = (
        [SimpleNamespace(operation=store, node_id=42, lock_params=object())]
        if has_store
        else []
    )
    cache._pending_store_counts = {req.rid: 1} if has_store else {}
    cache.token_to_kv_pool_allocator = mock.Mock()
    cache.dec_lock_ref = mock.Mock()
    connector = cache.lmcache_connector = mock.Mock()
    connector.ready_prefix_count.side_effect = lambda operations: sum(
        op.ready for op in operations
    )

    def complete_load(operation):
        operation.result = load_success
        return load_success

    connector.complete_load.side_effect = complete_load
    with mock.patch.object(UnifiedRadixCache, "on_release"):
        cache.on_release(req, checkpointed=False)
    cache.check_hicache_events()
    connector.finish_request.assert_not_called()
    cache.token_to_kv_pool_allocator.free.assert_not_called()

    if has_store:
        first = flow.load if first_completed == "load" else store
        first.ready = True
        cache.check_hicache_events()
        connector.finish_request.assert_not_called()
    flow.load.ready = store.ready = True
    cache.check_hicache_events()
    cache.check_hicache_events()
    connector.finish_request.assert_called_once_with(req.rid)
    connector.complete_load.assert_called_once_with(flow.load)
    cache.token_to_kv_pool_allocator.free.assert_called_once()
    torch.testing.assert_close(
        cache.token_to_kv_pool_allocator.free.call_args.args[0],
        flow.load.device_indices,
    )
    assert not cache.has_pending_cache_operations()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
