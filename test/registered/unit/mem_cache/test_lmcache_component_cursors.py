import importlib.util
import sys
from types import ModuleType, SimpleNamespace
from unittest import mock

import pytest
import torch

from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.mem_cache.base_prefix_cache import (
    IncLockRefResult,
    InsertResult,
    MatchResult,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


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


def test_check_hicache_events_drains_pending_stores_fifo_with_deque(cache_class):
    from collections import deque

    cache = cache_class.__new__(cache_class)
    cache._external_flows = {}
    op1 = SimpleNamespace(request_id="req1")
    op2 = SimpleNamespace(request_id="req1")
    op3 = SimpleNamespace(request_id="req2")
    cache._pending_stores = deque(
        [
            SimpleNamespace(operation=op1, node_id=10, lock_params="lock1"),
            SimpleNamespace(operation=op2, node_id=20, lock_params="lock2"),
            SimpleNamespace(operation=op3, node_id=30, lock_params="lock3"),
        ]
    )
    cache._pending_store_counts = {"req1": 2, "req2": 1}
    cache._session_finish_requested = set()
    completed_ops = []
    cache.lmcache_connector = SimpleNamespace(
        ready_prefix_count=lambda ops: 2 if len(ops) == 3 else 0,
        complete_store=lambda op: completed_ops.append(op),
    )
    unlocked = []
    cache.dec_lock_ref = lambda node_id, lock_params: unlocked.append(
        (node_id, lock_params)
    )
    cache._finish_session_if_store_idle = mock.Mock()

    cache.check_hicache_events()

    assert completed_ops == [op1, op2]
    assert unlocked == [(10, "lock1"), (20, "lock2")]
    assert len(cache._pending_stores) == 1
    assert cache._pending_stores[0].operation is op3
    assert "req1" not in cache._pending_store_counts
    assert cache._pending_store_counts["req2"] == 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
