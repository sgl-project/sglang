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
    InitLoadBackParams,
    InsertResult,
    MatchResult,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.storage.lmcache.load_cost_profile import LoadCostKey
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


@pytest.mark.parametrize(
    "minimum, length, local, submits",
    [
        (0, 255, 0, True),
        (512, 511, 0, False),
        (512, 512, 0, True),
        (512, 1024, 900, True),
    ],
)
def test_lmcache_lookup_admission_uses_rank_common_upper_bound(
    cache_class, minimum, length, local, submits
):
    cache = cache_class.__new__(cache_class)
    cache._external_flows = {}
    cache._min_retrieve_tokens = minimum
    cache.tree_core = SimpleNamespace(
        page_size=1, is_eagle=False, prefetch_anchor_info=lambda _: (None, None)
    )
    cache.lmcache_connector = mock.Mock(chunk_size=256)
    handle = CacheRequestHandle("request", 0)
    tokens = list(range(length))

    cache.prefetch_from_storage(
        handle, None, tokens[local:], matched_prefix_tokens=tokens[:local]
    )

    assert cache.lmcache_connector.submit_lookup.called is submits
    assert bool(cache._external_flows) is submits


@pytest.mark.parametrize(
    "minimum, total, local, peer_local, expected_hit, release_end, cost_load",
    [
        (0, 256, 255, 255, 1, 0, None),
        (256, 512, 300, 300, 0, 512, None),
        (256, 512, 256, 256, 256, 256, None),
        (256, 512, 500, 0, 512, 0, None),
        (256, 256, 512, 512, 0, 256, None),
        (0, 512, 500, 0, 0, 512, False),
        (0, 512, 500, 0, 512, 0, True),
        (0, 512, 300, 300, 0, 512, False),
        (0, 512, 300, 300, 212, 256, True),
        (512, 256, 0, 0, 0, 256, True),
    ],
)
def test_lmcache_short_hit_releases_locks_and_preserves_rank_consensus(
    cache_class, minimum, total, local, peer_local, expected_hit, release_end, cost_load
):
    cache = cache_class.__new__(cache_class)
    handle = CacheRequestHandle("request", 0)
    flow = SimpleNamespace(
        cancelled=False,
        total_hit=None,
        key=RadixKey(list(range(1024))),
        lookup=object(),
    )
    cache._external_flows = {handle.rid: flow}
    cache._min_retrieve_tokens = minimum
    cache.prefetch_loaded_tokens_by_reqid = {}
    cache._load_cost_profile = (
        mock.Mock(should_load=mock.Mock(return_value=cost_load))
        if cost_load is not None
        else None
    )
    cache._load_cost_model_name = "model"
    cache._load_cost_world_size = 2
    cache.prefetch_loaded_storage_start_by_reqid = {handle: 999}
    cache.lmcache_connector = mock.Mock(chunk_size=256)
    cache.lmcache_connector.poll_lookup.return_value = total
    cache.lmcache_connector.parallel_all_reduce.side_effect = lambda tensor, _: (
        tensor.fill_(peer_local)
    )
    with mock.patch.object(
        UnifiedRadixCache,
        "match_prefix",
        return_value=MatchResult(
            device_prefix_len=local,
            last_device_node=None,
            last_host_node=None,
            best_match_node=None,
        ),
    ):
        assert cache.check_prefetch_progress(handle)

    cache.lmcache_connector.parallel_all_reduce.assert_called_once()
    if cost_load is not None:
        if total <= peer_local or total - peer_local < minimum:
            cache._load_cost_profile.should_load.assert_not_called()
        else:
            cache._load_cost_profile.should_load.assert_called_once_with(
                LoadCostKey("model", 2, 256, 1024, peer_local, total - peer_local)
            )
    assert cache.prefetch_loaded_tokens_by_reqid[handle] == expected_hit
    if release_end:
        cache.lmcache_connector.free_lookup_locks.assert_called_once_with(
            handle.rid, start=0, end=release_end
        )
    else:
        cache.lmcache_connector.free_lookup_locks.assert_not_called()
    cache.lmcache_connector.submit_load.assert_not_called()
    if expected_hit:
        assert flow.total_hit == total
        assert flow.local_hit_tokens == peer_local
        assert cache.prefetch_loaded_storage_start_by_reqid[handle] == peer_local
    else:
        assert handle.rid not in cache._external_flows
        assert handle not in cache.prefetch_loaded_storage_start_by_reqid
        assert cache.init_load_back(
            InitLoadBackParams(
                best_match_node=42,
                host_hit_length=total,
                req=SimpleNamespace(rid=handle.rid),
            )
        ) == (0, 42)


def test_lmcache_pending_lookup_keeps_its_flow(cache_class):
    cache = cache_class.__new__(cache_class)
    handle = CacheRequestHandle("request", 0)
    flow = SimpleNamespace(cancelled=False, total_hit=None, lookup=object())
    cache._external_flows = {handle.rid: flow}
    cache.lmcache_connector = mock.Mock()
    cache.lmcache_connector.poll_lookup.return_value = None

    assert not cache.check_prefetch_progress(handle)
    assert cache._external_flows[handle.rid] is flow
    cache.lmcache_connector.parallel_all_reduce.assert_not_called()
    cache.lmcache_connector.free_lookup_locks.assert_not_called()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
