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
from sglang.srt.mem_cache.kv.plan import (
    TREE_TOKEN,
    TREE_WINDOW,
    TreeSlot,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.component_type import (
    ComponentType,
    tree_slot_ids,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci

# The window LMCache keeps (the tree's one window slot, as a one-window plan
# numbers it) and a further window, each by its sub-pool name and tree slot.
FIRST, SECOND = "window_4", "window_64"
SLOTS = (
    TreeSlot(component="FULL", family=TREE_TOKEN),
    TreeSlot(component="WINDOW_4", family=TREE_WINDOW, window=4),
)
FIRST_WINDOW = tree_slot_ids(SLOTS)["WINDOW_4"]
SECOND_WINDOW = ComponentType.of("WINDOW_64")

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
    cache._index_slots(SLOTS)
    cache.components = {FIRST_WINDOW: SimpleNamespace(window_sub_pool=FIRST)}
    cache.__dict__["_window_slot_sub_pools"] = {
        FIRST_WINDOW: FIRST,
        SECOND_WINDOW: SECOND,
    }
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
            window_evicted_seqlens={SECOND: 6},
        ),
        priority=0,
        last_node=None,
        lock=None,
    )
    if swa_enabled:
        req.kv.set_evicted_seqlen(FIRST, swa_cursor)
    return cache, req, flow


@pytest.mark.parametrize("swa_cursor", [0, 6])
def test_external_swa_load_preserves_independent_cursors(cache_class, swa_cursor):
    cache, req, flow = _cache_and_request(
        cache_class, swa_enabled=True, swa_cursor=swa_cursor
    )

    cache._publish_external_loaded_prefix(req, token_ids_len=12)

    assert flow.prefix_published
    assert req.kv.cache_protected_len == 2
    assert req.kv.window_evicted_seqlens == {FIRST: max(swa_cursor, 4), SECOND: 6}


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
    expected = {SECOND_WINDOW: 6}
    if swa_enabled:
        expected[FIRST_WINDOW] = 4
    assert insert_params.component_evicted_seqlens == expected
    req.kv.set_evicted_seqlen(SECOND, 99)
    assert insert_params.get_evicted_seqlen(SECOND_WINDOW) == 6
    assert req.kv.cache_protected_len == 8
    assert req.lock.receipt.node_id == 42
    assert flow.prefix_published
    assert flow.mamba_value is None


@pytest.mark.parametrize("swa_enabled", [False, True])
def test_external_slots_on_a_shared_memory_reclaim_the_load_demand(
    cache_class, swa_enabled
):
    """On sub-pools sharing one memory, an external load reclaims its whole demand
    (token rows and the window tail) and reads no availability estimate."""
    from sglang.srt.mem_cache.kv.alloc.demand import KVDemand

    cache, _, _ = _cache_and_request(cache_class, swa_enabled=swa_enabled)
    cache.supports_prefix_sharing = lambda: True
    cache.evict_for_alloc = mock.Mock()
    allocator = mock.Mock(pools_share_memory=True, hands_out_virtual_ids=True)
    allocator.prealloc_shortfall.return_value = None
    estimate = mock.Mock(side_effect=AssertionError("read an estimate"))
    allocator.available_size = estimate
    allocator.full_attn_allocator.available_size = estimate
    allocator.swa_attn_allocator.available_size = estimate
    taken = torch.arange(8)
    allocator.alloc.return_value = taken
    allocator.full_attn_allocator.alloc.return_value = taken
    cache.token_to_kv_pool_allocator = allocator

    with (
        mock.patch.object(cache_class, "sliding_window_size", 4),
        mock.patch.object(cache_class, "page_size", 1),
    ):
        assert cache._allocate_external_slots(8) is taken

    window = {FIRST: 4} if swa_enabled else {}
    cache.evict_for_alloc.assert_called_once_with(
        KVDemand(full_tokens=8, window_tokens=window)
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))


@pytest.mark.parametrize(
    "disable_overlap,pp_size,current_is_consumer",
    [(True, 1, True), (False, 1, False), (True, 2, False)],
)
def test_lmcache_load_fences_actual_execution_stream(
    cache_class, disable_overlap, pp_size, current_is_consumer
):
    cache = cache_class.__new__(cache_class)
    cache._forward_stream = object()
    current = object()
    device_module = SimpleNamespace(current_stream=lambda: current)
    with mock.patch.dict(
        cache_class._load_stream.__globals__,
        {
            "get_schedule": lambda: SimpleNamespace(
                disable_overlap_schedule=disable_overlap
            ),
            "get_parallel": lambda: SimpleNamespace(pp_size=pp_size),
            "torch": SimpleNamespace(get_device_module=lambda device: device_module),
        },
    ):
        consumer = cache._load_stream(torch.device("cuda"))
    assert consumer is (current if current_is_consumer else cache._forward_stream)


@pytest.mark.parametrize("resident_len", [6, 8])
def test_attention_writeback_uses_full_kv_beyond_recurrent_checkpoint(
    cache_class, resident_len
):
    """A joint match may stop before a dirty block whose attention KV is resident."""
    cache = cache_class.__new__(cache_class)
    cache._mamba_component = SimpleNamespace(component_type=ComponentType.MAMBA)
    cache.tree_core = mock.Mock(is_eagle=False, page_size=1)
    # The key ends inside a longer node. Pin its path, but copy only the hole.
    cache.tree_core.match_full_device_prefix.return_value = (resident_len, 42, 12)
    cache.lmcache_connector = mock.Mock(chunk_size=4)
    operation = object()
    cache.lmcache_connector.submit_store.return_value = operation
    cache._pending_stores = []
    cache._pending_store_counts = {}
    cache.inc_lock_ref = mock.Mock(return_value=IncLockRefResult(node_id=42))
    cache.dec_lock_ref = mock.Mock()
    cache.path_device_indices = mock.Mock(return_value=torch.arange(12))
    req = SimpleNamespace(rid="suffix", extra_key=None, cache_salt="salt")
    joint = MatchResult(4, 41, 41, 41, full_kv_hit_length=8)
    with mock.patch.object(UnifiedRadixCache, "match_prefix", return_value=joint):
        cache._submit_store(req, list(range(12)), writeback_range=(4, 8))

    if resident_len < 8:
        cache.lmcache_connector.submit_store.assert_not_called()
        cache.inc_lock_ref.assert_not_called()
        assert not cache._pending_stores
        return
    cache.lmcache_connector.submit_store.assert_called_once()
    args, kwargs = cache.lmcache_connector.submit_store.call_args
    assert args[:2] == ("suffix", list(range(8)))
    assert kwargs["writeback_range"] == (4, 8)
    assert kwargs["device_indices_start"] == 4
    indices = cache.lmcache_connector.device_indices_by_group.call_args
    assert torch.equal(indices.args[0], torch.arange(4, 8))
    assert indices.kwargs["mamba_value"] is None
    cache.inc_lock_ref.assert_called_once_with(
        42, skip_lock_components=(ComponentType.MAMBA,)
    )
    cache.dec_lock_ref.assert_not_called()
    assert len(cache._pending_stores) == 1
    assert cache._pending_store_counts == {"suffix": 1}


@pytest.mark.parametrize("raises", [False, True])
def test_attention_writeback_failed_admission_releases_transfer_lock(
    cache_class, raises
):
    cache = cache_class.__new__(cache_class)
    cache._mamba_component = SimpleNamespace(component_type=ComponentType.MAMBA)
    cache.tree_core = mock.Mock(is_eagle=False, page_size=1)
    cache.tree_core.match_full_device_prefix.return_value = (8, 42, 8)
    cache.lmcache_connector = mock.Mock(chunk_size=4)
    cache.lmcache_connector.submit_store.return_value = None
    if raises:
        cache.lmcache_connector.submit_store.side_effect = RuntimeError("admission")
    cache._pending_stores = []
    cache._pending_store_counts = {}
    cache.inc_lock_ref = mock.Mock(return_value=IncLockRefResult(node_id=42))
    cache.dec_lock_ref = mock.Mock()
    cache.path_device_indices = mock.Mock(return_value=torch.arange(8))
    req = SimpleNamespace(rid="suffix", extra_key=None, cache_salt="salt")
    joint = MatchResult(4, 41, 41, 41, full_kv_hit_length=8)
    with mock.patch.object(UnifiedRadixCache, "match_prefix", return_value=joint):
        cache._submit_store(req, list(range(8)), writeback_range=(4, 8))
    cache.lmcache_connector.submit_store.assert_called_once()
    cache.dec_lock_ref.assert_called_once()
    assert cache.dec_lock_ref.call_args.args[0] == 42
    assert not cache._pending_stores
    assert not cache._pending_store_counts


@pytest.mark.parametrize("stored_prefix", [0, 4])
def test_ordinary_store_still_requires_joint_recurrent_checkpoint(
    cache_class, stored_prefix
):
    cache = cache_class.__new__(cache_class)
    cache._mamba_component = SimpleNamespace(component_type=ComponentType.MAMBA)
    cache.tree_core = mock.Mock(is_eagle=False, page_size=1)
    state = torch.tensor([7])
    cache.tree_core.get_component_device_value.return_value = state
    cache.lmcache_connector = mock.Mock(chunk_size=4)
    cache.lmcache_connector.get_store_start.return_value = stored_prefix
    cache.lmcache_connector.submit_store.return_value = object()
    cache._pending_stores = []
    cache._pending_store_counts = {}
    cache.inc_lock_ref = mock.Mock(return_value=IncLockRefResult(node_id=41))
    cache.path_device_indices = mock.Mock(return_value=torch.arange(8))
    req = SimpleNamespace(rid="ordinary", extra_key=None, cache_salt="salt")
    joint = MatchResult(4, 41, 41, 41, full_kv_hit_length=8)
    with mock.patch.object(UnifiedRadixCache, "match_prefix", return_value=joint):
        cache._submit_store(req, list(range(8)))
    cache.tree_core.match_full_device_prefix.assert_not_called()
    if stored_prefix:
        cache.lmcache_connector.submit_store.assert_not_called()
        cache.inc_lock_ref.assert_not_called()
        return
    args, kwargs = cache.lmcache_connector.submit_store.call_args
    assert args[:2] == ("ordinary", list(range(4)))
    assert kwargs["writeback_range"] is None
    indices = cache.lmcache_connector.device_indices_by_group.call_args
    assert torch.equal(indices.args[0], torch.arange(4))
    assert indices.kwargs["mamba_value"] is state
    cache.inc_lock_ref.assert_called_once_with(41, skip_lock_components=())
