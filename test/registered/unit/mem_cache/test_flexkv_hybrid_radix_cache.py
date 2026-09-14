import importlib.util
import sys
import threading
from array import array
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest
import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    InitLoadBackParams,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _load_hybrid_cache_class():
    """Load the wrapper without requiring the optional FlexKV package."""
    module_name = "_flexkv_hybrid_radix_cache_under_test"
    connector_name = "flexkv.integration.sglang.connector"
    connector_stub = ModuleType(connector_name)
    connector_stub.FlexKVConnector = object
    connector_stub.FlexKVHostReleaseShim = object

    module_path = (
        Path(__file__).resolve().parents[4]
        / "python/sglang/srt/mem_cache/storage/flexkv/flexkv_hybrid_radix_cache.py"
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
    return module.FlexKVHybridRadixCache


FlexKVHybridRadixCache = _load_hybrid_cache_class()


def test_pool_accounting_delegates_to_inner_cache():
    inner = MagicMock()
    inner.evictable_size.return_value = 1280
    inner.full_evictable_size.return_value = 1024
    inner.swa_evictable_size.return_value = 256
    inner.protected_size.return_value = 640
    inner.full_protected_size.return_value = 512
    inner.swa_protected_size.return_value = 128

    cache = FlexKVHybridRadixCache.__new__(FlexKVHybridRadixCache)
    cache._inner_cache = inner

    assert cache.evictable_size() == 1280
    assert cache.full_evictable_size() == 1024
    assert cache.swa_evictable_size() == 256
    assert cache.protected_size() == 640
    assert cache.full_protected_size() == 512
    assert cache.swa_protected_size() == 128


def test_evict_does_not_poll_cross_rank_store_completion():
    inner = MagicMock()
    result = object()
    inner.evict.return_value = result
    connector = MagicMock()
    cache = FlexKVHybridRadixCache.__new__(FlexKVHybridRadixCache)
    cache._inner_cache = inner
    cache.flexkv_connector = connector
    params = object()

    assert cache.evict(params) is result

    inner.evict.assert_called_once_with(params)
    connector.check_completed_stores.assert_not_called()


def test_scheduler_hook_polls_cross_rank_store_completion():
    connector = MagicMock()
    connector.check_completed_stores.return_value = []
    cache = FlexKVHybridRadixCache.__new__(FlexKVHybridRadixCache)
    cache.flexkv_connector = connector
    cache._node_lock = threading.Lock()
    cache._inflight_store_nodes = {}

    cache.check_hicache_events()

    connector.check_completed_stores.assert_called_once_with()
    connector.drain_launched_loads.assert_called_once_with()


def test_restored_swa_tail_marks_older_prefix_as_evicted_before_cache_insert():
    inner = MagicMock()
    cache = FlexKVHybridRadixCache.__new__(FlexKVHybridRadixCache)
    cache._inner_cache = inner
    cache._restore_leases = {}
    cache._aborted_restore_leases = {}
    req = SimpleNamespace(
        rid="request",
        kv=SimpleNamespace(swa_evicted_seqlen=0),
        _flexkv_swa_evicted_seqlen=10240,
    )

    cache.cache_unfinished_req(req, chunked=True)

    assert req.kv.swa_evicted_seqlen == 10240
    assert not hasattr(req, "_flexkv_swa_evicted_seqlen")
    inner.cache_unfinished_req.assert_called_once_with(req, chunked=True)


def test_restore_lease_blocks_duplicate_lookup_until_cache_commit():
    node = object()
    inner_match = MatchResult(
        device_indices=torch.tensor([10, 11, 12, 13], dtype=torch.int64),
        last_device_node=node,
        last_host_node=node,
        best_match_node=node,
    )
    inner = MagicMock()
    inner.match_prefix.return_value = inner_match

    restored = torch.tensor([20, 21, 22, 23], dtype=torch.int64)
    allocator = MagicMock()
    connector = MagicMock()
    connector.enable_layerwise = False
    connector.lookup_kv.return_value = (7, 4)
    connector.retrieve_kv.return_value = 4

    cache = FlexKVHybridRadixCache.__new__(FlexKVHybridRadixCache)
    cache._inner_cache = inner
    cache.token_to_kv_pool_allocator = allocator
    cache.flexkv_connector = connector
    cache.disable = False
    cache.page_size = 4
    cache.device = torch.device("cpu")
    cache._load_markers = {}
    cache._restore_leases = {}
    cache._aborted_restore_leases = {}
    cache._restore_generation = 0
    cache._alloc_restore_slots = MagicMock(return_value=restored)
    cache.supports_swa = MagicMock(return_value=False)

    req = SimpleNamespace(
        rid="request",
        prefix_indices=inner_match.device_indices,
        last_node=node,
        kv=SimpleNamespace(cache_protected_len=4, swa_evicted_seqlen=0),
        pending_restore_generation=None,
        pending_restore_slots=None,
    )
    key = RadixKey(array("q", range(8)), extra_key=None)
    params = MatchPrefixParams(key=key, req=req)

    first_match = cache.match_prefix(params)
    loaded_indices, _ = cache.init_load_back(
        InitLoadBackParams(
            best_match_node=first_match.best_match_node,
            host_hit_length=first_match.host_hit_length,
            req=req,
        )
    )

    assert loaded_indices is restored
    assert cache.has_uncommitted_restore(req)
    assert req.pending_restore_generation == 0
    assert req.pending_restore_slots is restored
    lease = cache._restore_leases[req.rid]

    # Even a device-only match could overwrite the caller's restored prefix.
    with pytest.raises(RuntimeError, match="prefix rematch before restore commit"):
        cache.match_prefix(params)
    inner.match_prefix.assert_called_once_with(params)
    connector.lookup_kv.assert_called_once()
    assert req.rid not in cache._load_markers

    # init_load_back still fails loud: it allocates and starts a DMA, so a
    # second restore would leave two writers on one region.
    with pytest.raises(RuntimeError, match="load-back before restore commit"):
        cache.init_load_back(
            InitLoadBackParams(
                best_match_node=first_match.best_match_node,
                host_hit_length=first_match.host_hit_length,
                req=req,
            )
        )
    connector.release_pending.assert_not_called()
    cache._alloc_restore_slots.assert_called_once()
    connector.retrieve_kv.assert_called_once()
    # Neither duplicate entry point disturbed the active lease.
    assert cache._restore_leases[req.rid] is lease
    assert req.pending_restore_slots is restored

    cache.cache_unfinished_req(req, chunked=True)

    assert not cache.has_uncommitted_restore(req)
    assert req._flexkv_uncached_restore is False
    assert req.pending_restore_generation is None
    assert req.pending_restore_slots is None


def test_partial_synchronous_restore_is_freed_in_full():
    restored = torch.tensor([20, 21, 22, 23], dtype=torch.int64)
    allocator = MagicMock()
    connector = MagicMock()
    connector.enable_layerwise = False
    connector.retrieve_kv.return_value = 2

    cache = FlexKVHybridRadixCache.__new__(FlexKVHybridRadixCache)
    cache.token_to_kv_pool_allocator = allocator
    cache.flexkv_connector = connector
    cache.device = torch.device("cpu")
    cache._load_markers = {
        "request": SimpleNamespace(device_length=0),
    }
    cache._restore_leases = {}
    cache._aborted_restore_leases = {}
    cache._restore_generation = 0
    cache._alloc_restore_slots = MagicMock(return_value=restored)

    req = SimpleNamespace(rid="request", last_node=object())
    loaded_indices, _ = cache.init_load_back(
        InitLoadBackParams(best_match_node=req.last_node, host_hit_length=4, req=req)
    )

    assert loaded_indices.numel() == 0
    allocator.free.assert_called_once_with(restored)
    assert not cache.has_uncommitted_restore(req)
    assert req.pending_restore_generation is None
    assert req.pending_restore_slots is None


def test_restore_launch_exception_is_retained_until_safe_reset():
    restored = torch.tensor([20, 21, 22, 23], dtype=torch.int64)
    allocator = MagicMock()
    connector = MagicMock()
    connector.enable_layerwise = True
    connector.start_load_kv_layerwise.side_effect = RuntimeError("launch failed")

    cache = FlexKVHybridRadixCache.__new__(FlexKVHybridRadixCache)
    cache.token_to_kv_pool_allocator = allocator
    cache.flexkv_connector = connector
    cache.device = torch.device("cpu")
    cache._load_markers = {
        "request": SimpleNamespace(device_length=0),
    }
    cache._restore_leases = {}
    cache._aborted_restore_leases = {}
    cache._restore_generation = 0
    cache._alloc_restore_slots = MagicMock(return_value=restored)

    req = SimpleNamespace(rid="request", last_node=object())
    with pytest.raises(RuntimeError, match="launch failed"):
        cache.init_load_back(
            InitLoadBackParams(
                best_match_node=req.last_node,
                host_hit_length=4,
                req=req,
            )
        )

    allocator.free.assert_not_called()
    assert cache.has_uncommitted_restore(req)
    assert req.pending_restore_slots is restored


def test_finished_release_commits_restore_lease_after_inner_cache():
    restored = torch.tensor([20, 21, 22, 23], dtype=torch.int64)
    req = SimpleNamespace(
        rid="request",
        pending_restore_generation=3,
        pending_restore_slots=restored,
        _flexkv_uncached_restore=True,
        kv=SimpleNamespace(kv_committed_len=4),
        origin_input_ids=array("q", range(4)),
        output_ids=array("q"),
    )
    inner = MagicMock()
    cache = FlexKVHybridRadixCache.__new__(FlexKVHybridRadixCache)
    cache._inner_cache = inner
    cache._aborted_restore_leases = {}
    cache._restore_leases = {
        "request": SimpleNamespace(
            generation=3,
            rid=req.rid,
            req=req,
            device_indices=restored,
        )
    }

    cache.cache_finished_req(req, is_insert=False, kv_len_to_handle=4)

    inner.cache_finished_req.assert_called_once_with(
        req, is_insert=False, kv_len_to_handle=4
    )
    assert not cache.has_uncommitted_restore(req)
    assert req.pending_restore_generation is None
    assert req.pending_restore_slots is None
    assert req._flexkv_uncached_restore is False


def test_prefill_boundary_is_stored_with_an_independent_tracking_key():
    inner = MagicMock()
    inner.is_eagle = False
    inner.root_node = object()
    node = object()
    indices = torch.tensor([8, 9, 10, 11], dtype=torch.int64)
    inner.match_prefix.return_value = SimpleNamespace(
        last_device_node=node,
        device_indices=indices,
    )
    dec_params = object()
    inner.inc_lock_ref.return_value = SimpleNamespace(to_dec_params=lambda: dec_params)

    connector = MagicMock()
    connector.store_kv.side_effect = [17, 18]
    cache = FlexKVHybridRadixCache.__new__(FlexKVHybridRadixCache)
    cache._inner_cache = inner
    cache.flexkv_connector = connector
    cache.page_size = 4
    cache._node_lock = threading.Lock()
    cache._store_generation = 0
    cache._inflight_store_nodes = {}
    cache._restore_leases = {}
    cache._aborted_restore_leases = {}

    req = SimpleNamespace(
        rid="request",
        extra_key=None,
        kv=SimpleNamespace(swa_evicted_seqlen=0),
        get_fill_ids=lambda: array("q", [1, 2, 3, 4]),
    )

    with (
        patch("torch.cuda.current_stream", return_value=MagicMock()),
        patch("torch.cuda.stream", return_value=nullcontext()),
    ):
        cache.cache_unfinished_req(req)
        cache._store_prefix(req, [1, 2, 3, 4])

    inner.cache_unfinished_req.assert_called_once_with(req)
    first_store, second_store = connector.store_kv.call_args_list
    assert first_store.args[:2] == ("request:flexkv-store:0", [1, 2, 3, 4])
    assert first_store.args[2] is indices
    assert second_store.args[:2] == ("request:flexkv-store:1", [1, 2, 3, 4])
    assert second_store.args[2] is indices
    assert cache._inflight_store_nodes == {
        "request:flexkv-store:0": (node, dec_params),
        "request:flexkv-store:1": (node, dec_params),
    }


def test_reset_drains_flexkv_before_releasing_inner_slots():
    calls = MagicMock()
    connector = MagicMock()
    inner = MagicMock()
    calls.attach_mock(connector, "connector")
    calls.attach_mock(inner, "inner")

    cache = FlexKVHybridRadixCache.__new__(FlexKVHybridRadixCache)
    cache.flexkv_connector = connector
    cache._inner_cache = inner
    cache._load_markers = {}
    cache._restore_leases = {}
    cache._aborted_restore_leases = {}
    cache._restore_generation = 0
    cache._node_lock = threading.Lock()
    cache._inflight_store_nodes = {}

    cache.reset()

    assert calls.mock_calls[:2] == [
        call.connector.reset(),
        call.inner.reset(),
    ]


def test_reset_frees_uncommitted_restore_after_connector_drain():
    calls = MagicMock()
    connector = MagicMock()
    allocator = MagicMock()
    inner = MagicMock()
    calls.attach_mock(connector, "connector")
    calls.attach_mock(allocator, "allocator")
    calls.attach_mock(inner, "inner")
    restored = torch.tensor([20, 21, 22, 23], dtype=torch.int64)
    req = SimpleNamespace(
        rid="request",
        pending_restore_generation=0,
        pending_restore_slots=restored,
        _flexkv_uncached_restore=True,
    )

    cache = FlexKVHybridRadixCache.__new__(FlexKVHybridRadixCache)
    cache.flexkv_connector = connector
    cache.token_to_kv_pool_allocator = allocator
    cache._inner_cache = inner
    cache._load_markers = {}
    cache._aborted_restore_leases = {}
    cache._restore_leases = {
        "request": SimpleNamespace(
            generation=0,
            rid=req.rid,
            req=req,
            device_indices=restored,
        ),
    }
    cache._restore_generation = 1
    cache._node_lock = threading.Lock()
    cache._inflight_store_nodes = {}

    cache.reset()

    assert calls.mock_calls[:3] == [
        call.connector.reset(),
        call.allocator.free(restored),
        call.inner.reset(),
    ]
    assert req.pending_restore_generation is None
    assert req.pending_restore_slots is None
    assert req._flexkv_uncached_restore is False


def test_page_size_one_restore_requests_swa_for_the_full_hit():
    allocator = MagicMock()
    restored_slots = torch.arange(8, dtype=torch.int64)
    allocator.alloc.side_effect = [None, restored_slots]

    cache = FlexKVHybridRadixCache.__new__(FlexKVHybridRadixCache)
    cache.page_size = 1
    cache.token_to_kv_pool_allocator = allocator
    cache.supports_swa = MagicMock(return_value=True)
    req = SimpleNamespace()

    with patch(
        "sglang.srt.mem_cache.common.evict_from_tree_cache"
    ) as evict_from_tree_cache:
        result = cache._alloc_restore_slots(req, host_hit_length=8)

    assert result is restored_slots
    evict_from_tree_cache.assert_called_once_with(cache, 8, swa_num_tokens=8)


def _make_layerwise_restore(loaded=4):
    cache = FlexKVHybridRadixCache.__new__(FlexKVHybridRadixCache)
    cache._inner_cache = MagicMock()
    cache.token_to_kv_pool_allocator = MagicMock()
    cache.flexkv_connector = MagicMock()
    cache.flexkv_connector.enable_layerwise = True
    cache.flexkv_connector.start_load_kv_layerwise.return_value = (loaded, 0)
    cache.device = torch.device("cpu")
    cache.disable = False
    cache.page_size = 4
    cache.supports_swa = lambda: False
    cache._load_markers = {"request": SimpleNamespace(device_length=0)}
    cache._restore_leases = {}
    cache._aborted_restore_leases = {}
    cache._restore_generation = 0
    cache._node_lock = threading.Lock()
    cache._inflight_store_nodes = {}
    cache._pending_store_launches = {}
    cache._pending_store_copies = {}
    restored = torch.arange(20, 24, dtype=torch.int64)
    cache._alloc_restore_slots = MagicMock(return_value=restored)
    req = SimpleNamespace(
        rid="request",
        last_node=object(),
        prefix_indices=torch.empty(0, dtype=torch.int64),
        kv=SimpleNamespace(
            cache_protected_len=0, kv_committed_len=4, swa_evicted_seqlen=0
        ),
        origin_input_ids=array("q", range(4)),
        output_ids=array("q"),
    )
    params = InitLoadBackParams(
        best_match_node=req.last_node, host_hit_length=4, req=req
    )
    return cache, req, params, restored


@pytest.mark.parametrize("loaded", [2, 5])
def test_unexpected_layerwise_length_retains_every_slot_until_drained_reset(loaded):
    cache, req, params, restored = _make_layerwise_restore(loaded)
    with pytest.raises(RuntimeError, match="Unexpected layerwise restore length"):
        cache.init_load_back(params)
    cache.token_to_kv_pool_allocator.free.assert_not_called()
    assert req.pending_restore_slots is restored
    assert cache.has_uncommitted_restore(req)
    order = []
    cache.flexkv_connector.reset.side_effect = lambda: order.append("drain")
    cache.token_to_kv_pool_allocator.free.side_effect = lambda _s: order.append("free")
    cache._inner_cache.reset.side_effect = lambda: order.append("tree")
    cache.reset()
    assert order == ["drain", "free", "tree"]
    cache.token_to_kv_pool_allocator.free.assert_called_once_with(restored)
    assert not cache.has_uncommitted_restore(req)


def test_zero_layerwise_return_releases_prelaunch_allocation():
    cache, req, params, restored = _make_layerwise_restore(0)
    loaded, _ = cache.init_load_back(params)
    assert loaded.numel() == 0
    cache.token_to_kv_pool_allocator.free.assert_called_once_with(restored)
    assert not cache.has_uncommitted_restore(req)


@pytest.mark.parametrize("method", ["cache_finished_req", "cache_unfinished_req"])
@pytest.mark.parametrize("mismatch", ["identity", "generation", "slots"])
def test_hybrid_lease_mismatch_fails_before_inner_cache_mutation(method, mismatch):
    cache, req, params, restored = _make_layerwise_restore()
    cache.init_load_back(params)
    lease = cache._restore_leases[req.rid]
    if mismatch == "identity":
        req = SimpleNamespace(**vars(req))
    elif mismatch == "generation":
        req.pending_restore_generation += 1
    else:
        req.pending_restore_slots = req.pending_restore_slots.clone()
    req._flexkv_swa_evicted_seqlen = 8

    with pytest.raises(RuntimeError, match="restore lease mismatch"):
        if method == "cache_finished_req":
            cache.cache_finished_req(req, is_insert=False)
        else:
            cache.cache_unfinished_req(req, chunked=True)
    getattr(cache._inner_cache, method).assert_not_called()
    assert req.kv.swa_evicted_seqlen == 0
    cache.token_to_kv_pool_allocator.free.assert_not_called()
    assert cache.has_uncommitted_restore(req)
    assert cache._restore_leases[req.rid] is lease
    cache.reset()
    cache.token_to_kv_pool_allocator.free.assert_called_once_with(restored)
    assert cache._restore_leases == {}


def test_hybrid_abort_unblocks_rid_but_keeps_slots_until_cleanup():
    cache, req, params, restored = _make_layerwise_restore()
    cache.init_load_back(params)
    cache.release_aborted_request(req.rid)

    assert not cache.has_uncommitted_restore(req)
    assert req._flexkv_uncached_restore is True
    assert cache._aborted_restore_leases[req.pending_restore_generation].req is req
    assert req.pending_restore_slots is restored
    # Abort notification is not completion of the asynchronous H2D writer.
    cache.token_to_kv_pool_allocator.free.assert_not_called()

    # The real completion path still runs cleanly afterwards.
    cache.cache_finished_req(req, is_insert=False, kv_len_to_handle=4)
    cache._inner_cache.cache_finished_req.assert_called_once_with(
        req, is_insert=False, kv_len_to_handle=4
    )
    assert not cache.has_uncommitted_restore(req)
    assert cache._aborted_restore_leases == {}
    cache.token_to_kv_pool_allocator.free.assert_not_called()

    # And the aborted rid stays schedulable: a requeued restore is accepted.
    cache._load_markers[req.rid] = SimpleNamespace(device_length=0)
    cache.init_load_back(params)
    assert cache.has_uncommitted_restore(req)
    assert req.pending_restore_slots is restored


def test_hybrid_old_abort_cleanup_preserves_reused_rid_with_real_inner_cache():
    cache, old, params, old_slots = _make_layerwise_restore()
    allocator = cache.token_to_kv_pool_allocator
    inner = RadixCache.create_simulated(mock_allocator=allocator, page_size=4)
    cache._inner_cache = inner
    old.last_node = inner.root_node
    old.kv.req_pool_idx = 0
    old.extra_key = None
    old.cache_salt = None
    cache.init_load_back(params)
    cache.release_aborted_request(old.rid)
    new = SimpleNamespace(**vars(old))
    new.kv = SimpleNamespace(**vars(old.kv))
    new_slots = torch.arange(40, 44, dtype=torch.int64)
    cache._alloc_restore_slots.return_value = new_slots
    cache._load_markers[new.rid] = SimpleNamespace(device_length=0)
    cache.init_load_back(
        InitLoadBackParams(best_match_node=new.last_node, host_hit_length=4, req=new)
    )
    live = set(old_slots.tolist() + new_slots.tolist())

    def free(slots):
        for slot in slots.tolist():
            assert slot in live, f"double free: {slot}"
            live.remove(slot)

    allocator.free.side_effect = free
    allocator.free_segments.side_effect = lambda spans: [
        free(slots[start:]) for slots, start in spans
    ]
    inner.req_to_token_pool = SimpleNamespace(req_to_token=old_slots.unsqueeze(0))
    cache.cache_finished_req(old, is_insert=False, kv_len_to_handle=4)
    assert live == set(new_slots.tolist())
    assert cache._restore_leases[new.rid].req is new
    assert cache._aborted_restore_leases == {}
    inner.req_to_token_pool.req_to_token = new_slots.unsqueeze(0)
    cache.cache_finished_req(new, is_insert=False, kv_len_to_handle=4)
    cache.reset()
    assert not live


@pytest.mark.parametrize("stale", ["generation", "slots"])
def test_hybrid_reset_reclaims_orphaned_abort_despite_stale_request_fields(stale):
    cache, req, params, restored = _make_layerwise_restore()
    cache.init_load_back(params)
    cache.release_aborted_request(req.rid)
    if stale == "generation":
        req.pending_restore_generation += 100
    else:
        req.pending_restore_slots = req.pending_restore_slots.clone()
    cache.reset()
    cache.token_to_kv_pool_allocator.free.assert_called_once_with(restored)
    assert cache._aborted_restore_leases == {}


@pytest.mark.parametrize("failure", [None, "copy", "connector"])
def test_hybrid_reset_waits_for_staged_copy_and_connector_before_free(failure):
    cache, req, params, _ = _make_layerwise_restore()
    cache.init_load_back(params)
    event = MagicMock()
    pending = SimpleNamespace(ready_event=event, cpu_indices=object())
    cache._pending_store_copies["store"] = pending
    order = []

    def drain(stage):
        order.append(stage)
        if stage == failure:
            raise RuntimeError("transfer still active")

    event.synchronize.side_effect = lambda: drain("copy")
    cache.flexkv_connector.reset.side_effect = lambda: drain("connector")
    cache.token_to_kv_pool_allocator.free.side_effect = lambda _s: order.append("free")
    cache._inner_cache.reset.side_effect = lambda: order.append("tree")
    if failure is None:
        cache.reset()
        assert order == ["copy", "connector", "free", "tree"]
        assert not cache.has_uncommitted_restore(req)
        assert cache._pending_store_copies == {}
    else:
        with pytest.raises(RuntimeError, match="transfer still active"):
            cache.reset()
        assert order == (["copy"] if failure == "copy" else ["copy", "connector"])
        assert cache.has_uncommitted_restore(req)
        assert cache._pending_store_copies["store"] is pending
        cache.token_to_kv_pool_allocator.free.assert_not_called()
        cache._inner_cache.reset.assert_not_called()
