"""Contract tests execute adapter methods without constructing model or KV pools."""

import ast
import json
import logging
from pathlib import Path
from types import MethodType
from types import SimpleNamespace as NS
from typing import Any, Optional
from unittest.mock import Mock

import pytest
import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    CacheRequestHandle,
    CacheRequestOutcome,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _tracking_key(rid, attempt=0):
    return json.dumps([rid, attempt], separators=(",", ":"))


ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "python/sglang/srt/mem_cache/storage/flexkv/flexkv_radix_cache.py"
HYBRID = SOURCE.with_name("flexkv_hybrid_radix_cache.py")
POLICY = ROOT / "python/sglang/srt/managers/schedule_policy.py"
SCHEDULER = ROOT / "python/sglang/srt/managers/scheduler.py"


def method(path, cls, name):
    tree = ast.parse(path.read_text())
    class_node = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls
    )
    body = next(
        n for n in class_node.body if isinstance(n, ast.FunctionDef) and n.name == name
    )
    body.decorator_list = []
    namespace = {
        "TreeNode": object,
        "CacheRequestHandle": CacheRequestHandle,
        "CacheRequestOutcome": CacheRequestOutcome,
        "_request_key": lambda handle: _tracking_key(handle.rid, handle.attempt_id),
        "Req": object,
        "Any": Any,
        "Optional": Optional,
        "AddReqResult": NS(OTHER="other"),
        "torch": torch,
        "InitLoadBackParams": object,
        "_RestoreLease": NS,
        "logger": logging.getLogger(__name__),
    }
    exec(
        compile(ast.Module(body=[body], type_ignores=[]), str(path), "exec"), namespace
    )
    return namespace[name]


@pytest.mark.parametrize("matched", [[], [10, 11, 12, 13]])
def test_full_chain_and_candidate_offset(matched):
    connector = Mock(_chunked_prefetch=True)
    cache = NS(flexkv_connector=connector, page_size=1)
    method(SOURCE, "FlexKVRadixCache", "prefetch_from_storage")(
        cache,
        CacheRequestHandle("r", 0),
        None,
        [20, 21],
        None,
        None,
        matched_prefix_tokens=matched,
    )
    connector.prefetch_async.assert_called_once_with(
        _tracking_key("r"),
        matched + [20, 21],
        sglang_req_id="r",
        candidate_start_token=len(matched),
    )


def test_scheduler_starts_flexkv_without_hicache_storage_backend():
    cache = Mock()
    scheduler = NS(enable_flexkv=True, enable_hicache_storage=False, tree_cache=cache)
    req = NS(rid="r", cache_request_handle=CacheRequestHandle("r", 0))
    method(SCHEDULER, "Scheduler", "_prefetch_kvcache")(scheduler, req)
    cache.prefetch_request.assert_called_once_with(req)


@pytest.mark.parametrize(
    "path,cls", [(SOURCE, "FlexKVRadixCache"), (HYBRID, "FlexKVHybridRadixCache")]
)
def test_queue_entry_does_not_lookup_remote_or_allocate_restore(path, cls):
    cache = NS(prefetch_from_storage=Mock())
    req = NS(
        rid="r",
        cache_request_handle=CacheRequestHandle("r", 0),
        full_untruncated_fill_ids=list(range(10)),
        extra_key=None,
        cache_salt=None,
        init_next_round_input=Mock(),
        _compute_max_prefix_len=lambda n: n - 1,
    )
    method(path, cls, "prefetch_request")(cache, req)
    req.init_next_round_input.assert_called_once_with(tree_cache=None, cow_mamba=False)
    cache.prefetch_from_storage.assert_called_once_with(
        req.cache_request_handle,
        None,
        list(range(9)),
        extra_key=None,
        cache_salt=None,
    )


def test_abort_releases_flexkv_without_hicache_enabled():
    cache = Mock()
    scheduler = NS(
        enable_flexkv=True,
        enable_hicache_storage=False,
        enable_hierarchical_cache=False,
        enable_unified_cache_external_linker=False,
        tree_cache=cache,
    )
    method(SCHEDULER, "Scheduler", "_release_aborted_request")(
        scheduler, NS(cache_request_handle=CacheRequestHandle("r", 0))
    )
    cache.finish.assert_called_once_with(
        CacheRequestHandle("r", 0), CacheRequestOutcome.ABORT
    )


def test_prefetch_stats_pass_through():
    connector = Mock(_chunked_prefetch=True)
    connector.pop_prefetch_loaded_span.return_value = (16, 32)
    cache = NS(flexkv_connector=connector, page_size=1)
    assert method(SOURCE, "FlexKVRadixCache", "pop_prefetch_loaded_span")(
        cache, CacheRequestHandle("r", 0)
    ) == (16, 32)


@pytest.mark.parametrize("key,salt", [("adapter", None), (None, "salt")])
def test_unsupported_namespace_does_not_use_unscoped_prefetch(key, salt):
    connector = Mock()
    cache = NS(flexkv_connector=connector, page_size=1)
    method(SOURCE, "FlexKVRadixCache", "prefetch_from_storage")(
        cache, CacheRequestHandle("r", 0), None, [1], extra_key=key, cache_salt=salt
    )
    connector.prefetch_async.assert_not_called()


def test_empty_request_does_not_launch_prefetch():
    cache = NS(prefetch_from_storage=Mock())
    req = NS(full_untruncated_fill_ids=[], init_next_round_input=Mock())
    method(SOURCE, "FlexKVRadixCache", "prefetch_request")(cache, req)
    cache.prefetch_from_storage.assert_not_called()


@pytest.mark.parametrize("matched", [[], [1, 2, 3, 4]])
def test_hybrid_preserves_hash_chain_and_candidate(matched):
    connector = Mock(_chunked_prefetch=True)
    cache = NS(flexkv_connector=connector, page_size=1)
    method(HYBRID, "FlexKVHybridRadixCache", "prefetch_from_storage")(
        cache, CacheRequestHandle("swa", 0), None, [5, 6], matched_prefix_tokens=matched
    )
    connector.prefetch_async.assert_called_once_with(
        _tracking_key("swa"),
        matched + [5, 6],
        sglang_req_id="swa",
        candidate_start_token=len(matched),
    )


@pytest.fixture
def hybrid_restore():
    req = NS(
        rid="restore",
        cache_request_handle=CacheRequestHandle("restore", 0),
        prefix_indices=torch.arange(256),
        last_node=object(),
        kv=NS(cache_protected_len=128),
    )
    slots = torch.arange(256, 768)
    connector = Mock(enable_layerwise=False)
    allocator = NS(free=Mock(), alloc_extend_swa_tail=Mock())
    cache = NS(
        flexkv_connector=connector,
        page_size=256,
        token_to_kv_pool_allocator=allocator,
        _restore_generation=7,
        _restore_leases={},
        _aborted_restore_leases={},
        _load_markers={_tracking_key(req.rid): NS(device_length=256)},
        _alloc_restore_slots=Mock(return_value=slots),
        supports_swa=lambda: True,
        _empty_indices=lambda: torch.empty(0, dtype=torch.int64),
    )
    cache._restore_lease_matches_req = method(
        HYBRID, "FlexKVHybridRadixCache", "_restore_lease_matches_req"
    )
    for name in (
        "_commit_restore",
        "_validate_restore_lease",
        "_forget_restore_lease",
        "init_load_back",
    ):
        setattr(
            cache,
            name,
            MethodType(method(HYBRID, "FlexKVHybridRadixCache", name), cache),
        )

    def retrieve(rid, actual_slots):
        lease = cache._restore_leases[rid]
        assert lease.device_indices is actual_slots
        assert req.pending_restore_slots is actual_slots
        return actual_slots.numel()

    connector.retrieve_kv.side_effect = retrieve
    return cache, req, slots


def test_hybrid_restore_owns_slots_before_h2d_and_preserves_tree_boundary(
    hybrid_restore,
):
    cache, req, slots = hybrid_restore
    restored, node = cache.init_load_back(NS(req=req, host_hit_length=512))
    assert restored is slots and node is req.last_node
    assert req.kv.cache_protected_len == 256
    assert req._flexkv_swa_evicted_seqlen == 512
    assert req.pending_restore_slots is slots
    cache._commit_restore(req)
    assert not cache._restore_leases and req.pending_restore_slots is None
    cache.token_to_kv_pool_allocator.free.assert_not_called()


def test_hybrid_failed_restore_frees_once(hybrid_restore):
    cache, req, slots = hybrid_restore
    cache.flexkv_connector.retrieve_kv.side_effect = None
    cache.flexkv_connector.retrieve_kv.return_value = 0
    restored, _ = cache.init_load_back(NS(req=req, host_hit_length=512))
    assert restored.numel() == 0 and not cache._restore_leases
    cache.token_to_kv_pool_allocator.free.assert_called_once_with(slots)
    assert req.pending_restore_slots is None


def test_hybrid_duplicate_restore_does_not_allocate_again(hybrid_restore):
    cache, req, _ = hybrid_restore
    cache.init_load_back(NS(req=req, host_hit_length=512))
    with pytest.raises(RuntimeError, match="before restore commit"):
        cache.init_load_back(NS(req=req, host_hit_length=512))
    assert cache._alloc_restore_slots.call_count == 1
    assert _tracking_key(req.rid) in cache._restore_leases
    cache.token_to_kv_pool_allocator.free.assert_not_called()


def test_hybrid_stale_generation_cannot_free_current_slots(hybrid_restore):
    cache, req, _ = hybrid_restore
    cache.init_load_back(NS(req=req, host_hit_length=512))
    req.pending_restore_generation -= 1
    with pytest.raises(RuntimeError, match="lease mismatch"):
        cache._commit_restore(req)
    assert _tracking_key(req.rid) in cache._restore_leases
    cache.token_to_kv_pool_allocator.free.assert_not_called()


@pytest.mark.parametrize(
    "flexkv,admitted,running,expected_full",
    [
        (True, False, False, False),
        (True, True, False, True),
        (True, False, True, True),
        (False, False, False, True),
    ],
)
def test_capacity_rejection_keeps_idle_flexkv_schedulable(
    flexkv, admitted, running, expected_full
):
    """Execute the scheduler's NO_TOKEN handling, including rejected-req cleanup."""
    tree = ast.parse(SCHEDULER.read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Scheduler"
    )
    method_node = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "_get_new_batch_prefill_raw"
    )
    stop_branch = next(
        n
        for n in ast.walk(method_node)
        if isinstance(n, ast.If)
        and ast.unparse(n.test) == "res != AddReqResult.CONTINUE"
    )
    module = ast.fix_missing_locations(
        ast.Module(
            body=[
                ast.For(
                    target=ast.Name(id="_", ctx=ast.Store()),
                    iter=ast.Tuple(elts=[ast.Constant(0)], ctx=ast.Load()),
                    body=[stop_branch],
                    orelse=[],
                )
            ],
            type_ignores=[],
        )
    )
    req = NS(kv=NS(holds_mamba=False))
    batch = NS(batch_is_full=False, is_empty=lambda: not running)
    namespace = dict(
        self=NS(
            enable_hierarchical_cache=False,
            enable_unified_cache_external_linker=False,
            enable_flexkv=flexkv,
            tree_cache=NS(has_uncommitted_restore=lambda r: False),
        ),
        get_memory=lambda: NS(enable_flexkv=flexkv),
        AddReqResult=NS(CONTINUE="continue", NO_TOKEN="no_token"),
        res="no_token",
        running_batch=batch,
        adder=NS(can_run_list=[req] if admitted else []),
        req=req,
        has_uncommitted_restore=None,
    )
    exec(compile(module, str(SCHEDULER), "exec"), namespace)
    assert batch.batch_is_full is expected_full
    if not admitted:
        assert req.kv.mamba_cow_src_index is None
        assert req.kv.mamba_needs_clear is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))


@pytest.mark.parametrize(
    "path,cls", [(SOURCE, "FlexKVRadixCache"), (HYBRID, "FlexKVHybridRadixCache")]
)
def test_stale_abort_does_not_cancel_retried_prefetch(path, cls):
    import threading

    old = CacheRequestHandle("same-rid", 0)
    current = CacheRequestHandle("same-rid", 1)
    old_key = _tracking_key(old.rid, old.attempt_id)
    new_key = _tracking_key(current.rid, current.attempt_id)
    connector = Mock(_chunked_prefetch=True)
    cache = NS(
        flexkv_connector=connector,
        page_size=1,
        _load_markers={new_key: object()},
        _restore_leases={},
        _aborted_restore_leases={},
        _node_lock=threading.Lock(),
        _pending_store_launches={},
        _pending_store_copies={},
        _inflight_store_nodes={},
        _release_restore_prefix=Mock(),
    )
    prefetch = method(path, cls, "prefetch_from_storage")
    prefetch(cache, old, None, [1, 2])
    prefetch(cache, current, None, [1, 2])
    assert [call.args[0] for call in connector.prefetch_async.call_args_list] == [
        old_key,
        new_key,
    ]
    method(path, cls, "release_aborted_request")(cache, old)
    connector.cancel_prefetch.assert_called_once_with(old_key)
    assert new_key in cache._load_markers
    method(path, cls, "check_prefetch_progress")(cache, current)
    connector.check_prefetch_progress.assert_called_once_with(new_key)
