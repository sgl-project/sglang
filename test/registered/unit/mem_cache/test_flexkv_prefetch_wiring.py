"""FlexKV wait-complete prefetch wiring (minimal scheduler surface)."""

import importlib.util
import json
import sys
from array import array
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.base_prefix_cache import (
    CacheRequestHandle,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.srt.mem_cache.storage.flexkv import _flexkv_factory
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _tracking_key(rid, attempt=0):
    return json.dumps([rid, attempt], separators=(",", ":"))


def _load_flexkv_module(module_filename: str, module_name: str):
    connector_name = "flexkv.integration.sglang.connector"
    connector_stub = ModuleType(connector_name)
    connector_stub.FlexKVConnector = object
    connector_stub.FlexKVHostReleaseShim = object

    module_path = (
        Path(__file__).resolve().parents[4]
        / "python/sglang/srt/mem_cache/storage/flexkv"
        / module_filename
    )
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    with patch.dict(sys.modules, {connector_name: connector_stub}):
        spec.loader.exec_module(module)
    return module


@pytest.fixture(
    params=[
        ("flexkv_radix_cache.py", "FlexKVRadixCache"),
        ("flexkv_hybrid_radix_cache.py", "FlexKVHybridRadixCache"),
    ]
)
def lookup_cache(request):
    module_filename, class_name = request.param
    module = _load_flexkv_module(module_filename, f"_flexkv_lookup_ut_{class_name}")
    cache_class = getattr(module, class_name)
    inner = RadixCache.create_simulated(page_size=4)
    cache = cache_class.__new__(cache_class)
    if class_name == "FlexKVRadixCache":
        cache.__dict__.update(inner.__dict__)
    else:
        cache._inner_cache = inner
    cache.flexkv_connector = MagicMock(supports_cache_namespace=True)
    cache._load_markers = {}
    cache._restore_leases = {}
    cache._aborted_restore_leases = {}
    return cache


@pytest.mark.parametrize("rematch", ["device_hit", "host_miss", "empty"])
def test_rematch_releases_the_previous_host_lookup(lookup_cache, rematch):
    cache = lookup_cache
    req = SimpleNamespace(
        rid="waiting", cache_request_handle=CacheRequestHandle("waiting", 0)
    )
    rid = _tracking_key("waiting")
    key = RadixKey(array("q", range(8)))
    cache.flexkv_connector.lookup_kv.return_value = (17, 8)
    first_match = cache.match_prefix(MatchPrefixParams(key=key, req=req))
    assert first_match.host_hit_length == 8
    cache.flexkv_connector.release_pending.assert_not_called()
    # The scheduler rematches waiting requests every round; only a later hit
    # makes the connector replace (and cancel) the previously held task.
    if rematch == "device_hit":
        inner = getattr(cache, "_inner_cache", cache)
        inner.insert(InsertParams(key=key, value=torch.arange(100, 108)))
    elif rematch == "host_miss":
        cache.flexkv_connector.lookup_kv.return_value = (-1, 0)
    else:
        key = RadixKey(array("q"))

    result = cache.match_prefix(MatchPrefixParams(key=key, req=req))

    cache.flexkv_connector.release_pending.assert_called_once_with(rid)
    assert result.host_hit_length == 0
    assert rid not in cache._load_markers


def test_completion_releases_only_its_unused_lookup(lookup_cache):
    cache = lookup_cache
    req = SimpleNamespace(
        rid="finished",
        cache_request_handle=CacheRequestHandle("finished", 0),
        origin_input_ids=array("q", range(8)),
        output_ids=array("q"),
        extra_key=None,
        cache_salt=None,
        kv=SimpleNamespace(kv_committed_len=8, swa_evicted_seqlen=0),
    )
    rid = _tracking_key("finished")
    other_rid = _tracking_key("waiting")
    cache.flexkv_connector.lookup_kv.return_value = (17, 8)
    cache.match_prefix(MatchPrefixParams(key=RadixKey(req.origin_input_ids), req=req))
    other_marker = object()
    cache._load_markers[other_rid] = other_marker
    # Seed Store ownership with each adapter's real tracking keys: the hybrid
    # adapter suffixes a generation, the ordinary one also stages copies.
    store_node = object()
    hybrid = hasattr(cache, "_inner_cache")
    if hybrid:
        dec_params = object()
        inflight = {f"{rid}:flexkv-store:0": (store_node, dec_params)}
        cache._inner_cache.dec_lock_ref = MagicMock()
        unlock = cache._inner_cache.dec_lock_ref
    else:
        inflight = {rid: store_node}
        staged = SimpleNamespace(node=store_node)
        cache._pending_store_copies = {rid: staged}
        cache.dec_lock_ref = MagicMock()
        unlock = cache.dec_lock_ref
    cache._inflight_store_nodes = dict(inflight)

    inner = getattr(cache, "_inner_cache", None)
    target = inner if inner is not None else RadixCache
    with patch.object(target, "cache_finished_req"):
        cache.cache_finished_req(req, is_insert=False, owned_kv_len=8)

    cache.flexkv_connector.release_pending.assert_called_once_with(rid)
    # Another request's lookup and in-flight Store ownership are untouched.
    assert cache._load_markers == {other_rid: other_marker}
    assert cache._inflight_store_nodes == inflight
    if not hybrid:
        assert cache._pending_store_copies == {rid: staged}
    unlock.assert_not_called()
    cache.flexkv_connector.cancel_prefetch.assert_not_called()


def test_scheduler_flexkv_prefetch_is_one_liner_to_tree_cache():
    sched = Scheduler.__new__(Scheduler)
    sched.enable_hicache_storage = False
    sched.enable_flexkv = True
    sched.tree_cache = MagicMock()

    req = SimpleNamespace(rid="x", cache_request_handle=CacheRequestHandle("x", 0))
    sched._prefetch_kvcache(req)
    sched.tree_cache.prefetch_request.assert_called_once_with(req)


def test_flexkv_radix_prefetch_request_page_aligns_and_launches():
    module = _load_flexkv_module(
        "flexkv_radix_cache.py", "_flexkv_radix_prefetch_request_ut"
    )
    cache = module.FlexKVRadixCache.__new__(module.FlexKVRadixCache)
    cache.page_size = 2
    cache.flexkv_connector = MagicMock()
    cache.flexkv_connector.prefetch_async = MagicMock(return_value=1)

    req = MagicMock()
    req.rid = "r1"
    req.cache_request_handle = CacheRequestHandle("r1", 0)
    req.extra_key = None
    req.cache_salt = None
    req.full_untruncated_fill_ids = [1, 2, 3, 4, 5]
    req._compute_max_prefix_len = MagicMock(return_value=4)
    req.init_next_round_input = MagicMock()

    cache.prefetch_request(req)

    req.init_next_round_input.assert_called_once_with(tree_cache=None, cow_mamba=False)
    args, _kwargs = cache.flexkv_connector.prefetch_async.call_args
    assert args[0] == _tracking_key("r1")
    assert list(args[1]) == [1, 2, 3, 4]


def test_scheduler_polls_explicit_flexkv_backend_without_legacy_flag():
    sched = Scheduler.__new__(Scheduler)
    sched.enable_hierarchical_cache = False
    sched.enable_unified_cache_external_linker = False
    sched.enable_hicache_storage = False
    sched.enable_flexkv = True
    sched.tree_cache = MagicMock()
    with patch(
        "sglang.srt.managers.scheduler.get_memory",
        return_value=SimpleNamespace(enable_flexkv=False),
    ):
        sched._process_hicache_events()
    sched.tree_cache.check_hicache_events.assert_called_once_with()


@pytest.mark.parametrize("entrypoint", ["scheduler", "shutdown"])
@pytest.mark.parametrize(
    "module_filename, class_name",
    [
        ("flexkv_radix_cache.py", "FlexKVRadixCache"),
        ("flexkv_hybrid_radix_cache.py", "FlexKVHybridRadixCache"),
    ],
)
def test_shutdown_releases_flexkv_and_inner_host_resources(
    module_filename, class_name, entrypoint
):
    module = _load_flexkv_module(module_filename, f"_flexkv_shutdown_ut_{class_name}")
    cache_class = getattr(module, class_name)
    cache = cache_class.__new__(cache_class)
    order = []
    cache.token_to_kv_pool_host = MagicMock()
    cache.token_to_kv_pool_host.destroy.side_effect = lambda: order.append("flexkv")
    hybrid = class_name == "FlexKVHybridRadixCache"
    if hybrid:
        cache._inner_cache = MagicMock()
        cache._inner_cache.release_host_resources.side_effect = lambda: order.append(
            "inner"
        )

    if entrypoint == "shutdown":
        cache.shutdown()
    else:
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.tree_cache = cache
        scheduler.hisparse_coordinator = None
        scheduler.decode_offload_manager = None
        with (
            patch("sglang.srt.managers.scheduler.destroy_global_experts_capturer"),
            patch("sglang.srt.managers.scheduler.destroy_global_indexer_capturer"),
            patch("sglang.srt.managers.scheduler.rank_consensus_checker"),
        ):
            scheduler.release_host_resources()

    cache.token_to_kv_pool_host.destroy.assert_called_once()
    assert order == (["flexkv", "inner"] if hybrid else ["flexkv"])


@pytest.mark.parametrize("enable_dp_attention, expected_rank", [(False, 2), (True, 3)])
@pytest.mark.parametrize("hybrid", [False, True])
def test_factory_uses_published_parallel_context(
    enable_dp_attention, expected_rank, hybrid
):
    parallel = SimpleNamespace(
        pp_group=SimpleNamespace(rank_in_group=1),
        attn_tp_group=object(),
        attn_cp_group=SimpleNamespace(rank_in_group=0),
        dp_rank=2,
        attn_dp_rank=3,
    )
    context = SimpleNamespace(
        server_args=SimpleNamespace(enable_dp_attention=enable_dp_attention),
        params=SimpleNamespace(is_eagle=False),
        model_config=object(),
        tp_rank=6,
        tp_size=8,
        tp_group=object(),
        # Stale worker state must not leak into the connector topology.
        tp_worker=SimpleNamespace(ps=SimpleNamespace(dp_rank=99, attn_dp_rank=99)),
        enable_hierarchical_cache=False,
        is_hybrid_ssm=False,
        is_hybrid_swa=hybrid,
    )
    radix_module = MagicMock()
    hybrid_module = MagicMock()
    unified_module = MagicMock()
    with (
        patch.dict(
            sys.modules,
            {
                "flexkv.integration.sglang.connector": MagicMock(),
                "sglang.srt.mem_cache.storage.flexkv.flexkv_radix_cache": radix_module,
                "sglang.srt.mem_cache.storage.flexkv.flexkv_hybrid_radix_cache": hybrid_module,
                "sglang.srt.mem_cache.unified_radix_cache": unified_module,
            },
        ),
        patch(
            "sglang.srt.mem_cache.storage.flexkv.get_serving",
            return_value=SimpleNamespace(enable_streaming_session=False),
        ),
        patch(
            "sglang.srt.mem_cache.storage.flexkv.get_memory",
            return_value=SimpleNamespace(flexkv_config_file=None),
        ),
        patch("sglang.srt.runtime_context.get_parallel", return_value=parallel),
    ):
        result = _flexkv_factory(context)

    constructor = (
        hybrid_module.FlexKVHybridRadixCache
        if hybrid
        else radix_module.FlexKVRadixCache
    )
    assert result is constructor.return_value
    assert constructor.call_args.kwargs["dp_rank"] == expected_rank
    assert constructor.call_args.kwargs["pp_rank"] == 1
    assert constructor.call_args.kwargs["attn_cp_rank"] == 0
    if hybrid:
        assert context.params.tree_components == (ComponentType.FULL, ComponentType.SWA)
        unified_module.UnifiedRadixCache.assert_called_once_with(context.params)
