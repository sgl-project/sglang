"""The last token of a bigram page and draft format are cache identity."""

from array import array
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.storage.flexkv.flexkv_cache_lifecycle import (
    FlexKVCacheLifecycleMixin,
    _draft_pool_kwargs,
    _flexkv_token_ids,
    _namespace_kwargs,
)
from sglang.srt.speculative import base_spec_worker as spec
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def test_different_boundary_token_cannot_reuse_the_last_bigram_page():
    first = RadixKey(array("q", [10, 20, 30, 40, 50]), is_bigram=True)
    second = RadixKey(array("q", [10, 20, 30, 40, 51]), is_bigram=True)
    assert first.raw_token_ids()[:4] == second.raw_token_ids()[:4]
    encoded = _flexkv_token_ids(first.page_aligned(4))
    other = _flexkv_token_ids(second.page_aligned(4))
    assert len(encoded) == len(other) == 4
    assert encoded[:3] == other[:3]
    assert encoded[-1] != other[-1]
    assert encoded[-1] == (40 << 32) | 50


def test_bigram_page_slices_share_one_boundary_without_losing_a_slot():
    key = RadixKey(array("q", range(9)), is_bigram=True)
    assert _flexkv_token_ids(key[:4]) + _flexkv_token_ids(key[4:]) == _flexkv_token_ids(
        key
    )
    assert len(_flexkv_token_ids(key[:4])) == 4


def test_draft_format_namespaces_even_an_unscoped_request():
    connector = NS(
        supports_cache_namespace=True, cache_format_namespace="draft-layout-a"
    )
    assert _namespace_kwargs(connector, None, None) == {"namespace": ["draft-layout-a"]}
    first = _namespace_kwargs(connector, None, "tenant")
    connector.cache_format_namespace = "draft-layout-b"
    second = _namespace_kwargs(connector, None, "tenant")
    assert first["namespace"][0] != second["namespace"][0]
    assert first["namespace"][1:] == second["namespace"][1:]


def test_eagle_prefetch_uses_one_hash_value_per_slot_with_raw_boundary():
    connector = NS(
        supports_cache_namespace=True, _chunked_prefetch=True, prefetch_async=Mock()
    )
    cache = NS(flexkv_connector=connector, page_size=4, is_eagle=True)
    handle = CacheRequestHandle("draft", 0)
    FlexKVCacheLifecycleMixin.prefetch_from_storage(
        cache,
        handle,
        token_ids=[5, 6, 7, 8],
        matched_prefix_tokens=[0, 1, 2, 3, 4],
        cache_salt="tenant",
    )
    call = connector.prefetch_async.call_args
    assert list(call.args[1]) == list(
        _flexkv_token_ids(RadixKey(array("q", range(9)), is_bigram=True))
    )
    assert call.kwargs["candidate_start_token"] == 4
    assert (
        call.kwargs["namespace"]
        == _namespace_kwargs(connector, None, "tenant")["namespace"]
    )


@pytest.mark.parametrize("length,expected", [(1024, 896), (96, 0), (128, 0)])
def test_prefetch_stops_at_the_same_window_replay_boundary_as_match(length, expected):
    launch = Mock()
    cache = NS(
        prefetch_from_storage=launch, window_reprefill_tail_tokens=lambda req: 128
    )
    req = NS(
        init_next_round_input=Mock(),
        full_untruncated_fill_ids=array("q", range(length)),
        _compute_max_prefix_len=lambda length: length - 1,
        cache_request_handle=CacheRequestHandle("bounded", 0),
        extra_key=None,
        cache_salt=None,
    )
    FlexKVCacheLifecycleMixin.prefetch_request(cache, req)
    assert len(launch.call_args.args[2]) == expected


@pytest.mark.parametrize("explicit_backend", [False, True])
@pytest.mark.parametrize("packed", [False, True])
def test_flexkv_activation_passes_the_actual_draft_pool(
    monkeypatch, explicit_backend, packed
):
    monkeypatch.setattr(
        spec,
        "get_memory",
        lambda: NS(
            enable_hierarchical_cache=False,
            enable_unified_cache_external_linker=False,
            enable_flexkv=not explicit_backend,
            radix_cache_backend="flexkv" if explicit_backend else None,
        ),
    )
    monkeypatch.setattr(
        spec,
        "get_disagg",
        lambda: NS(disaggregation_decode_retraction_backup="cpu_tensor"),
    )
    pool = NS(layer_router=NS(part_view=lambda part: None))
    target = NS(spec_algorithm=SpeculativeAlgorithm.EAGLE, token_to_kv_pool=pool)
    runner = NS(
        token_to_kv_pool=pool,
        runner_ids=NS(is_fused_draft=False),
        model_config=NS(num_nextn_predict_layers=int(packed)),
    )
    worker = NS(
        target_worker=NS(model_runner=target),
        _draft_model_runners=lambda: (runner,) if packed else (runner, runner),
    )
    plan = spec.BaseSpecWorker._build_hicache_draft_plan(worker)
    assert plan.device_pools == ((pool,) if packed else (pool, pool))
    assert target.mtp_draft_device_pools == ((pool,) if packed else ())
    assert _draft_pool_kwargs(NS(mtp_draft_device_pools=(pool,))) == {
        "draft_kv_pools": (pool,)
    }
    runner.runner_ids.is_fused_draft = True
    with pytest.raises(NotImplementedError, match="fused draft"):
        spec.BaseSpecWorker._build_hicache_draft_plan(worker)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
