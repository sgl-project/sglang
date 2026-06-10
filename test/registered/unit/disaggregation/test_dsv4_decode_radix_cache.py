from types import SimpleNamespace

import torch

from sglang.srt.disaggregation.decode import DecodePreallocQueue
from sglang.srt.environ import envs
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchResult
from sglang.srt.mem_cache.common import maybe_cache_unfinished_req
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.mem_cache.kv_cache_builder import is_supported_dsv4_decode_radix_mtp
from sglang.srt.mem_cache.unified_cache_components.swa_component import SWAComponent
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


def _make_queue(*, page_size: int, enable_decode_radix: bool = True):
    queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
    queue.scheduler = SimpleNamespace(
        server_args=SimpleNamespace(
            disaggregation_decode_enable_radix_cache=enable_decode_radix
        ),
        running_batch=SimpleNamespace(reqs=[]),
    )
    queue.token_to_kv_pool_allocator = SimpleNamespace(page_size=page_size)
    queue.token_to_kv_pool = DeepSeekV4TokenToKVPool.__new__(DeepSeekV4TokenToKVPool)
    queue.token_to_kv_pool.compression_ratios = [0, 4, 128]
    queue.transfer_queue = SimpleNamespace(queue=[])
    return queue


def test_dsv4_decode_radix_prefix_is_c128_safe():
    queue = _make_queue(page_size=64)

    assert queue._dsv4_safe_prefix_len(127) == 0
    assert queue._dsv4_safe_prefix_len(128) == 128
    assert queue._dsv4_safe_prefix_len(383) == 256


def test_dsv4_decode_radix_prefix_respects_page_alignment():
    queue = _make_queue(page_size=256)

    assert queue._dsv4_safe_prefix_len(255) == 0
    assert queue._dsv4_safe_prefix_len(256) == 256
    assert queue._dsv4_safe_prefix_len(511) == 256


def test_dsv4_decode_radix_prefix_unchanged_when_disabled():
    queue = _make_queue(page_size=64, enable_decode_radix=False)

    assert queue._dsv4_safe_prefix_len(383) == 383


def test_dsv4_singleflight_waits_for_large_inflight_shared_prefix():
    queue = _make_queue(page_size=256)
    shared_prefix = list(range(5000))
    inflight_req = SimpleNamespace(
        origin_input_ids=shared_prefix + [1],
        dsv4_decode_radix_cache_prompt_once=True,
    )
    waiting_req = SimpleNamespace(origin_input_ids=shared_prefix + [2])
    queue.transfer_queue.queue = [SimpleNamespace(req=inflight_req)]

    assert queue._should_wait_for_dsv4_inflight_prompt(
        waiting_req,
        prefix_len=0,
        preallocated_reqs=[],
    )
    assert not queue._should_wait_for_dsv4_inflight_prompt(
        waiting_req,
        prefix_len=4096,
        preallocated_reqs=[],
    )


def test_dsv4_singleflight_ignores_short_shared_prefix():
    queue = _make_queue(page_size=256)
    shared_prefix = list(range(1024))
    inflight_req = SimpleNamespace(
        origin_input_ids=shared_prefix + [1],
        dsv4_decode_radix_cache_prompt_once=True,
    )
    waiting_req = SimpleNamespace(origin_input_ids=shared_prefix + [2])
    queue.transfer_queue.queue = [SimpleNamespace(req=inflight_req)]

    assert not queue._should_wait_for_dsv4_inflight_prompt(
        waiting_req,
        prefix_len=0,
        preallocated_reqs=[],
    )


def test_allow_radix_cache_insert_once_bypasses_skip_once():
    req = SimpleNamespace(
        skip_radix_cache_insert=True,
        allow_radix_cache_insert_once=True,
    )
    tree_cache = SimpleNamespace(num_cache_unfinished_req=0)

    def cache_unfinished_req(_req, **_kwargs):
        tree_cache.num_cache_unfinished_req += 1

    tree_cache.cache_unfinished_req = cache_unfinished_req

    maybe_cache_unfinished_req(req, tree_cache)
    maybe_cache_unfinished_req(req, tree_cache)

    assert tree_cache.num_cache_unfinished_req == 1
    assert req.allow_radix_cache_insert_once is False


def test_dsv4_decode_radix_mtp_guard_allows_eagle_topk1_online_c128():
    server_args = SimpleNamespace(speculative_eagle_topk=1)

    with envs.SGLANG_OPT_USE_ONLINE_COMPRESS.override(True):
        with envs.SGLANG_EXPERIMENTAL_ONLINE_C128_MTP.override(True):
            assert is_supported_dsv4_decode_radix_mtp(
                spec_algorithm=SpeculativeAlgorithm.EAGLE,
                server_args=server_args,
            )


def test_dsv4_decode_radix_mtp_guard_rejects_non_minimal_paths():
    server_args = SimpleNamespace(speculative_eagle_topk=1)

    with envs.SGLANG_OPT_USE_ONLINE_COMPRESS.override(True):
        with envs.SGLANG_EXPERIMENTAL_ONLINE_C128_MTP.override(True):
            assert not is_supported_dsv4_decode_radix_mtp(
                spec_algorithm=SpeculativeAlgorithm.EAGLE3,
                server_args=server_args,
            )
            assert not is_supported_dsv4_decode_radix_mtp(
                spec_algorithm=SpeculativeAlgorithm.FROZEN_KV_MTP,
                server_args=server_args,
            )

    server_args.speculative_eagle_topk = 2
    with envs.SGLANG_OPT_USE_ONLINE_COMPRESS.override(True):
        with envs.SGLANG_EXPERIMENTAL_ONLINE_C128_MTP.override(True):
            assert not is_supported_dsv4_decode_radix_mtp(
                spec_algorithm=SpeculativeAlgorithm.EAGLE,
                server_args=server_args,
            )


def test_dsv4_prompt_insert_uses_prompt_snapshot_and_restores_fill_ids():
    req = SimpleNamespace(
        dsv4_decode_radix_cache_prompt_once=True,
        dsv4_decode_radix_cache_prompt_len=3,
        skip_radix_cache_insert=True,
        allow_radix_cache_insert_once=False,
        fill_ids=[1, 2, 3, 4, 5],
        extra_key=None,
        cache_protected_len=0,
        swa_evicted_seqlen=4,
    )
    tree_cache = SimpleNamespace(
        inserted_fill_ids=[],
        inserted_swa_evicted_seqlens=[],
        inserted_force_leaf_creation=[],
        page_size=2,
        sliding_window_size=2,
        is_eagle=True,
    )

    def cache_unfinished_req(inserted_req, **_kwargs):
        tree_cache.inserted_fill_ids.append(list(inserted_req.fill_ids))
        tree_cache.inserted_swa_evicted_seqlens.append(
            inserted_req.swa_evicted_seqlen
        )
        tree_cache.inserted_force_leaf_creation.append(
            inserted_req.force_radix_leaf_creation
        )
        inserted_req.cache_protected_len = 2

    tree_cache.cache_unfinished_req = cache_unfinished_req
    processor = SimpleNamespace(tree_cache=tree_cache)

    SchedulerBatchResultProcessor._maybe_insert_dsv4_decode_radix_prompt(processor, req)
    SchedulerBatchResultProcessor._maybe_insert_dsv4_decode_radix_prompt(processor, req)

    assert tree_cache.inserted_fill_ids == [[1, 2, 3]]
    assert tree_cache.inserted_swa_evicted_seqlens == [2]
    assert tree_cache.inserted_force_leaf_creation == [True]
    assert req.fill_ids == [1, 2, 3, 4, 5]
    assert req.cache_protected_len == 2
    assert req.swa_evicted_seqlen == 4
    assert req.force_radix_leaf_creation is False
    assert req.allow_radix_cache_insert_once is False
    assert req.dsv4_decode_radix_cache_prompt_once is False


def test_swa_component_force_leaf_creation_allows_full_only_leaf():
    component = SWAComponent.__new__(SWAComponent)

    assert SWAComponent.should_skip_leaf_creation(
        component,
        total_prefix_len=0,
        key_len=256,
        params=InsertParams(swa_evicted_seqlen=256),
    )
    assert not SWAComponent.should_skip_leaf_creation(
        component,
        total_prefix_len=0,
        key_len=256,
        params=InsertParams(swa_evicted_seqlen=256, force_leaf_creation=True),
    )


def test_dsv4_decode_radix_match_uses_full_match_for_full_only_leaf():
    queue = _make_queue(page_size=2)
    node = object()
    tree_cache = SimpleNamespace(captured_params=None)

    def supports_mamba():
        return False

    def match_prefix(params):
        tree_cache.captured_params = params
        return MatchResult(
            device_indices=torch.tensor([1, 2], dtype=torch.int64),
            last_device_node=node,
            last_host_node=node,
            best_match_node=node,
        )

    def inc_lock_ref(_node):
        return SimpleNamespace(swa_uuid_for_lock=None)

    tree_cache.supports_mamba = supports_mamba
    tree_cache.match_prefix = match_prefix
    tree_cache.inc_lock_ref = inc_lock_ref
    queue.tree_cache = tree_cache

    req = SimpleNamespace(
        origin_input_ids=[1, 2, 3],
        extra_key=None,
        prefix_indices=None,
        last_node=None,
        last_host_node=None,
        best_match_node=None,
        host_hit_length=0,
        swa_uuid_for_lock=None,
    )

    prefix_indices, prefix_len = DecodePreallocQueue._match_prefix_and_lock(queue, req)

    assert prefix_indices.tolist() == [1, 2]
    assert prefix_len == 2
    assert tree_cache.captured_params.return_full_match is True
    assert req.last_node is node
