from types import SimpleNamespace

from collections import namedtuple
from types import SimpleNamespace

from sglang.srt.disaggregation.decode import DecodePreallocQueue
from sglang.srt.environ import envs
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.mem_cache.common import maybe_cache_unfinished_req
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.mem_cache.kv_cache_builder import is_supported_dsv4_decode_radix_mtp
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

Range = namedtuple("Range", ["start", "end"])


def _make_queue(*, page_size: int, enable_decode_radix: bool = True):
    queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
    queue.scheduler = SimpleNamespace(
        server_args=SimpleNamespace(
            disaggregation_decode_enable_radix_cache=enable_decode_radix
        )
    )
    queue.token_to_kv_pool_allocator = SimpleNamespace(page_size=page_size)
    queue.token_to_kv_pool = DeepSeekV4TokenToKVPool.__new__(DeepSeekV4TokenToKVPool)
    queue.token_to_kv_pool.compression_ratios = [0, 4, 128]
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
        extend_range=Range(0, 5),
        full_untruncated_fill_ids=[1, 2, 3, 4, 5],
    )
    req.get_fill_ids = lambda: [1, 2, 3, 4, 5][: req.extend_range.end]
    tree_cache = SimpleNamespace(inserted_fill_ids=[])

    def cache_unfinished_req(inserted_req, **_kwargs):
        tree_cache.inserted_fill_ids.append(list(inserted_req.get_fill_ids()))

    tree_cache.cache_unfinished_req = cache_unfinished_req
    processor = SimpleNamespace(tree_cache=tree_cache)

    SchedulerBatchResultProcessor._maybe_insert_dsv4_decode_radix_prompt(
        processor, req, is_prebuilt_batch=True
    )
    SchedulerBatchResultProcessor._maybe_insert_dsv4_decode_radix_prompt(
        processor, req, is_prebuilt_batch=True
    )

    assert tree_cache.inserted_fill_ids == [[1, 2, 3]]
    assert req.extend_range.end == 5
    assert req.allow_radix_cache_insert_once is False
    assert req.dsv4_decode_radix_cache_prompt_once is False
