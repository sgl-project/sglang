from types import SimpleNamespace

from sglang.srt.disaggregation.decode import DecodePreallocQueue
from sglang.srt.mem_cache.common import maybe_cache_unfinished_req
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool


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
