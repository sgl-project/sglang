import unittest

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    EvictParams,
    EvictResult,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.common import available_and_evictable_str
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool, SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.swa_radix_cache import SWARadixCache
from sglang.srt.utils import get_device
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=8, suite="stage-b-test-1-gpu-large")
register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd")


class _DummyReq:
    def __init__(self):
        self._kv_committed_len = 0
        self.swa_prefix_lock_released = False

    def pop_committed_kv_cache(self):
        return self._kv_committed_len


def _build_swa_tree(
    is_eagle: bool,
    page_size: int = 1,
    req_size: int = 8,
    max_context_len: int = 64,
    kv_size: int = 64,
    kv_size_swa: int = 32,
    sliding_window_size: int = 4,
):
    head_num = 8
    head_dim = 128
    num_layers = 24
    global_interval = 4
    dtype = torch.bfloat16
    device = get_device()
    full_attention_layer_ids = [i for i in range(0, num_layers, global_interval)]
    full_attention_layer_ids_set = set(full_attention_layer_ids)
    swa_attention_layer_ids = [
        i for i in range(num_layers) if i not in full_attention_layer_ids_set
    ]

    req_to_token_pool = ReqToTokenPool(
        size=req_size,
        max_context_len=max_context_len,
        device=device,
        enable_memory_saver=False,
    )
    kv_pool = SWAKVPool(
        size=kv_size,
        size_swa=kv_size_swa,
        page_size=page_size,
        dtype=dtype,
        head_num=head_num,
        head_dim=head_dim,
        swa_attention_layer_ids=swa_attention_layer_ids,
        full_attention_layer_ids=full_attention_layer_ids,
        enable_kvcache_transpose=False,
        device=device,
    )
    allocator = SWATokenToKVPoolAllocator(
        size=kv_size,
        size_swa=kv_size_swa,
        page_size=page_size,
        dtype=dtype,
        device=device,
        kvcache=kv_pool,
        need_sort=False,
    )
    tree = SWARadixCache(
        params=CacheInitParams(
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=allocator,
            page_size=page_size,
            disable=False,
            is_eagle=is_eagle,
            sliding_window_size=sliding_window_size,
        ),
    )
    return tree, allocator, req_to_token_pool


def _swa_alloc(allocator, need_size):
    """SWA-pool alloc that also works for page_size > 1 (built-in alloc asserts page_size == 1)."""
    if allocator.page_size == 1:
        return allocator.alloc(need_size)

    assert need_size % allocator.page_size == 0
    full_indices = allocator.full_attn_allocator.alloc(need_size)
    swa_indices = allocator.swa_attn_allocator.alloc(need_size)
    assert full_indices is not None and swa_indices is not None
    allocator.full_to_swa_index_mapping[full_indices] = swa_indices
    return full_indices


def _insert(tree, allocator, token_ids):
    indices = _swa_alloc(allocator, len(token_ids))
    assert indices is not None
    tree.insert(InsertParams(key=RadixKey(token_ids), value=indices))


def _insert_chain(tree, allocator, token_ids):
    _insert(tree, allocator, token_ids)
    match = tree.match_prefix(MatchPrefixParams(key=RadixKey(token_ids)))
    return match.last_device_node


def _expected_tail_size(window: int, page_size: int) -> int:
    """Mirror of _maybe_split_leaf_for_swa_lock's tail_size formula."""
    return (window + page_size - 1) // page_size * page_size


class TestSWA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def test_swa_memory_pool(self):
        size = 16
        size_swa = 16
        page_size = 1
        head_num = 8
        head_dim = 128
        num_layers = 48
        global_interval = 4
        dtype = torch.bfloat16
        device = get_device()
        full_attention_layer_ids = [i for i in range(0, num_layers, global_interval)]
        full_attention_layer_ids_set = set(full_attention_layer_ids)
        swa_attention_layer_ids = [
            i for i in range(num_layers) if i not in full_attention_layer_ids_set
        ]
        pool = SWAKVPool(
            size=size,
            size_swa=size_swa,
            page_size=page_size,
            dtype=dtype,
            head_num=head_num,
            head_dim=head_dim,
            swa_attention_layer_ids=swa_attention_layer_ids,
            full_attention_layer_ids=full_attention_layer_ids,
            enable_kvcache_transpose=False,
            device=device,
        )
        alloc = SWATokenToKVPoolAllocator(
            size=size,
            size_swa=size_swa,
            page_size=page_size,
            dtype=dtype,
            device=device,
            kvcache=pool,
            need_sort=False,
        )
        self.assertEqual(
            alloc.full_available_size() + alloc.swa_available_size(), size + size_swa
        )
        index = alloc.alloc(1)
        self.assertEqual(
            alloc.full_available_size() + alloc.swa_available_size(),
            size_swa + size_swa - 2,
        )
        alloc.free_swa(index)
        result = alloc.translate_loc_from_full_to_swa(index)
        print(result)

    def test_swa_radix_cache_1(self):
        # args
        req_size = 10
        max_context_len = 128
        kv_size = 128
        kv_size_swa = 64
        page_size = 1
        sliding_window_size = 4
        head_num = 8
        head_dim = 128
        num_layers = 48
        global_interval = 4
        dtype = torch.bfloat16
        device = get_device()
        full_attention_layer_ids = [i for i in range(0, num_layers, global_interval)]
        full_attention_layer_ids_set = set(full_attention_layer_ids)
        swa_attention_layer_ids = [
            i for i in range(num_layers) if i not in full_attention_layer_ids_set
        ]
        # setup req to token pool
        req_to_token_pool = ReqToTokenPool(
            size=req_size,
            max_context_len=max_context_len,
            device=device,
            enable_memory_saver=False,
        )
        # setup kv pool
        kv_pool = SWAKVPool(
            size=kv_size,
            size_swa=kv_size_swa,
            page_size=page_size,
            dtype=dtype,
            head_num=head_num,
            head_dim=head_dim,
            swa_attention_layer_ids=swa_attention_layer_ids,
            full_attention_layer_ids=full_attention_layer_ids,
            enable_kvcache_transpose=False,
            device=device,
        )
        # setup token to kv pool allocator
        allocator = SWATokenToKVPoolAllocator(
            size=kv_size,
            size_swa=kv_size_swa,
            page_size=page_size,
            dtype=dtype,
            device=device,
            kvcache=kv_pool,
            need_sort=False,
        )
        # setup radix cache
        tree = SWARadixCache(
            params=CacheInitParams(
                req_to_token_pool=req_to_token_pool,
                token_to_kv_pool_allocator=allocator,
                disable=False,
                page_size=page_size,
                sliding_window_size=sliding_window_size,
            ),
        )

        # test
        print(
            f"[Start] allocator swa available size: {allocator.swa_available_size()}, full available size: {allocator.full_available_size()}"
        )
        req1_token_ids, req1_kv_indices = [1, 2, 3], allocator.alloc(3)
        self.assertEqual(len(req1_token_ids), len(req1_kv_indices))
        print(
            f"req1: inserting, req1_token_ids: {req1_token_ids}, req1_kv_indices: {req1_kv_indices}"
        )
        key = RadixKey(req1_token_ids)
        result = tree.insert(InsertParams(key=key, value=req1_kv_indices[: len(key)]))
        prefix_len = result.prefix_len
        print(
            f"req1: prefix_len: {prefix_len}, allocator swa available size: {allocator.swa_available_size()}, full available size: {allocator.full_available_size()}"
        )
        req2_token_ids, req2_kv_indices = [1, 2, 3, 4, 5, 6, 7], allocator.alloc(7)
        self.assertEqual(len(req2_token_ids), len(req2_kv_indices))
        print(
            f"req2: inserting, req2_token_ids: {req2_token_ids}, req2_kv_indices: {req2_kv_indices}"
        )
        key = RadixKey(req2_token_ids)
        result = tree.insert(InsertParams(key=key, value=req2_kv_indices[: len(key)]))
        prefix_len = result.prefix_len
        print(
            f"req2: prefix_len: {prefix_len}, allocator swa available size: {allocator.swa_available_size()}, full available size: {allocator.full_available_size()}"
        )
        req3_token_ids, req3_kv_indices = [10, 11, 12], allocator.alloc(3)
        self.assertEqual(len(req3_token_ids), len(req3_kv_indices))
        print(
            f"req3: inserting, req3_token_ids: {req3_token_ids}, req3_kv_indices: {req3_kv_indices}"
        )
        key = RadixKey(req3_token_ids)
        result = tree.insert(InsertParams(key=key, value=req3_kv_indices[: len(key)]))
        prefix_len = result.prefix_len
        print(
            f"req3: prefix_len: {prefix_len}, allocator swa available size: {allocator.swa_available_size()}, full available size: {allocator.full_available_size()}"
        )
        req4_token_ids, req4_kv_indices = [1, 2, 3, 4, 5, 60, 70], allocator.alloc(7)
        self.assertEqual(len(req4_token_ids), len(req4_kv_indices))
        print(
            f"req4: inserting, req4_token_ids: {req4_token_ids}, req4_kv_indices: {req4_kv_indices}"
        )
        key = RadixKey(req4_token_ids)
        result = tree.insert(InsertParams(key=key, value=req4_kv_indices[: len(key)]))
        prefix_len = result.prefix_len
        print(
            f"req4: prefix_len: {prefix_len}, allocator swa available size: {allocator.swa_available_size()}, full available size: {allocator.full_available_size()}"
        )

        tree.pretty_print()
        full_num_tokens, swa_num_tokens = 1, 0
        print(f"evicting {full_num_tokens} full token and {swa_num_tokens} swa token")
        tree.evict(
            EvictParams(num_tokens=full_num_tokens, swa_num_tokens=swa_num_tokens)
        )
        tree.pretty_print()

        full_num_tokens, swa_num_tokens = 0, 1
        print(f"evicting {full_num_tokens} full token and {swa_num_tokens} swa token")
        tree.evict(
            EvictParams(num_tokens=full_num_tokens, swa_num_tokens=swa_num_tokens)
        )
        tree.pretty_print()

        full_num_tokens, swa_num_tokens = 1, 2
        print(f"evicting {full_num_tokens} full token and {swa_num_tokens} swa token")
        tree.evict(
            EvictParams(num_tokens=full_num_tokens, swa_num_tokens=swa_num_tokens)
        )
        tree.pretty_print()

        req5_token_ids = [1, 2, 3, 4, 5]
        result = tree.match_prefix(MatchPrefixParams(key=RadixKey(req5_token_ids)))
        kv_indices, last_node = result.device_indices, result.last_device_node
        print(
            f"req5: token_ids: {req5_token_ids}, matched kv_indices: {kv_indices}, last_node.key: {last_node.key}"
        )
        self.assertEqual(len(kv_indices), 0)

        req6_token_ids = [1, 2, 3, 4, 5, 60, 70]
        result = tree.match_prefix(MatchPrefixParams(key=RadixKey(req6_token_ids)))
        kv_indices, last_node = result.device_indices, result.last_device_node
        print(
            f"req6: token_ids: {req6_token_ids}, matched kv_indices: {kv_indices}, last_node.key: {last_node.key}"
        )
        self.assertEqual(len(kv_indices), 7)
        self.assertEqual(len(last_node.key), 2)
        self.assertEqual(last_node.key.token_ids[0], 60)
        self.assertEqual(last_node.key.token_ids[1], 70)

        print(tree.available_and_evictable_str())
        print(available_and_evictable_str(tree))
        tree.sanity_check()

    def test_swa_radix_cache_eagle(self):
        # args
        req_size = 10
        max_context_len = 128
        kv_size = 128
        kv_size_swa = 64
        page_size = 1
        sliding_window_size = 4
        head_num = 8
        head_dim = 128
        num_layers = 48
        global_interval = 4
        dtype = torch.bfloat16
        device = get_device()
        full_attention_layer_ids = [i for i in range(0, num_layers, global_interval)]
        full_attention_layer_ids_set = set(full_attention_layer_ids)
        swa_attention_layer_ids = [
            i for i in range(num_layers) if i not in full_attention_layer_ids_set
        ]
        # setup req to token pool
        req_to_token_pool = ReqToTokenPool(
            size=req_size,
            max_context_len=max_context_len,
            device=device,
            enable_memory_saver=False,
        )
        # setup kv pool
        kv_pool = SWAKVPool(
            size=kv_size,
            size_swa=kv_size_swa,
            page_size=page_size,
            dtype=dtype,
            head_num=head_num,
            head_dim=head_dim,
            swa_attention_layer_ids=swa_attention_layer_ids,
            full_attention_layer_ids=full_attention_layer_ids,
            enable_kvcache_transpose=False,
            device=device,
        )
        # setup token to kv pool allocator
        allocator = SWATokenToKVPoolAllocator(
            size=kv_size,
            size_swa=kv_size_swa,
            page_size=page_size,
            dtype=dtype,
            device=device,
            kvcache=kv_pool,
            need_sort=False,
        )
        # setup radix cache
        tree = SWARadixCache(
            params=CacheInitParams(
                req_to_token_pool=req_to_token_pool,
                token_to_kv_pool_allocator=allocator,
                page_size=page_size,
                disable=False,
                is_eagle=True,
                sliding_window_size=sliding_window_size,
            ),
        )

        # test
        print(
            f"[Start] allocator swa available size: {allocator.swa_available_size()}, full available size: {allocator.full_available_size()}"
        )
        req1_token_ids, req1_kv_indices = [1, 2, 3], allocator.alloc(3)
        self.assertEqual(len(req1_token_ids), len(req1_kv_indices))
        print(
            f"req1: inserting, req1_token_ids: {req1_token_ids}, req1_kv_indices: {req1_kv_indices}"
        )
        key = RadixKey(req1_token_ids)
        result = tree.insert(InsertParams(key=key, value=req1_kv_indices[: len(key)]))
        prefix_len = result.prefix_len
        self.assertEqual(prefix_len, 0)
        print(
            f"req1: prefix_len: {prefix_len}, allocator swa available size: {allocator.swa_available_size()}, full available size: {allocator.full_available_size()}"
        )
        req2_token_ids, req2_kv_indices = [1, 2, 3, 4, 5, 6, 7], allocator.alloc(7)
        self.assertEqual(len(req2_token_ids), len(req2_kv_indices))
        print(
            f"req2: inserting, req2_token_ids: {req2_token_ids}, req2_kv_indices: {req2_kv_indices}"
        )
        key = RadixKey(req2_token_ids)
        result = tree.insert(InsertParams(key=key, value=req2_kv_indices[: len(key)]))
        prefix_len = result.prefix_len
        self.assertEqual(prefix_len, 2)
        print(
            f"req2: prefix_len: {prefix_len}, allocator swa available size: {allocator.swa_available_size()}, full available size: {allocator.full_available_size()}"
        )
        req3_token_ids, req3_kv_indices = [10, 11, 12], allocator.alloc(3)
        self.assertEqual(len(req3_token_ids), len(req3_kv_indices))
        print(
            f"req3: inserting, req3_token_ids: {req3_token_ids}, req3_kv_indices: {req3_kv_indices}"
        )
        key = RadixKey(req3_token_ids)
        result = tree.insert(InsertParams(key=key, value=req3_kv_indices[: len(key)]))
        prefix_len = result.prefix_len
        self.assertEqual(prefix_len, 0)
        print(
            f"req3: prefix_len: {prefix_len}, allocator swa available size: {allocator.swa_available_size()}, full available size: {allocator.full_available_size()}"
        )
        req4_token_ids, req4_kv_indices = [1, 2, 3, 4, 5, 60, 70], allocator.alloc(7)
        self.assertEqual(len(req4_token_ids), len(req4_kv_indices))
        print(
            f"req4: inserting, req4_token_ids: {req4_token_ids}, req4_kv_indices: {req4_kv_indices}"
        )
        key = RadixKey(req4_token_ids)
        result = tree.insert(InsertParams(key=key, value=req4_kv_indices[: len(key)]))
        prefix_len = result.prefix_len
        self.assertEqual(prefix_len, 4)
        print(
            f"req4: prefix_len: {prefix_len}, allocator swa available size: {allocator.swa_available_size()}, full available size: {allocator.full_available_size()}"
        )

        tree.pretty_print()
        full_num_tokens, swa_num_tokens = 1, 0
        print(f"evicting {full_num_tokens} full token and {swa_num_tokens} swa token")
        evict_result = tree.evict(
            EvictParams(num_tokens=full_num_tokens, swa_num_tokens=swa_num_tokens)
        )
        assert isinstance(evict_result, EvictResult)
        assert (
            evict_result.num_tokens_evicted >= full_num_tokens
        )  # May evict more due to node granularity
        print(
            f"evicted {evict_result.num_tokens_evicted} full tokens, {evict_result.swa_num_tokens_evicted} swa tokens"
        )
        tree.pretty_print()

        full_num_tokens, swa_num_tokens = 0, 1
        print(f"evicting {full_num_tokens} full token and {swa_num_tokens} swa token")
        evict_result = tree.evict(
            EvictParams(num_tokens=full_num_tokens, swa_num_tokens=swa_num_tokens)
        )
        assert isinstance(evict_result, EvictResult)
        assert (
            evict_result.swa_num_tokens_evicted >= swa_num_tokens
        ), f"evicted {evict_result.swa_num_tokens_evicted} swa tokens, expected {swa_num_tokens}"
        tree.pretty_print()

        full_num_tokens, swa_num_tokens = 1, 2
        print(f"evicting {full_num_tokens} full token and {swa_num_tokens} swa token")
        evict_result = tree.evict(
            EvictParams(num_tokens=full_num_tokens, swa_num_tokens=swa_num_tokens)
        )
        assert isinstance(evict_result, EvictResult)
        assert (
            evict_result.num_tokens_evicted >= full_num_tokens
        ), f"evicted {evict_result.num_tokens_evicted} full tokens, expected {full_num_tokens}"
        assert (
            evict_result.swa_num_tokens_evicted >= swa_num_tokens
        ), f"evicted {evict_result.swa_num_tokens_evicted} swa tokens, expected {swa_num_tokens}"
        tree.pretty_print()

        req5_token_ids = [1, 2, 3, 4, 5]
        result = tree.match_prefix(MatchPrefixParams(key=RadixKey(req5_token_ids)))
        kv_indices, last_node = result.device_indices, result.last_device_node
        print(
            f"req5: token_ids: {req5_token_ids}, matched kv_indices: {kv_indices}, last_node.key: {last_node.key}"
        )
        self.assertEqual(len(kv_indices), 0)  # no swa prefix matched

        req6_token_ids = [1, 2, 3, 4, 5, 60, 70]
        result = tree.match_prefix(MatchPrefixParams(key=RadixKey(req6_token_ids)))
        kv_indices, last_node = result.device_indices, result.last_device_node
        print(
            f"req6: token_ids: {req6_token_ids}, matched kv_indices: {kv_indices}, last_node.key: {last_node.key}"
        )
        self.assertEqual(len(kv_indices), 6)
        self.assertEqual(len(last_node.key), 2)
        # Bigram view: token_ids holds raw tokens; iteration yields bigram tuples.
        self.assertTrue(last_node.key.is_bigram)
        self.assertEqual(list(last_node.key), [(5, 60), (60, 70)])

    def test_swa_cache_finished_req_eagle_uses_cache_protected_len_and_bigram_key(self):
        tree, allocator, req_to_token_pool = _build_swa_tree(is_eagle=True)

        # Case 1: is_insert=True should pass bigram key and use cache_protected_len.
        req = _DummyReq()
        req.req_pool_idx = 0
        req.origin_input_ids = [1, 2, 3, 4, 5, 6]
        req.output_ids = []
        req._kv_committed_len = len(req.origin_input_ids)
        kv_indices = allocator.alloc(req._kv_committed_len)
        req_to_token_pool.write(
            (req.req_pool_idx, slice(0, req._kv_committed_len)), kv_indices
        )
        req.extra_key = None
        req.last_node = tree.root_node
        req.swa_uuid_for_lock = None
        req.swa_evicted_seqlen = 0
        req.cache_protected_len = 1
        # Intentionally mismatch to ensure code does not use len(prefix_indices).
        req.prefix_indices = torch.tensor([7, 8, 9, 10, 11], device=tree.device)

        captured = {}
        original_insert = tree.insert

        def wrapped_insert(params):
            captured["prev_prefix_len"] = params.prev_prefix_len
            captured["is_bigram"] = params.key.is_bigram
            captured["key_len"] = len(params.key)
            return original_insert(params)

        tree.insert = wrapped_insert
        tree.cache_finished_req(req, is_insert=True)

        self.assertEqual(captured["prev_prefix_len"], req.cache_protected_len)
        self.assertTrue(captured["is_bigram"])
        self.assertEqual(captured["key_len"], len(req.origin_input_ids) - 1)

        # Case 2: is_insert=False should free [cache_protected_len:page_aligned_len]
        # even when len(prefix_indices) is intentionally larger.
        req2 = _DummyReq()
        req2.req_pool_idx = 1
        req2.origin_input_ids = [11, 12, 13, 14, 15, 16]
        req2.output_ids = []
        req2._kv_committed_len = len(req2.origin_input_ids)
        kv_indices2 = allocator.alloc(req2._kv_committed_len)
        req_to_token_pool.write(
            (req2.req_pool_idx, slice(0, req2._kv_committed_len)), kv_indices2
        )
        req2.extra_key = None
        req2.last_node = tree.root_node
        req2.swa_uuid_for_lock = None
        req2.swa_evicted_seqlen = 0
        req2.cache_protected_len = 1
        req2.prefix_indices = torch.tensor([21, 22, 23, 24, 25], device=tree.device)

        freed_lens = []
        original_free = allocator.free

        def wrapped_free(indices):
            freed_lens.append(int(indices.numel()))
            return original_free(indices)

        allocator.free = wrapped_free
        tree.cache_finished_req(req2, is_insert=False)

        # EAGLE + page_size=1 => page_aligned_len = committed_len - 1 = 5
        # Expected frees:
        #   overlap range [1:5] -> 4
        #   tail range [5:]     -> 1
        self.assertEqual(freed_lens, [4, 1])


# Optimization: SGLANG_OPT_SWA_SPLIT_LEAF_ON_INSERT.
# Splits a freshly-inserted leaf at the (page-aligned) sliding-window
# boundary so a future inc_lock_ref protects only ~sliding_window_size SWA
# tokens instead of the whole chunked-prefill chain.
class TestSWASplitLeafOnInsert(CustomTestCase):
    def _insert_and_lock(self, *, window, page_size, leaf_len, flag_on):
        tree, allocator, _ = _build_swa_tree(
            is_eagle=False,
            kv_size=128,
            kv_size_swa=64,
            sliding_window_size=window,
            page_size=page_size,
        )
        token_ids = list(range(leaf_len))
        with envs.SGLANG_OPT_SWA_SPLIT_LEAF_ON_INSERT.override(flag_on):
            leaf = _insert_chain(tree, allocator, token_ids)
        result = tree.inc_lock_ref(leaf)
        return tree, leaf, result

    def test_flag_off_protects_full_leaf(self):
        tree, leaf, _ = self._insert_and_lock(
            window=4, page_size=1, leaf_len=12, flag_on=False
        )
        self.assertEqual(len(leaf.value), 12)
        self.assertEqual(tree.swa_protected_size_, 12)

    def test_flag_on_caps_protection_at_window(self):
        # (window, page_size, leaf_len, expected_tail_size); leaf_len picked
        # > tail_size and page-aligned for page_size > 1.
        cases = [
            (4, 1, 12, 4),
            (4, 1, 5, 4),
            (1, 1, 5, 1),
            (4, 2, 12, 4),
            (8, 2, 12, 8),
            (4, 4, 12, 4),
            # window NOT page-aligned -> tail rounds up to page boundary.
            (3, 2, 12, 4),
            (5, 4, 12, 8),
            (3, 4, 12, 4),
        ]
        for window, page_size, leaf_len, expected_tail in cases:
            with self.subTest(window=window, page_size=page_size, leaf_len=leaf_len):
                self.assertEqual(_expected_tail_size(window, page_size), expected_tail)
                tree, leaf, _ = self._insert_and_lock(
                    window=window,
                    page_size=page_size,
                    leaf_len=leaf_len,
                    flag_on=True,
                )
                self.assertEqual(len(leaf.value), expected_tail)
                self.assertEqual(tree.swa_protected_size_, expected_tail)

    def test_flag_on_no_split_when_leaf_within_window(self):
        # leaf_len <= tail_size: split must no-op.
        cases = [
            (4, 1, 4),
            (4, 1, 3),
            (4, 2, 4),
            (3, 2, 4),
            (8, 2, 4),
            (4, 4, 4),
        ]
        for window, page_size, leaf_len in cases:
            with self.subTest(window=window, page_size=page_size, leaf_len=leaf_len):
                tree, leaf, _ = self._insert_and_lock(
                    window=window,
                    page_size=page_size,
                    leaf_len=leaf_len,
                    flag_on=True,
                )
                self.assertEqual(len(leaf.value), leaf_len)
                self.assertEqual(tree.swa_protected_size_, leaf_len)

    def test_match_prefix_returns_full_chain_after_split(self):
        tree, allocator, _ = _build_swa_tree(
            is_eagle=False,
            kv_size=128,
            kv_size_swa=64,
            sliding_window_size=4,
            page_size=1,
        )
        token_ids = list(range(12))
        with envs.SGLANG_OPT_SWA_SPLIT_LEAF_ON_INSERT.override(True):
            inserted_leaf = _insert_chain(tree, allocator, token_ids)
        self.assertEqual(len(inserted_leaf.value), 4)
        match = tree.match_prefix(MatchPrefixParams(key=RadixKey(token_ids)))
        self.assertEqual(match.device_indices.shape[0], 12)
        self.assertIs(match.last_device_node, inserted_leaf)

    def test_dec_lock_ref_after_split_balances_to_zero(self):
        tree, leaf, result = self._insert_and_lock(
            window=4, page_size=1, leaf_len=12, flag_on=True
        )
        self.assertEqual(tree.swa_protected_size_, 4)
        self.assertEqual(tree.full_protected_size_, 12)

        tree.dec_lock_ref(
            leaf,
            params=DecLockRefParams(swa_uuid_for_lock=result.swa_uuid_for_lock),
        )

        self.assertEqual(tree.swa_protected_size_, 0)
        self.assertEqual(tree.full_protected_size_, 0)
        tree.sanity_check()


if __name__ == "__main__":
    unittest.main()
