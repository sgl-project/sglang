import unittest

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
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

register_cuda_ci(est_time=8, suite="stage-b-test-large-1-gpu")
register_amd_ci(est_time=10, suite="stage-b-test-small-1-gpu-amd")


class TestSWA(unittest.TestCase):
    class _DummyReq:
        def __init__(self):
            self._kv_committed_len = 0

        def pop_committed_kv_cache(self):
            return self._kv_committed_len

    def _build_swa_tree(
        self,
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
        result = tree.insert(
            InsertParams(key=RadixKey(req1_token_ids), value=req1_kv_indices)
        )
        prefix_len = result.prefix_len
        print(
            f"req1: prefix_len: {prefix_len}, allocator swa available size: {allocator.swa_available_size()}, full available size: {allocator.full_available_size()}"
        )
        req2_token_ids, req2_kv_indices = [1, 2, 3, 4, 5, 6, 7], allocator.alloc(7)
        self.assertEqual(len(req2_token_ids), len(req2_kv_indices))
        print(
            f"req2: inserting, req2_token_ids: {req2_token_ids}, req2_kv_indices: {req2_kv_indices}"
        )
        result = tree.insert(
            InsertParams(key=RadixKey(req2_token_ids), value=req2_kv_indices)
        )
        prefix_len = result.prefix_len
        print(
            f"req2: prefix_len: {prefix_len}, allocator swa available size: {allocator.swa_available_size()}, full available size: {allocator.full_available_size()}"
        )
        req3_token_ids, req3_kv_indices = [10, 11, 12], allocator.alloc(3)
        self.assertEqual(len(req3_token_ids), len(req3_kv_indices))
        print(
            f"req3: inserting, req3_token_ids: {req3_token_ids}, req3_kv_indices: {req3_kv_indices}"
        )
        result = tree.insert(
            InsertParams(key=RadixKey(req3_token_ids), value=req3_kv_indices)
        )
        prefix_len = result.prefix_len
        print(
            f"req3: prefix_len: {prefix_len}, allocator swa available size: {allocator.swa_available_size()}, full available size: {allocator.full_available_size()}"
        )
        req4_token_ids, req4_kv_indices = [1, 2, 3, 4, 5, 60, 70], allocator.alloc(7)
        self.assertEqual(len(req4_token_ids), len(req4_kv_indices))
        print(
            f"req4: inserting, req4_token_ids: {req4_token_ids}, req4_kv_indices: {req4_kv_indices}"
        )
        result = tree.insert(
            InsertParams(key=RadixKey(req4_token_ids), value=req4_kv_indices)
        )
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
        result = tree.insert(
            InsertParams(key=RadixKey(req1_token_ids), value=req1_kv_indices)
        )
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
        result = tree.insert(
            InsertParams(key=RadixKey(req2_token_ids), value=req2_kv_indices)
        )
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
        result = tree.insert(
            InsertParams(key=RadixKey(req3_token_ids), value=req3_kv_indices)
        )
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
        result = tree.insert(
            InsertParams(key=RadixKey(req4_token_ids), value=req4_kv_indices)
        )
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
        self.assertEqual(last_node.key.token_ids[0], (5, 60))
        self.assertEqual(last_node.key.token_ids[1], (60, 70))

    def test_swa_cache_finished_req_eagle_uses_cache_protected_len_and_bigram_key(self):
        tree, allocator, req_to_token_pool = self._build_swa_tree(is_eagle=True)

        # Case 1: is_insert=True should pass bigram key and use cache_protected_len.
        req = self._DummyReq()
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
        req2 = self._DummyReq()
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


if __name__ == "__main__":
    unittest.main()
