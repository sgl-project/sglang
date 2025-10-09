import unittest

import torch

from sglang.srt.mem_cache.allocator import SWAKVPool, SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.swa_radix_cache import SWARadixCache


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
        head_num = 8
        head_dim = 128
        num_layers = 48
        global_interval = 4
        dtype = torch.bfloat16
        device = "cuda"
        full_attention_layer_ids = [i for i in range(0, num_layers, global_interval)]
        full_attention_layer_ids_set = set(full_attention_layer_ids)
        swa_attention_layer_ids = [
            i for i in range(num_layers) if i not in full_attention_layer_ids_set
        ]
        pool = SWAKVPool(
            size=size,
            size_swa=size_swa,
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
        sliding_window_size = 4
        head_num = 8
        head_dim = 128
        num_layers = 48
        global_interval = 4
        dtype = torch.bfloat16
        device = "cuda"
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
            dtype=dtype,
            device=device,
            kvcache=kv_pool,
            need_sort=False,
        )
        # setup radix cache
        tree = SWARadixCache(
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=allocator,
            sliding_window_size=sliding_window_size,
            page_size=1,
            disable=False,
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
        prefix_len = tree.insert(RadixKey(req1_token_ids), req1_kv_indices)
        print(
            f"req1: prefix_len: {prefix_len}, allocator swa available size: {allocator.swa_available_size()}, full available size: {allocator.full_available_size()}"
        )
        req2_token_ids, req2_kv_indices = [1, 2, 3, 4, 5, 6, 7], allocator.alloc(7)
        self.assertEqual(len(req2_token_ids), len(req2_kv_indices))
        print(
            f"req2: inserting, req2_token_ids: {req2_token_ids}, req2_kv_indices: {req2_kv_indices}"
        )
        prefix_len = tree.insert(RadixKey(req2_token_ids), req2_kv_indices)
        print(
            f"req2: prefix_len: {prefix_len}, allocator swa available size: {allocator.swa_available_size()}, full available size: {allocator.full_available_size()}"
        )
        req3_token_ids, req3_kv_indices = [10, 11, 12], allocator.alloc(3)
        self.assertEqual(len(req3_token_ids), len(req3_kv_indices))
        print(
            f"req3: inserting, req3_token_ids: {req3_token_ids}, req3_kv_indices: {req3_kv_indices}"
        )
        prefix_len = tree.insert(RadixKey(req3_token_ids), req3_kv_indices)
        print(
            f"req3: prefix_len: {prefix_len}, allocator swa available size: {allocator.swa_available_size()}, full available size: {allocator.full_available_size()}"
        )
        req4_token_ids, req4_kv_indices = [1, 2, 3, 4, 5, 60, 70], allocator.alloc(7)
        self.assertEqual(len(req4_token_ids), len(req4_kv_indices))
        print(
            f"req4: inserting, req4_token_ids: {req4_token_ids}, req4_kv_indices: {req4_kv_indices}"
        )
        prefix_len = tree.insert(RadixKey(req4_token_ids), req4_kv_indices)
        print(
            f"req4: prefix_len: {prefix_len}, allocator swa available size: {allocator.swa_available_size()}, full available size: {allocator.full_available_size()}"
        )

        tree.pretty_print()
        full_num_tokens, swa_num_tokens = 1, 0
        print(f"evicting {full_num_tokens} full token and {swa_num_tokens} swa token")
        tree.evict(full_num_tokens=full_num_tokens, swa_num_tokens=swa_num_tokens)
        tree.pretty_print()

        full_num_tokens, swa_num_tokens = 0, 1
        print(f"evicting {full_num_tokens} full token and {swa_num_tokens} swa token")
        tree.evict(full_num_tokens=full_num_tokens, swa_num_tokens=swa_num_tokens)
        tree.pretty_print()

        full_num_tokens, swa_num_tokens = 1, 2
        print(f"evicting {full_num_tokens} full token and {swa_num_tokens} swa token")
        tree.evict(full_num_tokens=full_num_tokens, swa_num_tokens=swa_num_tokens)
        tree.pretty_print()

        req5_token_ids = [1, 2, 3, 4, 5]
        result = tree.match_prefix(RadixKey(req5_token_ids))
        kv_indices, last_node = result.device_indices, result.last_device_node
        print(
            f"req5: token_ids: {req5_token_ids}, matched kv_indices: {kv_indices}, last_node.key: {last_node.key}"
        )
        self.assertEqual(len(kv_indices), 0)

        req6_token_ids = [1, 2, 3, 4, 5, 60, 70]
        result = tree.match_prefix(RadixKey(req6_token_ids))
        kv_indices, last_node = result.device_indices, result.last_device_node
        print(
            f"req6: token_ids: {req6_token_ids}, matched kv_indices: {kv_indices}, last_node.key: {last_node.key}"
        )
        self.assertEqual(len(kv_indices), 7)
        self.assertEqual(len(last_node.key), 2)
        self.assertEqual(last_node.key.token_ids[0], 60)
        self.assertEqual(last_node.key.token_ids[1], 70)

    def test_swa_radix_cache_eagle(self):
        # args
        req_size = 10
        max_context_len = 128
        kv_size = 128
        kv_size_swa = 64
        sliding_window_size = 4
        head_num = 8
        head_dim = 128
        num_layers = 48
        global_interval = 4
        dtype = torch.bfloat16
        device = "cuda"
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
            dtype=dtype,
            device=device,
            kvcache=kv_pool,
            need_sort=False,
        )
        # setup radix cache
        tree = SWARadixCache(
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=allocator,
            sliding_window_size=sliding_window_size,
            page_size=1,
            disable=False,
            is_eagle=True,
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
        prefix_len = tree.insert(RadixKey(req1_token_ids), req1_kv_indices)
        self.assertEqual(prefix_len, 0)
        print(
            f"req1: prefix_len: {prefix_len}, allocator swa available size: {allocator.swa_available_size()}, full available size: {allocator.full_available_size()}"
        )
        req2_token_ids, req2_kv_indices = [1, 2, 3, 4, 5, 6, 7], allocator.alloc(7)
        self.assertEqual(len(req2_token_ids), len(req2_kv_indices))
        print(
            f"req2: inserting, req2_token_ids: {req2_token_ids}, req2_kv_indices: {req2_kv_indices}"
        )
        prefix_len = tree.insert(RadixKey(req2_token_ids), req2_kv_indices)
        self.assertEqual(prefix_len, 2)
        print(
            f"req2: prefix_len: {prefix_len}, allocator swa available size: {allocator.swa_available_size()}, full available size: {allocator.full_available_size()}"
        )
        req3_token_ids, req3_kv_indices = [10, 11, 12], allocator.alloc(3)
        self.assertEqual(len(req3_token_ids), len(req3_kv_indices))
        print(
            f"req3: inserting, req3_token_ids: {req3_token_ids}, req3_kv_indices: {req3_kv_indices}"
        )
        prefix_len = tree.insert(RadixKey(req3_token_ids), req3_kv_indices)
        self.assertEqual(prefix_len, 0)
        print(
            f"req3: prefix_len: {prefix_len}, allocator swa available size: {allocator.swa_available_size()}, full available size: {allocator.full_available_size()}"
        )
        req4_token_ids, req4_kv_indices = [1, 2, 3, 4, 5, 60, 70], allocator.alloc(7)
        self.assertEqual(len(req4_token_ids), len(req4_kv_indices))
        print(
            f"req4: inserting, req4_token_ids: {req4_token_ids}, req4_kv_indices: {req4_kv_indices}"
        )
        prefix_len = tree.insert(RadixKey(req4_token_ids), req4_kv_indices)
        self.assertEqual(prefix_len, 4)
        print(
            f"req4: prefix_len: {prefix_len}, allocator swa available size: {allocator.swa_available_size()}, full available size: {allocator.full_available_size()}"
        )

        tree.pretty_print()
        full_num_tokens, swa_num_tokens = 1, 0
        print(f"evicting {full_num_tokens} full token and {swa_num_tokens} swa token")
        tree.evict(full_num_tokens=full_num_tokens, swa_num_tokens=swa_num_tokens)
        tree.pretty_print()

        full_num_tokens, swa_num_tokens = 0, 1
        print(f"evicting {full_num_tokens} full token and {swa_num_tokens} swa token")
        tree.evict(full_num_tokens=full_num_tokens, swa_num_tokens=swa_num_tokens)
        tree.pretty_print()

        full_num_tokens, swa_num_tokens = 1, 2
        print(f"evicting {full_num_tokens} full token and {swa_num_tokens} swa token")
        tree.evict(full_num_tokens=full_num_tokens, swa_num_tokens=swa_num_tokens)
        tree.pretty_print()

        req5_token_ids = [1, 2, 3, 4, 5]
        result = tree.match_prefix(RadixKey(req5_token_ids))
        kv_indices, last_node = result.device_indices, result.last_device_node
        print(
            f"req5: token_ids: {req5_token_ids}, matched kv_indices: {kv_indices}, last_node.key: {last_node.key}"
        )
        self.assertEqual(len(kv_indices), 0)  # no swa prefix matched

        req6_token_ids = [1, 2, 3, 4, 5, 60, 70]
        result = tree.match_prefix(RadixKey(req6_token_ids))
        kv_indices, last_node = result.device_indices, result.last_device_node
        print(
            f"req6: token_ids: {req6_token_ids}, matched kv_indices: {kv_indices}, last_node.key: {last_node.key}"
        )
        self.assertEqual(len(kv_indices), 6)
        self.assertEqual(len(last_node.key), 2)
        self.assertEqual(last_node.key.token_ids[0], (5, 60))
        self.assertEqual(last_node.key.token_ids[1], (60, 70))


if __name__ == "__main__":
    unittest.main()
