import os
import unittest

import torch

from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.mamba_radix_cache import MambaRadixCache
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool, HybridReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.sampling.sampling_params import SamplingParams


class TestMamba(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def test_hybrid_linear_kv_pool(self):
        size = 16
        head_num = 2
        head_dim = 256
        num_layers = 48
        global_interval = 4
        dtype = torch.bfloat16
        device = "cuda"
        full_attention_layer_ids = [
            i for i in range(global_interval - 1, num_layers, global_interval)
        ]
        pool = HybridLinearKVPool(
            size=size,
            dtype=dtype,
            page_size=1,
            head_num=head_num,
            head_dim=head_dim,
            full_attention_layer_ids=full_attention_layer_ids,
            enable_kvcache_transpose=False,
            device=device,
            enable_memory_saver=False,
            mamba_pool=None,
        )
        assert pool._transfer_full_attention_id(global_interval - 1) == 0
        assert pool._transfer_full_attention_id(2 * global_interval - 1) == 1
        with self.assertRaises(ValueError) as context:
            pool._transfer_full_attention_id(1)
        self.assertIn(
            "layer_id=1 not in full attention layers:", str(context.exception)
        )

    def test_mamba_pool(self):
        max_num_reqs = 10
        mamba_cache_size = 20
        max_context_len = 128
        device = "cuda"
        global_interval = 4
        num_layers = 48
        full_attention_layer_ids = [
            i for i in range(global_interval - 1, num_layers, global_interval)
        ]
        mamba_layers = [
            i for i in range(num_layers) if i not in full_attention_layer_ids
        ]
        shape = Mamba2StateShape.create(
            tp_world_size=1,
            intermediate_size=4096,
            n_groups=16,
            num_heads=32,
            head_dim=128,
            state_size=128,
            conv_kernel=4,
        )
        os.environ["SGLANG_MAMBA_SSM_DTYPE"] = "bfloat16"
        mamba2_cache_params = Mamba2CacheParams(shape=shape, layers=mamba_layers)

        req_to_token_pool = HybridReqToTokenPool(
            size=max_num_reqs,
            mamba_size=mamba_cache_size,
            max_context_len=max_context_len,
            device=device,
            enable_memory_saver=False,
            cache_params=mamba2_cache_params,
            speculative_num_draft_tokens=3,
        )

        assert req_to_token_pool.available_size() == max_num_reqs
        assert req_to_token_pool.mamba_pool.available_size() == mamba_cache_size

        sampling_params = SamplingParams(
            temperature=0,
            max_new_tokens=1,
        )
        req = Req(
            rid=0,
            origin_input_text="",
            origin_input_ids=[],
            sampling_params=sampling_params,
        )

        # alloc req
        req_index = req_to_token_pool.alloc(1, [req])
        assert req_to_token_pool.available_size() == max_num_reqs - 1
        assert req_to_token_pool.mamba_pool.available_size() == mamba_cache_size - 1

        # free req
        req_to_token_pool.free(req_index)
        assert req_to_token_pool.available_size() == max_num_reqs
        assert req_to_token_pool.mamba_pool.available_size() == mamba_cache_size

        # alloc req without free mamba cache
        req.mamba_pool_idx = None
        req_index = req_to_token_pool.alloc(1, [req])
        req_to_token_pool.free(req_index, free_mamba_cache=False)
        assert req_to_token_pool.available_size() == max_num_reqs
        assert req_to_token_pool.mamba_pool.available_size() == mamba_cache_size - 1

        # alloc again
        req_index = req_to_token_pool.alloc(1, [req])
        assert req_to_token_pool.available_size() == max_num_reqs - 1
        assert req_to_token_pool.mamba_pool.available_size() == mamba_cache_size - 1

    def test_mamba_radix_cache_1(self):
        # kv cache
        size = 128
        dtype = torch.bfloat16
        head_num = 2
        head_dim = 256
        num_layers = 48
        global_interval = 4
        max_num_reqs = 10
        mamba_cache_size = 20
        max_context_len = 128
        device = "cuda"
        full_attention_layer_ids = [
            i for i in range(global_interval - 1, num_layers, global_interval)
        ]

        # mamba
        mamba_layers = [
            i for i in range(num_layers) if i not in full_attention_layer_ids
        ]
        os.environ["SGLANG_MAMBA_SSM_DTYPE"] = "bfloat16"
        shape = Mamba2StateShape.create(
            tp_world_size=1,
            intermediate_size=4096,
            n_groups=16,
            num_heads=32,
            head_dim=128,
            state_size=128,
            conv_kernel=4,
        )
        mamba2_cache_params = Mamba2CacheParams(shape=shape, layers=mamba_layers)

        req_to_token_pool = HybridReqToTokenPool(
            size=max_num_reqs,
            mamba_size=mamba_cache_size,
            max_context_len=max_context_len,
            device=device,
            enable_memory_saver=False,
            cache_params=mamba2_cache_params,
            speculative_num_draft_tokens=3,
        )
        # setup kv pool
        pool = HybridLinearKVPool(
            size=size,
            dtype=dtype,
            page_size=1,
            head_num=head_num,
            head_dim=head_dim,
            full_attention_layer_ids=full_attention_layer_ids,
            enable_kvcache_transpose=False,
            device=device,
            enable_memory_saver=False,
            mamba_pool=req_to_token_pool.mamba_pool,
        )

        # setup token to kv pool allocator
        allocator = TokenToKVPoolAllocator(
            size=size,
            dtype=dtype,
            device=device,
            kvcache=pool,
            need_sort=False,
        )
        # setup radix cache
        tree = MambaRadixCache(
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=allocator,
            page_size=1,
            disable=False,
        )

        def make_dummy_req():
            sampling_params = SamplingParams(
                temperature=0,
                max_new_tokens=1,
            )
            req = Req(
                rid=0,
                origin_input_text="",
                origin_input_ids=[],
                sampling_params=sampling_params,
            )
            req_to_token_pool.alloc(1, reqs=[req])
            return req

        mamba_pool = req_to_token_pool.mamba_pool
        # test
        print(
            f"[Start] allocator mamba available size: {mamba_pool.available_size()}, full available size: {allocator.available_size()}"
        )
        req1 = make_dummy_req()
        req1_token_ids, req1_kv_indices = [1, 2, 3], allocator.alloc(3)
        assert len(req1_token_ids) == len(req1_kv_indices)
        print(
            f"req1: inserting, req1_token_ids: {req1_token_ids}, req1_kv_indices: {req1_kv_indices}"
        )
        prefix_len = tree.insert(
            RadixKey(req1_token_ids), req1_kv_indices, req1.mamba_pool_idx.unsqueeze(0)
        )
        print(
            f"req1: prefix_len: {prefix_len}, allocator mamba available size: {mamba_pool.available_size()}, full available size: {allocator.available_size()}"
        )
        req2 = make_dummy_req()
        req2_token_ids, req2_kv_indices = [1, 2, 3, 4, 5, 6, 7], allocator.alloc(7)
        assert len(req2_token_ids) == len(req2_kv_indices)
        print(
            f"req2: inserting, req2_token_ids: {req2_token_ids}, req2_kv_indices: {req2_kv_indices}"
        )
        prefix_len = tree.insert(
            RadixKey(req2_token_ids), req2_kv_indices, req2.mamba_pool_idx.unsqueeze(0)
        )
        print(
            f"req2: prefix_len: {prefix_len}, allocator mamba available size: {mamba_pool.available_size()}, full available size: {allocator.available_size()}"
        )

        req3 = make_dummy_req()
        req3_token_ids, req3_kv_indices = [10, 11, 12], allocator.alloc(3)
        assert len(req3_token_ids) == len(req3_kv_indices)
        print(
            f"req3: inserting, req3_token_ids: {req3_token_ids}, req3_kv_indices: {req3_kv_indices}"
        )
        prefix_len = tree.insert(
            RadixKey(req3_token_ids), req3_kv_indices, req3.mamba_pool_idx.unsqueeze(0)
        )
        print(
            f"req3: prefix_len: {prefix_len}, allocator mamba available size: {mamba_pool.available_size()}, full available size: {allocator.available_size()}"
        )
        req4 = make_dummy_req()
        req4_token_ids, req4_kv_indices = [1, 2, 3, 4, 5, 60, 70], allocator.alloc(7)
        assert len(req4_token_ids) == len(req4_kv_indices)
        print(
            f"req4: inserting, req4_token_ids: {req4_token_ids}, req4_kv_indices: {req4_kv_indices}"
        )
        prefix_len = tree.insert(
            RadixKey(req4_token_ids), req4_kv_indices, req4.mamba_pool_idx.unsqueeze(0)
        )
        print(
            f"req4: prefix_len: {prefix_len}, allocator mamba available size: {mamba_pool.available_size()}, full available size: {allocator.available_size()}"
        )

        tree.pretty_print()
        full_num_tokens = 1
        print(f"evicting {full_num_tokens} full token")
        tree.evict(full_num_tokens=full_num_tokens)
        tree.pretty_print()

        mamba_num = 1
        print(f"evicting {mamba_num} mamba")
        tree.evict_mamba(mamba_num=mamba_num)
        tree.pretty_print()

        req5_token_ids = [1, 2, 3, 4, 5]
        result = tree.match_prefix(RadixKey(req5_token_ids))
        kv_indices, last_node = result.device_indices, result.last_device_node
        print(
            f"req5: token_ids: {req5_token_ids}, matched kv_indices: {kv_indices}, last_node.key: {last_node.key}"
        )
        assert len(kv_indices) == 0

        req6_token_ids = [1, 2, 3, 4, 5, 60, 70]
        result = tree.match_prefix(RadixKey(req6_token_ids))
        kv_indices, last_node = result.device_indices, result.last_device_node
        print(
            f"req6: token_ids: {req6_token_ids}, matched kv_indices: {kv_indices}, last_node.key: {last_node.key}"
        )
        assert len(kv_indices) == 7
        assert len(last_node.key) == 2

        req7_token_ids = [1, 2, 3, 4, 5, 6, 7]
        result = tree.match_prefix(RadixKey(req7_token_ids))
        kv_indices, last_node = result.device_indices, result.last_device_node
        print(
            f"req7: token_ids: {req7_token_ids}, matched kv_indices: {kv_indices}, last_node.key: {last_node.key}"
        )
        assert len(kv_indices) == 7
        assert len(last_node.key) == 2

        mamba_num = 1
        print(f"evicting {mamba_num} mamba")
        tree.evict_mamba(mamba_num=mamba_num)
        tree.pretty_print()

        req8_token_ids = [1, 2, 3, 4, 5, 60, 70]
        result = tree.match_prefix(RadixKey(req8_token_ids))
        kv_indices, last_node = result.device_indices, result.last_device_node
        print(
            f"req8: token_ids: {req8_token_ids}, matched kv_indices: {kv_indices}, last_node.key: {last_node.key}"
        )
        assert len(kv_indices) == 0
        assert len(last_node.key) == 0

        req9_token_ids = [1, 2, 3, 4, 5, 6, 7]
        req9 = make_dummy_req()
        result = tree.match_prefix(
            RadixKey(req9_token_ids), **({"req": req9, "cow_mamba": True})
        )
        kv_indices, last_node = result.device_indices, result.last_device_node
        assert req9.mamba_pool_idx is not None
        assert torch.all(
            mamba_pool.mamba_cache.conv[0][:, req9.mamba_pool_idx]
            == mamba_pool.mamba_cache.conv[0][:, last_node.mamba_value]
        )
        assert torch.all(
            mamba_pool.mamba_cache.temporal[:, req9.mamba_pool_idx]
            == mamba_pool.mamba_cache.temporal[:, last_node.mamba_value]
        )


if __name__ == "__main__":
    unittest.main()
