"""Regression for low-ratio indexer capacity at the highest cached FULL page."""

import unittest
from array import array

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA indexer stores")
class TestDeepseekV41PrefixCache(CustomTestCase):
    def setUp(self):
        self.page_size = 256
        self.size = 2 * self.page_size
        self.device = "cuda"
        set_global_server_args_for_scheduler(
            ServerArgs(model_path="dummy", page_size=self.page_size)
        )
        # One source and one sharing layer for each low ratio.
        self.pool = DeepSeekV4TokenToKVPool(
            max_num_reqs=1,
            swa_size=self.size,
            c4_size=0,
            c128_size=0,
            c4_state_pool_size=0,
            c128_state_pool_size=0,
            page_size=self.page_size,
            swa_page_size=128,
            dtype=torch.float8_e4m3fn,
            c4_state_dtype=torch.float32,
            c128_state_dtype=torch.float32,
            qk_nope_head_dim=448,
            qk_rope_head_dim=64,
            indexer_head_dim=128,
            layer_num=4,
            device=self.device,
            enable_memory_saver=False,
            compression_ratios=[2, 2, 1, 1],
            kv_source_layers=[0, 2],
            full_size=self.size,
        )
        self.allocator = SWATokenToKVPoolAllocator(
            self.size,
            self.size,
            self.page_size,
            torch.int64,
            self.device,
            self.pool,
            False,
        )
        with envs.SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND.override("python"):
            self.cache = UnifiedRadixCache(
                CacheInitParams(
                    disable=False,
                    req_to_token_pool=ReqToTokenPool(1, self.size, self.device, False),
                    token_to_kv_pool_allocator=self.allocator,
                    page_size=self.page_size,
                    sliding_window_size=128,
                    tree_components=(ComponentType.FULL, ComponentType.SWA),
                )
            )

    def test_cached_highest_page_reads_shared_indexer_stores(self):
        locs = self.allocator.full_attn_allocator.alloc(self.size)
        swa = self.allocator.swa_attn_allocator.alloc(self.size)
        self.allocator.set_full_to_swa_mapping(locs, swa)
        self.assertEqual(int(locs[-1]), self.size + self.page_size - 1)
        # Reverse the pages so a match must retain physical, not logical, slots.
        locs = locs.view(2, self.page_size).flip(0).flatten()
        key = RadixKey(array("q", range(self.size)))
        self.cache.insert(InsertParams(key=key, value=locs))
        hit = self.cache.match_prefix(MatchPrefixParams(key=key)).device_indices
        torch.testing.assert_close(hit, locs)

        for ratio, source, consumer in ((2, 0, 1), (1, 2, 3)):
            with self.subTest(ratio=ratio):
                slots = locs[::ratio] // ratio
                index_pool = self.pool.index_pools[ratio]
                capacity = (
                    index_pool.get_index_k_with_scale_buffer(0).shape[0]
                    * index_pool.page_size
                )
                # Fail before a GPU write if reserved page 0 was omitted.
                self.assertLess(int(slots.max()), capacity)
                keys = (
                    torch.randint(
                        0,
                        2,
                        (len(slots), 128),
                        generator=torch.Generator().manual_seed(ratio),
                    )
                    .to(device=self.device, dtype=torch.bfloat16)
                    .mul_(2)
                    .sub_(1)
                )
                self.pool.set_index_k_fp4(source, slots, keys)
                self.assertEqual(self.pool.source_layer_of(consumer), source)
                for layer in (source, consumer):
                    actual = self.pool.get_low_ratio_index_k_dequant(
                        layer, hit[::ratio] // ratio
                    )
                    torch.testing.assert_close(actual, keys, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
