import unittest

import torch

from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool
from sglang.srt.mem_cache.pool_host.common import (
    _cuda_host_register,
    _cuda_host_unregister,
)
from sglang.srt.mem_cache.pool_host.dsa import (
    DSAIndexerPoolHost,
    make_dsa_indexer_pool_decl,
)
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=8, stage="base-b", runner_config="1-gpu-small")


class TestDSAHostRelease(CustomTestCase):
    def test_destroy_releases_index_pages_for_registration_reuse(self):
        pool = DSATokenToKVPool(
            size=256,
            page_size=64,
            kv_lora_rank=128,
            dtype=torch.bfloat16,
            qk_rope_head_dim=32,
            layer_num=2,
            device="cuda",
            enable_memory_saver=False,
            kv_cache_dim=160,
            index_head_dim=128,
        )
        anchor = MLATokenToKVPoolHost(
            pool,
            host_to_device_ratio=2,
            host_size=0,
            page_size=64,
            layout="layer_first",
            pin_memory=False,
        )
        self.addCleanup(anchor.destroy)
        host = DSAIndexerPoolHost(
            decl=make_dsa_indexer_pool_decl(pool), anchor_host=anchor
        )
        buffer = host.index_k_with_scale_buffer
        self.addCleanup(_cuda_host_unregister, buffer)
        host.destroy()
        self.assertIsNone(host.index_k_with_scale_buffer)
        self.assertEqual(host.index_k_data_refs, [])
        host.destroy()
        _cuda_host_register(buffer)
        torch.cuda.synchronize()


if __name__ == "__main__":
    unittest.main()
