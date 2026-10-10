import unittest

import torch

from sglang.srt.mem_cache.memory_pool import MHATokenToKOnlyPool, MHATokenToKVPool
from sglang.srt.mem_cache.pool_host.common import (
    _cuda_host_register,
    _cuda_host_unregister,
)
from sglang.srt.mem_cache.pool_host.mha import (
    MHATokenToKOnlyPoolHost,
    MHATokenToKVPoolHost,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=8, stage="base-b", runner_config="1-gpu-small")


class TestKOnlyHostRelease(CustomTestCase):
    def test_destroy_releases_registered_key_pages(self):
        common = dict(
            size=256,
            page_size=64,
            dtype=torch.bfloat16,
            head_num=1,
            head_dim=128,
            layer_num=2,
            device="cuda",
            enable_memory_saver=False,
        )
        main_pool = MHATokenToKVPool(**common)
        index_pool = MHATokenToKOnlyPool(**common)
        anchor = MHATokenToKVPoolHost(
            main_pool, 2, 0, 64, "layer_first", pin_memory=False
        )
        self.addCleanup(anchor.destroy)
        host = MHATokenToKOnlyPoolHost(index_pool, anchor, "layer_first")
        buffer = host.k_buffer
        self.addCleanup(_cuda_host_unregister, buffer)
        host.destroy()
        host.destroy()
        _cuda_host_register(buffer)
        self.assertIsNone(host.k_buffer)
        self.assertEqual(host.k_data_refs, [])
        torch.cuda.synchronize()


if __name__ == "__main__":
    unittest.main()
