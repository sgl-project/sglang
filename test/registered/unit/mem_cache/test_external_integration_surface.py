"""Request-table calls that integrations outside sglang make.

LMCache's sglang adapter (`lmcache/integration/sglang/unified_kv_adapter.py`)
translates a cached recurrent-state slot through the request table before it
stores or retrieves the state: `req_to_token_pool.translate_mamba_indices(
mamba_value.view(-1))`. A static hybrid table answers with the same ids."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

import unittest

import torch

from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
from sglang.test.test_utils import CustomTestCase


class TestExternalIntegrationSurface(CustomTestCase):
    def test_lmcache_translates_a_state_slot_through_the_table(self):
        shape = Mamba2StateShape.create(
            tp_world_size=1,
            intermediate_size=64,
            n_groups=1,
            num_heads=4,
            head_dim=16,
            state_size=8,
            conv_kernel=4,
        )
        table = HybridReqToTokenPool(
            size=4,
            mamba_size=8,
            mamba_spec_state_size=4,
            max_context_len=32,
            device="cpu",
            enable_memory_saver=False,
            cache_params=Mamba2CacheParams(shape=shape, layers=[0]),
            mamba_layer_ids=[0],
            enable_mamba_extra_buffer=False,
            speculative_num_draft_tokens=3,
        )
        mamba_value = torch.tensor([[7]], dtype=torch.int64)
        physical = table.translate_mamba_indices(mamba_value.view(-1))
        self.assertEqual(physical.tolist(), [7])
        self.assertEqual(physical.dtype, torch.int64)


if __name__ == "__main__":
    unittest.main()
