"""Overflowing KV values must remain finite after conversion to FP8 storage."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, MLATokenToKVPool
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


class TestFP8KVCacheSaturation(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.enterContext(
            get_parallel().override(dcp_enabled=False, attn_dcp_size=1, attn_dcp_rank=0)
        )
        # Slot zero is reserved for CUDA-graph padding.
        self.loc = torch.tensor([1], dtype=torch.int64, device="cuda")

    def test_overflowing_kv_values_saturate_instead_of_nan(self):
        pool = MHATokenToKVPool(
            size=1,
            page_size=1,
            dtype=torch.float8_e4m3fn,
            head_num=1,
            head_dim=4,
            layer_num=1,
            device="cuda",
            enable_memory_saver=False,
        )
        cache_k = torch.tensor(
            [[[688.0, -700.0, 448.0, 1.0]]], dtype=torch.bfloat16, device="cuda"
        )
        cache_v = torch.tensor(
            [[[-449.0, 573.0, -448.0, 2.0]]], dtype=torch.bfloat16, device="cuda"
        )
        layer = SimpleNamespace(layer_id=0)

        pool.set_kv_buffer(
            layer,
            self.loc,
            cache_k,
            cache_v,
        )

        expected_k = torch.tensor(
            [[[448.0, -448.0, 448.0, 1.0]]], dtype=torch.float32, device="cuda"
        )
        expected_v = torch.tensor(
            [[[-448.0, 448.0, -448.0, 2.0]]], dtype=torch.float32, device="cuda"
        )
        torch.testing.assert_close(pool.get_key_buffer(0)[1].float(), expected_k[0])
        torch.testing.assert_close(pool.get_value_buffer(0)[1].float(), expected_v[0])

    def test_overflowing_mla_combined_values_saturate_instead_of_nan(self):
        self._check_mla_saturation(split=False)

    def test_overflowing_mla_split_values_saturate_instead_of_nan(self):
        self._check_mla_saturation(split=True)

    def _check_mla_saturation(self, split):
        pool = MLATokenToKVPool(
            size=1,
            page_size=1,
            dtype=torch.float8_e4m3fn,
            kv_lora_rank=4,
            qk_rope_head_dim=2,
            layer_num=1,
            device="cuda",
            enable_memory_saver=False,
        )
        cache_k_nope = torch.tensor(
            [[[688.0, -700.0, 448.0, -449.0]]],
            dtype=torch.bfloat16,
            device="cuda",
        )
        cache_k_rope = torch.tensor(
            [[[573.0, -448.0]]],
            dtype=torch.bfloat16,
            device="cuda",
        )
        layer = SimpleNamespace(layer_id=0)

        if split:
            pool.set_mla_kv_buffer(layer, self.loc, cache_k_nope, cache_k_rope)
        else:
            cache_k = torch.cat((cache_k_nope, cache_k_rope), dim=-1)
            pool.set_kv_buffer(layer, self.loc, cache_k, cache_k)

        expected_k = torch.tensor(
            [[[448.0, -448.0, 448.0, -448.0, 448.0, -448.0]]],
            dtype=torch.float32,
            device="cuda",
        )
        torch.testing.assert_close(pool.get_key_buffer(0)[1].float(), expected_k[0])


if __name__ == "__main__":
    unittest.main()
