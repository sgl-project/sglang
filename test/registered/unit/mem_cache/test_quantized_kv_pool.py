# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0

import types
import unittest

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeQuantMethod:
    name = "fake_quant"

    def __init__(self):
        self.k_scales_gpu = torch.tensor([2.0], dtype=torch.float32)
        self.v_scales_gpu = torch.tensor([3.0], dtype=torch.float32)
        self.store_calls = []

    def dequant_workspace_dtype(self):
        return torch.float32

    def create_buffers(self, size, head_num, head_dim, layer_num, device):
        return {
            "k_buffer": [
                torch.zeros(
                    (size, head_num, head_dim), dtype=torch.uint8, device=device
                )
                for _ in range(layer_num)
            ],
            "v_buffer": [
                torch.zeros(
                    (size, head_num, head_dim), dtype=torch.uint8, device=device
                )
                for _ in range(layer_num)
            ],
            "k_scale_buffer": [
                torch.zeros((size, head_num, 1), dtype=torch.uint8, device=device)
                for _ in range(layer_num)
            ],
            "v_scale_buffer": [
                torch.zeros((size, head_num, 1), dtype=torch.uint8, device=device)
                for _ in range(layer_num)
            ],
            "dq_k_buffer": torch.zeros(
                (size, head_num, head_dim), dtype=torch.float32, device=device
            ),
            "dq_v_buffer": torch.zeros(
                (size, head_num, head_dim), dtype=torch.float32, device=device
            ),
            "store_dtype": torch.uint8,
        }

    def quantize_and_store(
        self,
        k_buffer,
        v_buffer,
        k_scale_buffer,
        v_scale_buffer,
        loc,
        cache_k,
        cache_v,
        k_scale=None,
        v_scale=None,
    ):
        self.store_calls.append(
            {
                "loc": loc,
                "k_scale": k_scale,
                "v_scale": v_scale,
                "k_scale_buffer": k_scale_buffer,
                "v_scale_buffer": v_scale_buffer,
            }
        )
        k_buffer[loc] = 1
        v_buffer[loc] = 2
        k_scale_buffer[loc] = 3
        v_scale_buffer[loc] = 4


class TestQuantizedKVPool(unittest.TestCase):
    def test_quant_method_owns_buffers_and_store_path(self):
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

        quant_method = _FakeQuantMethod()
        pool = MHATokenToKVPool(
            size=4,
            page_size=1,
            dtype=torch.bfloat16,
            head_num=1,
            head_dim=8,
            layer_num=1,
            device="cpu",
            enable_memory_saver=False,
            quant_method=quant_method,
        )

        self.assertTrue(pool.is_quantized_kv_cache)
        self.assertIs(pool.quant_method, quant_method)
        self.assertIsNotNone(pool.k_scale_buffer)
        self.assertIs(pool.get_dequant_workspace()[0], pool.dq_k_buffer)

        loc = torch.tensor([0, 1], dtype=torch.int64)
        layer = types.SimpleNamespace(layer_id=0)
        pool.set_kv_buffer(
            layer,
            loc,
            torch.zeros((2, 1, 8), dtype=torch.bfloat16),
            torch.zeros((2, 1, 8), dtype=torch.bfloat16),
        )

        self.assertEqual(len(quant_method.store_calls), 1)
        call = quant_method.store_calls[0]
        self.assertIs(call["loc"], loc)
        self.assertTrue(torch.equal(call["k_scale"], quant_method.k_scales_gpu[0:1]))
        self.assertTrue(torch.equal(call["v_scale"], quant_method.v_scales_gpu[0:1]))
        self.assertEqual(pool.k_buffer[0][loc].unique().tolist(), [1])
        self.assertEqual(pool.v_buffer[0][loc].unique().tolist(), [2])


if __name__ == "__main__":
    unittest.main()
