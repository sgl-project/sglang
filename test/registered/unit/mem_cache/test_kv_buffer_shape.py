"""Unit tests for KV buffer shape helpers."""

import unittest

import torch

from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestKVBufferShape(CustomTestCase):
    def test_get_kv_buffer_shape_uses_local_start_layer(self):
        pool = MHATokenToKVPool(
            size=4,
            page_size=1,
            dtype=torch.float16,
            head_num=2,
            head_dim=4,
            layer_num=2,
            device="cpu",
            enable_memory_saver=False,
            start_layer=4,
            end_layer=6,
            enable_alt_stream=False,
        )

        with self.assertRaises(IndexError):
            pool.get_key_buffer(0)

        k_shape, v_shape = pool.get_kv_buffer_shape()
        self.assertEqual(k_shape, torch.Size((5, 2, 4)))
        self.assertEqual(v_shape, torch.Size((5, 2, 4)))
        self.assertEqual(k_shape, pool.get_key_buffer(pool.start_layer).shape)
        self.assertEqual(v_shape, pool.get_value_buffer(pool.start_layer).shape)


if __name__ == "__main__":
    unittest.main()
