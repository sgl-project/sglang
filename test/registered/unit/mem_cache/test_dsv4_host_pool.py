"""Unit tests for bugfix #26289: DeepSeekV4SingleKVPoolHost missing get_contiguous_buf_infos.

When --enable-hisparse is used in PD disaggregation mode, the decode path
calls transfer_kv_pool.get_contiguous_buf_infos(), but DeepSeekV4SingleKVPoolHost
does not define this method, causing an AttributeError crash at startup.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.mem_cache.hisparse_memory_pool import (  # noqa: E402
    DeepSeekV4SingleKVPoolHost,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDeepSeekV4SingleKVPoolHostGetContiguousBufInfos(CustomTestCase):

    def _make_host_pool(self, layer_num=2, size=256, page_size=16, dim=64):
        """Create a host pool without going through init_kv_buffer (which requires pin_memory)."""
        host_pool = object.__new__(DeepSeekV4SingleKVPoolHost)
        host_pool.layer_num = layer_num
        host_pool.size = size
        host_pool.page_size = page_size
        host_pool.kv_cache_total_dim = dim
        host_pool.dtype = torch.float16
        host_pool.kv_buffer = torch.empty(
            layer_num, size + page_size, dim, dtype=torch.float16
        )
        return host_pool

    def test_get_contiguous_buf_infos_exists(self):
        """Method should exist and be callable."""
        host_pool = self._make_host_pool()
        self.assertTrue(hasattr(host_pool, "get_contiguous_buf_infos"))
        self.assertTrue(callable(host_pool.get_contiguous_buf_infos))

    def test_get_contiguous_buf_infos_returns_three_lists(self):
        """Should return (data_ptrs, data_lens, item_lens)."""
        host_pool = self._make_host_pool(layer_num=2)
        result = host_pool.get_contiguous_buf_infos()
        self.assertEqual(len(result), 3)
        data_ptrs, data_lens, item_lens = result
        self.assertEqual(len(data_ptrs), 2)
        self.assertEqual(len(data_lens), 2)
        self.assertEqual(len(item_lens), 2)

    def test_data_ptrs_are_ints(self):
        """Data pointers should be integer memory addresses."""
        host_pool = self._make_host_pool(layer_num=1)
        data_ptrs, data_lens, item_lens = host_pool.get_contiguous_buf_infos()
        for ptr in data_ptrs:
            self.assertIsInstance(ptr, int)

    def test_data_lens_match_buffer_sizes(self):
        """Data lengths should match the byte size of each layer buffer."""
        layer_num = 3
        host_pool = self._make_host_pool(layer_num=layer_num)
        data_ptrs, data_lens, item_lens = host_pool.get_contiguous_buf_infos()
        self.assertEqual(len(data_lens), layer_num)
        for i in range(layer_num):
            self.assertEqual(data_lens[i], host_pool.kv_buffer[i].nbytes)

    def test_item_lens_match_page_sizes(self):
        """Item lengths should match the byte size of one page in each layer."""
        host_pool = self._make_host_pool(layer_num=2, dim=128, page_size=16)
        data_ptrs, data_lens, item_lens = host_pool.get_contiguous_buf_infos()
        for i in range(2):
            self.assertEqual(
                item_lens[i], host_pool.page_size * host_pool.kv_buffer[i][0].nbytes
            )


if __name__ == "__main__":
    unittest.main()
