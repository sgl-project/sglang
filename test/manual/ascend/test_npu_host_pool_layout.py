"""CPU regressions for NPU MHA Host Pool initialization; no CI registration.

Run with python3 test/manual/ascend/test_npu_host_pool_layout.py.
The CPU tensors reproduce non-FIA all-layer and FIA per-layer buffer forms.
"""

import unittest

import torch

from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost


class TestNpuHostPoolLayout(unittest.TestCase):
    def _make_host(self, representation, layout):
        pool = MHATokenToKVPool(
            size=4,
            page_size=2,
            dtype=torch.float16,
            head_num=2,
            head_dim=4,
            layer_num=64,
            device="cpu",
            enable_memory_saver=False,
        )
        if representation != "generic_list":
            for side in ("k_buffer", "v_buffer"):
                backing = torch.stack(getattr(pool, side)).reshape(64, 3, 2, 2, 4)
                buffers = (
                    backing
                    if representation == "npu_tensor"
                    else [layer.reshape(-1, 1, 2, 4) for layer in backing]
                )
                setattr(pool, side, buffers)
        host = MHATokenToKVPoolHost(
            pool,
            host_to_device_ratio=1,
            host_size=0,
            page_size=2,
            layout=layout,
            pin_memory=False,
        )
        self.addCleanup(host.destroy)
        return host

    def test_tensor_and_list_device_buffers_initialize(self):
        for representation in ("generic_list", "npu_tensor", "fia_list"):
            with self.subTest(representation=representation):
                host = self._make_host(representation, "layer_first")
                self.assertTrue(host.device_rows_packed)
                self.assertEqual(len(host.k_data_refs), 64)

    def test_layer_views_write_the_same_layer_across_all_pages(self):
        for layout in ("page_first",):
            with self.subTest(layout=layout):
                host = self._make_host("generic_list", layout)
                host.k_buffer.zero_()
                host.v_buffer.zero_()
                host.k_data_refs[63].fill_(7)
                host.v_data_refs[63].fill_(11)
                torch.testing.assert_close(
                    host.k_buffer[:, 63], torch.full_like(host.k_buffer[:, 63], 7)
                )
                torch.testing.assert_close(
                    host.v_buffer[:, 63], torch.full_like(host.v_buffer[:, 63], 11)
                )
                self.assertEqual(host.k_buffer[:, :63].count_nonzero().item(), 0)
                self.assertEqual(host.v_buffer[:, :63].count_nonzero().item(), 0)


if __name__ == "__main__":
    unittest.main()
