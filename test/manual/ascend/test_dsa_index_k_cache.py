"""Run on Ascend 950 with Triton Ascend:

python -m unittest discover -s test/manual/ascend -p test_dsa_index_k_cache.py -v
"""

import unittest

import torch
import torch_npu  # noqa: F401

from sglang.srt.hardware_backend.npu.attention.dsa_index_k_cache import (
    set_index_k_scale_buffer_npu,
)


@unittest.skipUnless(torch.npu.is_available(), "requires an Ascend NPU")
class TestPackedIndexCache(unittest.TestCase):
    def test_native_quantized_store(self):
        # Include first/last slots, page boundaries and shuffled physical pages.
        for page_size in (16, 64, 128, 1024):
            for loc_dtype in (torch.int32, torch.int64):
                for scale_column in (False, True):
                    with self.subTest(
                        page_size=page_size,
                        loc_dtype=loc_dtype,
                        scale_column=scale_column,
                    ):
                        self._check_store(page_size, loc_dtype, scale_column)

    def _check_store(self, page_size, loc_dtype, scale_column):
        torch.manual_seed(42)
        loc_cpu = torch.tensor(
            [3 * page_size - 1, page_size, 0, 2 * page_size + 1, page_size - 1],
            dtype=loc_dtype,
        )
        loc = loc_cpu.npu()
        expected = torch.full((4, page_size * 132), 0xA5, dtype=torch.uint8)
        # A nonzero aligned storage offset must also work.
        backing = torch.full(
            (expected.numel() + 8,), 0xA5, dtype=torch.uint8, device="npu"
        )
        cache = backing[4:-4].view(expected.shape)

        # Repeat with new values to verify overwrite as well as first insertion.
        for _ in range(2):
            x = torch.randn((loc.numel(), 128), dtype=torch.bfloat16).npu()
            key, scale = torch_npu.npu_dynamic_quant(x, dst_type=torch.float8_e4m3fn)
            scale = scale.to(torch.float32).reshape(-1)
            key_bytes = key.view(torch.uint8).cpu()
            scale_bytes = scale.cpu().view(torch.uint8).reshape(-1, 4)
            for token, location in enumerate(loc_cpu.tolist()):
                page, slot = divmod(location, page_size)
                expected[page, slot * 128 : (slot + 1) * 128] = key_bytes[token]
                start = page_size * 128 + slot * 4
                expected[page, start : start + 4] = scale_bytes[token]

            set_index_k_scale_buffer_npu(
                cache, loc, key, scale[:, None] if scale_column else scale, page_size
            )
            torch.npu.synchronize()
            # Full-buffer comparison also checks untouched slots/pages.
            self.assertTrue(torch.equal(cache.cpu(), expected))
            self.assertTrue(torch.all(backing[:4].cpu() == 0xA5).item())
            self.assertTrue(torch.all(backing[-4:].cpu() == 0xA5).item())

        set_index_k_scale_buffer_npu(cache, loc[:0], key[:0], scale[:0], page_size)
        self.assertTrue(torch.equal(cache.cpu(), expected))


if __name__ == "__main__":
    unittest.main()
