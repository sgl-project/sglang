import unittest

import torch

from sglang.multimodal_gen.runtime.models.dits.wanvideo import (
    _eager_temb_table_slices,
    _wan_temb_table_slices,
)


class TestWanTembTableSlices(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_fused_matches_eager_and_is_contiguous(self):
        torch.manual_seed(0)
        for batch, seq, hidden in [(1, 517, 3072), (2, 64, 1536)]:
            temb = torch.randn(
                batch, seq, 6, hidden, device="cuda", dtype=torch.bfloat16
            )
            table = torch.randn(1, 6, hidden, device="cuda", dtype=torch.float32)
            reference = _eager_temb_table_slices(table, temb)
            fused = _wan_temb_table_slices(table, temb)
            self.assertEqual(len(fused), 6)
            for ref, out in zip(reference, fused):
                self.assertTrue(torch.equal(ref, out))
                self.assertTrue(out.is_contiguous())

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_fp32_temb_also_supported(self):
        torch.manual_seed(1)
        temb = torch.randn(1, 33, 6, 512, device="cuda", dtype=torch.float32)
        table = torch.randn(1, 6, 512, device="cuda", dtype=torch.float32)
        reference = _eager_temb_table_slices(table, temb)
        fused = _wan_temb_table_slices(table, temb)
        for ref, out in zip(reference, fused):
            self.assertTrue(torch.equal(ref, out))

    def test_cpu_falls_back_to_eager(self):
        torch.manual_seed(2)
        temb = torch.randn(1, 9, 6, 64, dtype=torch.bfloat16)
        table = torch.randn(1, 6, 64, dtype=torch.float32)
        reference = _eager_temb_table_slices(table, temb)
        out = _wan_temb_table_slices(table, temb)
        for ref, got in zip(reference, out):
            self.assertTrue(torch.equal(ref, got))


if __name__ == "__main__":
    unittest.main()
