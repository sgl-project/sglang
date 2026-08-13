import unittest

import torch

from sglang.multimodal_gen.runtime.models.dits.ltx_2 import _ltx2_modulate


class TestLtx2ModulateMount(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_row_broadcast_matches_eager(self):
        torch.manual_seed(0)
        x = torch.randn(2, 517, 4096, device="cuda", dtype=torch.bfloat16)
        scale = torch.randn(2, 1, 4096, device="cuda", dtype=torch.bfloat16)
        shift = torch.randn(2, 1, 4096, device="cuda", dtype=torch.bfloat16)
        reference = x * (1 + scale) + shift
        self.assertTrue(torch.equal(reference, _ltx2_modulate(x, scale, shift)))

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_non_contiguous_rows_match_eager(self):
        # unbind()/squeeze() of the combined adaLN tables produces strided
        # (B, 1, D) views; the helper densifies them before the kernel.
        torch.manual_seed(1)
        x = torch.randn(2, 33, 512, device="cuda", dtype=torch.bfloat16)
        table = torch.randn(2, 1, 4, 512, device="cuda", dtype=torch.bfloat16)
        scale, shift = table.unbind(dim=2)[:2]
        reference = x * (1 + scale) + shift
        self.assertTrue(torch.equal(reference, _ltx2_modulate(x, scale, shift)))

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_per_token_rows_fall_back(self):
        torch.manual_seed(2)
        x = torch.randn(2, 16, 128, device="cuda", dtype=torch.bfloat16)
        scale = torch.randn(2, 16, 128, device="cuda", dtype=torch.bfloat16)
        shift = torch.randn(2, 16, 128, device="cuda", dtype=torch.bfloat16)
        reference = x * (1 + scale) + shift
        self.assertTrue(torch.equal(reference, _ltx2_modulate(x, scale, shift)))

    def test_cpu_falls_back(self):
        torch.manual_seed(3)
        x = torch.randn(1, 9, 64, dtype=torch.float32)
        scale = torch.randn(1, 1, 64, dtype=torch.float32)
        shift = torch.randn(1, 1, 64, dtype=torch.float32)
        reference = x * (1 + scale) + shift
        self.assertTrue(torch.equal(reference, _ltx2_modulate(x, scale, shift)))


if __name__ == "__main__":
    unittest.main()
