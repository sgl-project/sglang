import unittest

import torch
from sglang.srt.layers.engram import _engram_gate_torch, engram_gate


@unittest.skipUnless(
    hasattr(torch, "npu") and torch.npu.is_available(), "requires Ascend"
)
class TestEngramGateWorkspace(unittest.TestCase):
    def test_row_boundaries_and_mask(self):
        for rows in (511, 512, 513, 2048):
            x = torch.randn(rows, 4, 64, device="npu", dtype=torch.bfloat16)
            kv = torch.randn(rows, 320, device="npu", dtype=x.dtype)
            q = torch.randn_like(x[0])
            k = torch.randn_like(q)
            ids = torch.arange(rows, device=x.device) % 7
            args = x, kv, q, k, 1e-6, 1e-6, ids, 0
            actual = engram_gate(*args)
            torch.testing.assert_close(
                actual, _engram_gate_torch(*args), rtol=0, atol=0
            )
            torch.testing.assert_close(actual[ids == 0], x[ids == 0], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
