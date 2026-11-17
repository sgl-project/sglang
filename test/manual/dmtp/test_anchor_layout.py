"""T1-residual numerical contract: no single anchor or other injection modes."""

import unittest

import torch

from sglang.srt.speculative.mtp_prefix_contract import MTPPrefixConditioner


class ResidualTest(unittest.TestCase):
    def test_only_t1_changes_and_inputs_do_not_alias(self):
        for dtype in (torch.float32, torch.bfloat16):
            module = MTPPrefixConditioner(32, 8, 1e-6).to(dtype)
            for bs in (1, 2, 8, 31, 32):
                blocks = torch.randn(bs, 8, 32, dtype=dtype)
                hidden = torch.randn(bs, 32, dtype=dtype)
                before = blocks.clone()
                result = module(blocks, hidden)
                x = hidden.float()
                normalized = (
                    x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
                ).to(dtype)
                expected = blocks[:, 1] + module.up(
                    torch.nn.functional.silu(module.down(normalized))
                )
                torch.testing.assert_close(result[:, 1], expected, rtol=0, atol=0)
                self.assertTrue(torch.equal(result[:, 0], before[:, 0]))
                self.assertTrue(torch.equal(result[:, 2:], before[:, 2:]))
                self.assertTrue(torch.equal(blocks, before))

    def test_seven_slot_input_rejected(self):
        with self.assertRaises(ValueError):
            MTPPrefixConditioner(32, 8, 1e-6)(torch.zeros(1, 7, 32), torch.zeros(1, 32))


if __name__ == "__main__":
    unittest.main()
