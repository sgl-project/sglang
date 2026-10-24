import unittest
from types import SimpleNamespace

import torch

from sglang.multimodal_gen.runtime.models.dits.ltx_2 import LTX2TransformerBlock


def _reference_ada_values(table, batch_size, timestep, indices):
    num = int(table.shape[0])
    values = (
        table[indices].unsqueeze(0).unsqueeze(0).to(dtype=timestep.dtype)
        + timestep.reshape(batch_size, timestep.shape[1], num, -1)[:, :, indices, :]
    ).unbind(dim=2)
    return [t.squeeze(2) for t in values]


class TestLTX2AdaValuesLayout(unittest.TestCase):
    def test_values_are_contiguous_and_bitwise_equal(self):
        torch.manual_seed(0)
        for table_dtype, tokens, indices in (
            (torch.float32, 1, slice(0, 3)),
            (torch.bfloat16, 1, slice(None)),
            (torch.float32, 5, slice(6, 9)),
        ):
            table = torch.randn(9, 16, dtype=table_dtype)
            timestep = torch.randn(2, tokens, 9 * 16, dtype=torch.bfloat16)
            actual = LTX2TransformerBlock.get_ada_values(
                SimpleNamespace(), table, 2, timestep, indices
            )
            expected = _reference_ada_values(table, 2, timestep, indices)
            self.assertEqual(len(actual), len(expected))
            for a, e in zip(actual, expected):
                self.assertTrue(a.is_contiguous())
                self.assertTrue(torch.equal(a, e))

    def test_zero_one_mask_commutes_with_the_gate(self):
        torch.manual_seed(0)
        x = torch.randn(2, 6, 16, dtype=torch.bfloat16)
        attn = torch.randn(2, 6, 16, dtype=torch.bfloat16)
        gate = torch.randn(2, 1, 16, dtype=torch.bfloat16)
        mask = torch.tensor([1.0, 0.0], dtype=torch.bfloat16).view(2, 1, 1)
        self.assertTrue(torch.equal(x + (attn * mask) * gate, x + attn * (gate * mask)))


if __name__ == "__main__":
    unittest.main()
