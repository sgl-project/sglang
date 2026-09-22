import unittest
from types import SimpleNamespace

import torch

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.ltx_2.denoising import (
    LTX2DenoisingStage,
)
from sglang.test.test_utils import CustomTestCase


class TestLTX2SPPaddingMask(CustomTestCase):
    KEY = "sp_valid_len"

    def _build(self, valid, *, seq_len, batch_size=2):
        batch = SimpleNamespace(**({self.KEY: valid} if valid is not None else {}))
        return LTX2DenoisingStage._build_ltx2_sp_padding_mask(
            batch,
            seq_len=seq_len,
            batch_size=batch_size,
            key=self.KEY,
            device=torch.device("cpu"),
        )

    def test_missing_valid_returns_none(self):
        # Attribute absent on the batch -> no mask.
        self.assertIsNone(self._build(None, seq_len=8))

    def test_no_padding_returns_none(self):
        # valid == seq_len: an all-True mask would be a no-op, so return None
        # to keep the fused (unmasked) attention path.
        self.assertIsNone(self._build(8, seq_len=8))

    def test_valid_greater_than_seq_len_returns_none(self):
        # Defensive: valid > seq_len still means no padding.
        self.assertIsNone(self._build(12, seq_len=8))

    def test_padding_returns_real_mask(self):
        mask = self._build(5, seq_len=8, batch_size=2)
        self.assertIsNotNone(mask)
        self.assertEqual(mask.shape, (2, 8))
        self.assertEqual(mask.dtype, torch.bool)
        expected = torch.tensor([True] * 5 + [False] * 3, dtype=torch.bool)
        for row in mask:
            self.assertTrue(torch.equal(row, expected))

    def test_zero_valid_returns_all_false_mask(self):
        mask = self._build(0, seq_len=4, batch_size=1)
        self.assertIsNotNone(mask)
        self.assertEqual(mask.shape, (1, 4))
        self.assertFalse(mask.any().item())


if __name__ == "__main__":
    unittest.main()
