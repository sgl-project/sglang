"""Lightweight MUSA QSA regressions that avoid CUDA-only test dependencies."""

import unittest

import torch

from sglang.srt.layers.attention.qsa.kernel import qsa_fast_topk
from sglang.srt.utils.device import is_musa
from sglang.test.ci.ci_register import register_musa_ci
from sglang.test.test_utils import CustomTestCase

register_musa_ci(est_time=10, suite="nightly-musa-1-gpu", nightly=True)


@unittest.skipIf(not is_musa(), "MUSA device not available")
class TestQsaMusaTopk(CustomTestCase):
    def test_empty_rows_match_cpu_reference(self):
        """The first and last empty rows must both retain their -1 padding."""
        # Replay the CPU-generated S5000 failure tensor, including its preceding
        # random blocks, instead of relying on MUSA RNG behavior.
        torch.manual_seed(20261125)
        for shape in (
            (4, 17),
            (3, 700),
            (2, 0),
            (0, 7),
            (5,),
            (13,),
            (0,),
            (1, 81),
            (1, 102),
        ):
            torch.randn(*shape)
        logits = torch.randn(5, 116)
        starts = torch.tensor([107, 111, 115, 110, 109], dtype=torch.int32)
        ends = torch.tensor([107, 115, 116, 115, 109], dtype=torch.int32)

        actual = qsa_fast_topk(
            logits.to("musa"), starts.to("musa"), ends.to("musa"), topk=1
        ).cpu()
        expected = torch.full((5, 1), -1, dtype=torch.int32)
        for row in range(logits.shape[0]):
            start, end = int(starts[row]), int(ends[row])
            if end > start:
                expected[row, 0] = torch.topk(logits[row, start:end], 1).indices
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
