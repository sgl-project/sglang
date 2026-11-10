import itertools
import unittest

import torch

from sglang.kernels.ops.sampling.softcap_row_argmax import softcap_row_argmax
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-small")

DTYPES = (torch.float32, torch.float16, torch.bfloat16)


class TestSoftcapRowArgmax(CustomTestCase):
    def _check(self, logits, softcap, valid_width, col_offset):
        reference = logits[:, :valid_width].float()
        if softcap is not None:
            reference = torch.tanh(reference / softcap) * softcap
        values, indices = reference.max(dim=1)

        actual = softcap_row_argmax(logits, softcap, valid_width, col_offset)

        self.assertEqual(actual.shape, (logits.shape[0], 2))
        self.assertEqual(actual.dtype, torch.float32)
        self.assertEqual(actual.device, logits.device)
        torch.testing.assert_close(actual[:, 0], values, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(
            actual[:, 1], (indices + col_offset).float(), rtol=0, atol=0
        )
        return actual

    def test_negative_logits_with_padded_rows_and_partial_blocks(self):
        """Neither row padding nor zero-filled tail lanes may win the reduction.

        A nonzero storage offset and padded row stride also make width-based row
        addressing observably wrong on every row after the first.
        """
        for dtype, width, softcap in itertools.product(
            DTYPES, (4095, 4096, 4097), (None, 1.0, 30.0)
        ):
            with self.subTest(dtype=dtype, width=width, softcap=softcap):
                storage = torch.full((4, width + 17), 100.0, dtype=dtype, device="cuda")
                logits = storage[:, 1 : width + 6]
                logits[:, :width] = -4.0
                winners = (0, width // 2, width - 1, min(4096, width - 1))
                for row, column in enumerate(winners):
                    logits[row, column] = -(row + 1) / 4
                before = storage.clone()

                self._check(logits, softcap, width, col_offset=12345)

                torch.testing.assert_close(storage, before, rtol=0, atol=0)

    def test_first_tie_wins_within_and_across_blocks(self):
        """A later equal block maximum must not replace the first token ID."""
        width = 8193
        for dtype, softcap in itertools.product(DTYPES, (None, 30.0)):
            with self.subTest(dtype=dtype, softcap=softcap):
                logits = torch.full((3, width), -4.0, dtype=dtype, device="cuda")
                for row, columns in enumerate(((5, 7), (4095, 4096), (4096, 8192))):
                    logits[row, list(columns)] = 2.0
                # Token IDs are carried in fp32, including near its exact-integer
                # limit. These offsets obey the operator's < 2**24 contract.
                self._check(logits, softcap, width, col_offset=2**24 - width - 1)

    def test_softcap_saturation_changes_the_first_argmax(self):
        """Argmax before softcapping is not equivalent after fp32 saturation.

        The larger raw logit is in a later block, but tanh rounds both positive
        logits to one, so the earlier token must win the softcapped tie.
        """
        for dtype in DTYPES:
            with self.subTest(dtype=dtype):
                logits = torch.full((1, 4097), -4.0, dtype=dtype, device="cuda")
                logits[0, 3] = 100.0
                logits[0, 4096] = 200.0

                raw = self._check(logits, None, 4097, col_offset=17)
                capped = self._check(logits, 1.0, 4097, col_offset=17)

                self.assertEqual(raw[0, 1].item(), 4096 + 17)
                self.assertEqual(capped[0, 1].item(), 3 + 17)


if __name__ == "__main__":
    unittest.main()
