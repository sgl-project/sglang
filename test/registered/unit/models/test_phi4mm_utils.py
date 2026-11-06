"""CPU regression coverage for Phi4 streaming attention masks."""

import random
import unittest

import numpy as np
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from sglang.srt.models.phi4mm_utils import adaptive_enc_mask
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _expected_mask(seq_len, starts, left_window, right_window):
    starts = [int(start) for start in starts]
    left_boundaries = [0] + starts
    right_boundaries = starts + [seq_len]
    expected = torch.empty((seq_len, seq_len), dtype=torch.bool)
    for row in range(seq_len):
        chunk = next(
            index
            for index, (start, end) in enumerate(zip(left_boundaries, right_boundaries))
            if start <= row < end
        )
        left = left_boundaries[max(0, chunk - left_window)]
        right = right_boundaries[min(len(starts), chunk + right_window)]
        for column in range(seq_len):
            expected[row, column] = left <= column < right
    return expected


class _NoNonzero(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func == torch.ops.aten.nonzero.default:
            raise AssertionError("Chunk lookup enumerated the dense interval mask")
        return func(*args, **(kwargs or {}))


class TestAdaptiveEncMask(CustomTestCase):
    def assert_mask(self, seq_len, starts, left_window=0, right_window=0):
        expected = _expected_mask(seq_len, starts, left_window, right_window)
        result = adaptive_enc_mask(seq_len, starts, left_window, right_window)
        self.assertEqual(result.shape, (seq_len, seq_len))
        self.assertEqual(result.dtype, torch.bool)
        self.assertEqual(result.device.type, "cpu")
        torch.testing.assert_close(result, expected, rtol=0, atol=0)

    def test_chunk_boundaries_and_windows(self):
        for seq_len, starts in [
            (0, []),
            (0, [0]),
            (1, [0]),
            (8, [0, 3, 6]),
            (9, [0, 1, 4, 7]),
            (5, []),
            (7, [2, 5]),
            (8, [0, 0, 3, 3, 8]),
        ]:
            for left_window, right_window in [(0, 0), (1, 0), (0, 1), (2, 2), (20, 20)]:
                with self.subTest(
                    seq_len=seq_len,
                    starts=starts,
                    left_window=left_window,
                    right_window=right_window,
                ):
                    self.assert_mask(seq_len, starts, left_window, right_window)

    def test_sorted_adaptive_chunks(self):
        rng = random.Random(9889)
        for _ in range(50):
            seq_len = rng.randrange(16)
            starts = sorted(rng.randrange(seq_len + 1) for _ in range(rng.randrange(8)))
            left_window = rng.randrange(5)
            right_window = rng.randrange(5)
            with self.subTest(
                seq_len=seq_len,
                starts=starts,
                left_window=left_window,
                right_window=right_window,
            ):
                self.assert_mask(seq_len, starts, left_window, right_window)

    def test_input_representations_are_preserved(self):
        starts = [0, 3, 6]
        strided = torch.tensor([0, -1, 3, -1, 6, -1], dtype=torch.int64)[::2]
        self.assertFalse(strided.is_contiguous())
        inputs = [
            starts,
            np.array(starts, dtype=np.int64),
            torch.tensor(starts, dtype=torch.int32),
            torch.tensor(starts, dtype=torch.int64),
            strided,
        ]
        for boundaries in inputs:
            with self.subTest(input_type=type(boundaries)):
                original = torch.as_tensor(boundaries).clone()
                self.assert_mask(8, boundaries, left_window=1, right_window=1)
                torch.testing.assert_close(torch.as_tensor(boundaries), original)

    def test_chunk_lookup_does_not_enumerate_intervals(self):
        with _NoNonzero():
            result = adaptive_enc_mask(8, [0, 3, 6], left_window=1)
        torch.testing.assert_close(result, _expected_mask(8, [0, 3, 6], 1, 0))


if __name__ == "__main__":
    unittest.main()
