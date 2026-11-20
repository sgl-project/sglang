"""Batch filter and merge of watermark rows, as SamplingBatchInfo drives them."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

import unittest

import torch

from sglang.srt.sampling.watermarking.gumbel.adapter import GumbelRows
from sglang.srt.sampling.watermarking.interface import merge_watermark_rows
from sglang.test.test_utils import CustomTestCase


def _rows(keys, candidates):
    return GumbelRows(
        keys=torch.tensor(keys, dtype=torch.int64),
        context_windows=torch.full((len(keys),), 4, dtype=torch.int32),
        enabled=torch.tensor(candidates, dtype=torch.bool),
        candidates_host=list(candidates),
    )


class TestWatermarkRows(CustomTestCase):
    def test_filter_keeps_rows_or_drops_to_none(self):
        rows = _rows([11, 12, 13], [True, False, True])

        kept = rows.filter([2, 1], torch.tensor([2, 1]))
        self.assertEqual(kept.keys.tolist(), [13, 12])
        self.assertEqual(kept.candidates_host, [True, False])
        self.assertIsNone(rows.filter([1], torch.tensor([1])))
        # The overlapped forward may still read the old rows.
        self.assertEqual(rows.keys.tolist(), [11, 12, 13])

    def test_merge_aligns_a_side_without_rows(self):
        rows = _rows([11, 12], [True, True])

        self.assertIsNone(merge_watermark_rows(None, None, 2, 3))
        for merged, off in (
            (merge_watermark_rows(rows, None, 2, 3), slice(2, 5)),
            (merge_watermark_rows(None, rows, 3, 2), slice(0, 3)),
        ):
            self.assertEqual(len(merged.candidates_host), 5)
            self.assertEqual(merged.keys.shape[0], 5)
            self.assertFalse(merged.enabled[off].any())
            self.assertFalse(any(merged.candidates_host[off]))
        self.assertEqual(
            merge_watermark_rows(None, rows, 3, 2).keys[3:].tolist(), [11, 12]
        )
        self.assertEqual(rows.candidates_host, [True, True])


if __name__ == "__main__":
    unittest.main()
