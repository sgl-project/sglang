"""Batch filter and merge of watermark rows, as SamplingBatchInfo drives them."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

import unittest
from unittest.mock import MagicMock

import torch

from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.sampling.sampling_params import TOP_K_ALL
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


def _info(rows, batch_size):
    return SamplingBatchInfo(
        temperatures=torch.ones(batch_size, 1),
        top_ps=torch.ones(batch_size),
        top_ks=torch.full((batch_size,), TOP_K_ALL, dtype=torch.int32),
        min_ps=torch.zeros(batch_size),
        is_all_greedy=False,
        is_any_greedy=False,
        need_top_p_sampling=False,
        need_top_k_sampling=False,
        need_min_p_sampling=False,
        vocab_size=32,
        device="cpu",
        penalizer_orchestrator=MagicMock(is_required=False),
        watermark=rows,
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

    def test_sampling_batch_info_merge_and_filter(self):
        # Rows first: the other side's 3 rows come in off, window 1.
        info = _info(_rows([11, 12], [True, True]), 2)
        info.merge_batch(_info(None, 3))
        self.assertEqual(info.watermark.keys.tolist(), [11, 12, 0, 0, 0])
        self.assertEqual(info.watermark.context_windows.tolist(), [4, 4, 1, 1, 1])
        self.assertEqual(info.watermark.enabled.tolist(), [True, True] + [False] * 3)

        # No rows first: the 2 off rows are counted before the batch tensors grow.
        info = _info(None, 2)
        info.merge_batch(_info(_rows([11, 12, 13], [True, False, True]), 3))
        self.assertEqual(info.watermark.keys.tolist(), [0, 0, 11, 12, 13])
        self.assertEqual(
            info.watermark.candidates_host, [False, False, True, False, True]
        )

        info.filter_batch([3, 4], torch.tensor([3, 4]))
        self.assertEqual(info.watermark.keys.tolist(), [12, 13])
        info.filter_batch([0], torch.tensor([0]))
        self.assertIsNone(info.watermark)


if __name__ == "__main__":
    unittest.main()
