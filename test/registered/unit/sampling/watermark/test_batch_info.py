import unittest
from array import array
from types import SimpleNamespace

import torch

from sglang.srt.managers.overlap_utils import resolve_forward_inputs
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.sampling.watermark import (
    TextSealConfig,
    WatermarkRegistry,
    WatermarkRequestConfig,
    build_watermark_batch_info,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _request(rid, prompt, output, watermarked=False):
    return SimpleNamespace(
        rid=rid,
        origin_input_ids=array("q", prompt),
        output_ids=array("q", output),
        watermark=(
            WatermarkRequestConfig(provider="textseal") if watermarked else None
        ),
    )


class TestWatermarkBatchInfo(CustomTestCase):
    def setUp(self):
        self.registry = WatermarkRegistry(
            textseal=TextSealConfig(key_a=11, key_b=12, ngram=2)
        )

    def test_ordinary_batch_has_no_attachment(self):
        requests = [_request("ordinary", [1, 2], [])]
        self.assertIsNone(
            build_watermark_batch_info(requests, self.registry, device="cpu")
        )

    def test_constructs_prompt_and_generated_contexts(self):
        requests = [
            _request("a", [1], [2, 3], True),
            _request("b", [4], []),
            _request("c", [5, 6], [7], True),
        ]
        info = build_watermark_batch_info(requests, self.registry, device="cpu")

        self.assertEqual(info.enabled.tolist(), [True, False, True])
        self.assertFalse(info.all_enabled)
        self.assertEqual(info.contexts.tolist(), [[2, 3], [0, 0], [6, 7]])
        self.assertEqual(info.ngrams.tolist(), [2, 1, 2])
        self.assertEqual(info.key_a.tolist(), [11, 0, 11])

    def test_filter_reorder_and_merge_preserve_alignment(self):
        first = build_watermark_batch_info(
            [
                _request("a", [1, 2], [], True),
                _request("b", [3, 4], []),
            ],
            self.registry,
            device="cpu",
        )
        filtered = first.filter(torch.tensor([1, 0]))
        self.assertEqual(filtered.enabled.tolist(), [False, True])
        self.assertEqual(filtered.contexts.tolist(), [[0, 0], [1, 2]])

        second = build_watermark_batch_info(
            [_request("c", [5], [6, 7], True)],
            self.registry,
            device="cpu",
        )
        self.assertTrue(second.all_enabled)
        merged = filtered.merge(second)
        self.assertFalse(merged.all_enabled)
        self.assertEqual(merged.enabled.tolist(), [False, True, True])
        self.assertEqual(
            merged.contexts.tolist(),
            [[0, 0], [1, 2], [6, 7]],
        )

        requests = [
            _request("b", [3, 4], [8]),
            _request("a", [1, 2], [9], True),
            _request("c", [5], [6, 7, 10], True),
        ]
        refreshed = merged.refresh_contexts(requests)
        self.assertEqual(
            refreshed.contexts.tolist(),
            [[0, 0], [2, 9], [7, 10]],
        )

    def test_merge_preserves_unwatermarked_rows_in_both_orders(self):
        """Mixed batch merges must retain one watermark row per request."""
        marked = build_watermark_batch_info(
            [_request("marked", [1, 2], [], True)],
            self.registry,
            device="cpu",
        )

        ordinary_first = SimpleNamespace(watermark=None)
        SamplingBatchInfo.adjusted_merge_batch(
            ordinary_first,
            SimpleNamespace(watermark=marked),
            self_len=2,
            other_len=1,
        )
        self.assertEqual(
            ordinary_first.watermark.enabled.tolist(), [False, False, True]
        )
        self.assertEqual(
            ordinary_first.watermark.contexts.tolist(), [[0, 0], [0, 0], [1, 2]]
        )

        marked_first = SimpleNamespace(watermark=marked)
        SamplingBatchInfo.adjusted_merge_batch(
            marked_first,
            SimpleNamespace(watermark=None),
            self_len=1,
            other_len=2,
        )
        self.assertEqual(marked_first.watermark.enabled.tolist(), [True, False, False])
        self.assertEqual(
            marked_first.watermark.contexts.tolist(), [[1, 2], [0, 0], [0, 0]]
        )

    def test_overlap_relay_advances_context_with_current_decode_token(self):
        """Delayed request output IDs must not leave watermark context one token behind."""
        requests = [
            _request("a", [1], [2, 3], True),
            _request("b", [5], [6, 7], True),
        ]
        sampling_info = SimpleNamespace(
            watermark=build_watermark_batch_info(requests, self.registry, device="cpu")
        )
        no_spec = SimpleNamespace(is_none=lambda: True)
        batch = SimpleNamespace(
            prefill_input_ids_cpu=None,
            input_ids=None,
            req_pool_indices=torch.tensor([1, 3]),
            sampling_info=sampling_info,
            enable_overlap=False,
            spec_algorithm=no_spec,
        )
        future_map = SimpleNamespace(
            output_tokens_buf=torch.tensor([0, 8, 0, 9]), spec_algo=no_spec
        )

        resolve_forward_inputs(batch, future_map)

        self.assertEqual(batch.input_ids.tolist(), [8, 9])
        self.assertEqual(
            batch.sampling_info.watermark.contexts.tolist(),
            [[3, 8], [7, 9]],
        )


if __name__ == "__main__":
    unittest.main()
