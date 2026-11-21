import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.attention.flashinfer_prefill_kv_dequant import (
    FlashInferPrefillKVDequant,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestPrefillKVDequantPolicy(CustomTestCase):
    def setUp(self):
        self.capture = patch(
            "torch.cuda.is_current_stream_capturing", return_value=False
        )
        self.capture.start()
        self.addCleanup(self.capture.stop)
        self.twin = Mock()
        self.factory = Mock(return_value=self.twin)
        self.helper = FlashInferPrefillKVDequant(self.factory, 4, 8, 32)
        self.kwargs = dict(
            qo_indptr=torch.tensor([0, 4], dtype=torch.int32),
            paged_kv_indptr=torch.tensor([0, 8], dtype=torch.int32),
            paged_kv_indices=torch.tensor(
                [7, 3, 1, 5, 2, 0, 6, 4, 999], dtype=torch.int32
            ),
            paged_kv_last_page_len=torch.tensor([1], dtype=torch.int32),
            num_qo_heads=6,
            num_kv_heads=1,
            head_dim_qk=256,
            page_size=1,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.float8_e4m3fn,
            fixed_split_size=4096,
        )

    def prepare(self, **overrides):
        args = dict(
            num_new_tokens=4,
            num_pages=8,
            use_ragged=False,
            spec_info=None,
            original_wrapper=SimpleNamespace(is_cuda_graph_enabled=False),
            plan_kwargs=self.kwargs,
        )
        args.update(overrides)
        return self.helper.prepare(**args)

    def test_remaps_only_active_pages_and_preserves_plan_options(self):
        plan = self.prepare()
        self.assertTrue(torch.equal(plan.pages, self.kwargs["paged_kv_indices"][:8]))
        actual = self.twin.begin_forward.call_args.kwargs
        self.assertTrue(
            torch.equal(actual["paged_kv_indices"], torch.arange(8, dtype=torch.int32))
        )
        self.assertEqual(actual["kv_data_type"], torch.bfloat16)
        for key, value in self.kwargs.items():
            if key not in ["paged_kv_indices", "kv_data_type"]:
                self.assertIs(actual[key], value)
        self.assertEqual(self.kwargs["kv_data_type"], torch.float8_e4m3fn)

    def test_fallback_does_not_reuse_the_previous_plan(self):
        self.assertIsNotNone(self.prepare())
        for overrides in [
            dict(num_new_tokens=None),
            dict(num_new_tokens=3),
            dict(num_pages=7),
            dict(num_pages=33),
            dict(use_ragged=True),
            dict(spec_info=object()),
            dict(original_wrapper=SimpleNamespace(is_cuda_graph_enabled=True)),
        ]:
            with self.subTest(overrides=overrides):
                self.assertIsNone(self.prepare(**overrides))
        self.prepare()
        self.factory.assert_called_once()

    def test_custom_masks_dtypes_and_capture_fall_back(self):
        for key, value in [
            ("custom_mask", torch.ones(4)),
            ("packed_custom_mask", torch.ones(1)),
            ("prefix_len_ptr", torch.ones(1)),
            ("window_left", 16),
            ("kv_data_type", torch.bfloat16),
            ("q_data_type", torch.float16),
        ]:
            with self.subTest(key=key):
                kwargs = dict(self.kwargs, **{key: value})
                self.assertIsNone(self.prepare(plan_kwargs=kwargs))
        with patch("torch.cuda.is_current_stream_capturing", return_value=True):
            self.assertIsNone(self.prepare())
        self.factory.assert_not_called()

    def test_thresholds_and_host_page_count_are_validated(self):
        for thresholds in [(0, 8, 32), (4, -1, 32), (4, 32, 8)]:
            with self.subTest(thresholds=thresholds), self.assertRaises(ValueError):
                FlashInferPrefillKVDequant(self.factory, *thresholds)
        with self.assertRaisesRegex(ValueError, "page-index buffer"):
            self.prepare(num_pages=10)


if __name__ == "__main__":
    unittest.main()
