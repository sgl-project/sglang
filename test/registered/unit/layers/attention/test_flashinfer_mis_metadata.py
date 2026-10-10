"""Regression coverage for FlashInfer multi-item scoring metadata bounds."""

import unittest
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.attention.flashinfer_backend import (  # noqa: E402
    FLASHINFER_MIS_MAX_ITEM_LEN,
    FlashInferAttnBackend,
    _max_multi_item_scoring_item_len,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestFlashInferMISMetadata(CustomTestCase):
    @staticmethod
    def _make_forward_batch(delimiter_indices, seq_len):
        return SimpleNamespace(
            forward_mode=None,
            multi_item_delimiter_indices=[torch.tensor(delimiter_indices)],
            extend_prefix_lens_cpu=None,
            extend_seq_lens_cpu=[seq_len],
            input_ids=torch.empty(seq_len, dtype=torch.int64),
            positions=torch.zeros(seq_len, dtype=torch.int64),
        )

    def test_max_item_len_tracks_largest_delimited_span(self):
        """The bound must cover the largest item, including a non-final item."""
        self.assertEqual(
            _max_multi_item_scoring_item_len(torch.tensor([4, 65005]), 65006),
            65000,
        )
        self.assertEqual(
            _max_multi_item_scoring_item_len(torch.tensor([4, 21, 70022]), 70023),
            70000,
        )

    def test_allows_uint16_boundary(self):
        """An item at the vendor metadata limit must remain representable."""
        backend = SimpleNamespace(enable_mis=True)
        forward_batch = self._make_forward_batch(
            [4, 4 + FLASHINFER_MIS_MAX_ITEM_LEN + 1],
            4 + FLASHINFER_MIS_MAX_ITEM_LEN + 2,
        )

        params = FlashInferAttnBackend._process_multi_item_scoring(
            backend, forward_batch
        )

        self.assertTrue(params.is_enabled())
        self.assertEqual(params.token_pos_in_items_ptr.dtype, torch.uint16)
        self.assertEqual(params.max_item_len_ptr.dtype, torch.uint16)
        self.assertEqual(params.max_item_len_ptr.item(), FLASHINFER_MIS_MAX_ITEM_LEN)

    def test_rejects_item_len_that_overflows_uint16_positions(self):
        """Reject silent uint16 wrap instead of scoring with corrupted positions."""
        backend = SimpleNamespace(enable_mis=True)
        forward_batch = self._make_forward_batch(
            [4, 4 + FLASHINFER_MIS_MAX_ITEM_LEN + 2],
            4 + FLASHINFER_MIS_MAX_ITEM_LEN + 3,
        )

        with self.assertRaisesRegex(ValueError, "uint16 item-position metadata"):
            FlashInferAttnBackend._process_multi_item_scoring(backend, forward_batch)


if __name__ == "__main__":
    unittest.main()
