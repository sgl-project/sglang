"""Unit tests for FlashInfer MIS uint16 item-position metadata limits."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.flashinfer_backend import (
    FlashInferAttnBackend,
    _mis_max_item_position,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_forward_batch(seq_len: int, delimiter_indices):
    return SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        multi_item_delimiter_indices=[
            torch.tensor(delimiter_indices, dtype=torch.int64)
        ],
        extend_prefix_lens_cpu=None,
        extend_seq_lens_cpu=[seq_len],
        input_ids=torch.arange(seq_len, dtype=torch.int64),
        positions=torch.arange(seq_len, dtype=torch.int64),
    )


def _make_backend():
    backend = FlashInferAttnBackend.__new__(FlashInferAttnBackend)
    backend.enable_mis = True
    return backend


class TestFlashInferMISUint16Limit(CustomTestCase):
    def test_below_boundary_succeeds(self):
        params = _make_backend()._process_multi_item_scoring(
            _make_forward_batch(seq_len=65006, delimiter_indices=[4, 65005])
        )

        self.assertTrue(params.is_enabled())
        self.assertEqual(
            params.max_item_len_ptr.to(torch.int32).max().item(),
            65000,
        )

    def test_at_boundary_succeeds(self):
        params = _make_backend()._process_multi_item_scoring(
            _make_forward_batch(seq_len=65541, delimiter_indices=[4, 65540])
        )

        self.assertTrue(params.is_enabled())
        self.assertEqual(
            params.max_item_len_ptr.to(torch.int32).max().item(),
            65535,
        )

    def test_above_boundary_raises(self):
        with self.assertRaisesRegex(ValueError, "uint16 metadata limit"):
            _make_backend()._process_multi_item_scoring(
                _make_forward_batch(seq_len=70006, delimiter_indices=[4, 70005])
            )

    def test_multi_item_total_suffix_large_but_items_ok(self):
        params = _make_backend()._process_multi_item_scoring(
            _make_forward_batch(
                seq_len=70015,
                delimiter_indices=[
                    4,
                    7005,
                    14006,
                    21007,
                    28008,
                    35009,
                    42010,
                    49011,
                    56012,
                    63013,
                    70014,
                ],
            )
        )

        self.assertTrue(params.is_enabled())
        self.assertEqual(
            params.max_item_len_ptr.to(torch.int32).max().item(),
            7000,
        )

    def test_multi_item_with_one_long_item_raises(self):
        with self.assertRaisesRegex(ValueError, "uint16 metadata limit"):
            _make_backend()._process_multi_item_scoring(
                _make_forward_batch(
                    seq_len=70040, delimiter_indices=[4, 21, 70022, 70039]
                )
            )

    def test_helper_single_long_item(self):
        self.assertEqual(
            _mis_max_item_position(torch.tensor([4, 70005], dtype=torch.int64), 70006),
            70000,
        )

    def test_helper_multi_item_short_segments(self):
        self.assertEqual(
            _mis_max_item_position(
                torch.tensor([4, 21, 7022, 7039], dtype=torch.int64), 7040
            ),
            7000,
        )


if __name__ == "__main__":
    unittest.main()
