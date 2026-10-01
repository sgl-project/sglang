"""Tests for FlashInfer multi-item scoring metadata and position updates.

The builder runs on the host, so these tests use CPU tensors and require no
model or GPU compute. Importing the backend module still requires the CUDA test
environment.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")


def _build(
    delimiters,
    total_tokens,
    *,
    seq_lens=None,
    prefix_lens=None,
    mode=ForwardMode.EXTEND,
):
    """Construct a minimal forward_batch stub for the builder."""
    fb = SimpleNamespace(
        forward_mode=mode,
        multi_item_delimiter_indices=delimiters,
        input_ids=torch.zeros(total_tokens, dtype=torch.int64),
        positions=torch.zeros(total_tokens, dtype=torch.int64),
    )
    if seq_lens is not None:
        fb.extend_seq_lens_cpu = seq_lens
    if prefix_lens is not None:
        fb.extend_prefix_lens_cpu = prefix_lens
    return fb


def _run(fb, enable_mis=True):
    backend = SimpleNamespace(enable_mis=enable_mis)
    return FlashInferAttnBackend._process_multi_item_scoring(backend, fb)


class TestMISMetadataBuild(unittest.TestCase):
    def test_single_sequence(self):
        # Docstring Case 1: query of 7 tokens then 3 single-token items, each
        # preceded by a delimiter at indices 7, 9, 11, 13 (seq len 14).
        fb = _build([torch.tensor([7, 9, 11, 13])], 14)
        p = _run(fb)

        self.assertTrue(p.is_enabled())
        self.assertEqual(p.prefix_len_ptr.tolist(), [7])
        self.assertEqual(p.prefix_len_ptr.dtype, torch.uint32)
        self.assertEqual(p.token_pos_in_items_ptr.tolist(), [0, 1, 0, 1, 0, 1, 0])
        self.assertEqual(p.token_pos_in_items_ptr.dtype, torch.uint16)
        self.assertEqual(p.token_pos_in_items_len, 7)
        self.assertEqual(p.max_item_len_ptr.tolist(), [1])
        self.assertEqual(p.max_item_len_ptr.dtype, torch.uint16)
        # positions: prefix region untouched, suffix gets prefix_len+pos-1.
        self.assertEqual(
            fb.positions.tolist(),
            [0, 0, 0, 0, 0, 0, 0, 6, 7, 6, 7, 6, 7, 6],
        )

    def test_multi_token_items_max_item_len(self):
        # Items of length 3 (delimiters spaced by 3): pos within item reaches 2.
        fb = _build([torch.tensor([4, 7, 10])], 13)
        p = _run(fb)

        self.assertEqual(p.prefix_len_ptr.tolist(), [4])
        self.assertEqual(p.token_pos_in_items_ptr.tolist(), [0, 1, 2, 0, 1, 2, 0, 1, 2])
        self.assertEqual(p.max_item_len_ptr.tolist(), [2])
        self.assertEqual(
            fb.positions.tolist(),
            [0, 0, 0, 0, 3, 4, 5, 3, 4, 5, 3, 4, 5],
        )

    def test_batch_padding_and_per_sequence_max(self):
        # Two sequences of different suffix lengths -> token_pos is padded to the
        # longer one, and max_item_len is per sequence.
        delims = [torch.tensor([7, 9, 11, 13]), torch.tensor([3, 5, 7])]
        fb = _build(delims, 22, seq_lens=[14, 8])
        p = _run(fb)

        self.assertEqual(p.prefix_len_ptr.tolist(), [7, 3])
        self.assertEqual(p.token_pos_in_items_len, 7)
        # seq1 (len 7) then seq2 (len 5, padded with two zeros to 7).
        self.assertEqual(
            p.token_pos_in_items_ptr.tolist(),
            [0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0],
        )
        self.assertEqual(p.max_item_len_ptr.tolist(), [1, 1])
        self.assertEqual(
            fb.positions.tolist(),
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                6,
                7,
                6,
                7,
                6,
                7,
                6,  # seq1
                0,
                0,
                0,
                2,
                3,
                2,
                3,
                2,
            ],  # seq2 (offset by 14)
        )

    def test_prefix_cache_lens_offsets_prefix(self):
        # A cached prefix shifts prefix_len and the written positions.
        fb = _build([torch.tensor([7, 9, 11, 13])], 14, seq_lens=[14], prefix_lens=[5])
        p = _run(fb)

        self.assertEqual(p.prefix_len_ptr.tolist(), [12])  # 7 + 5
        self.assertEqual(p.token_pos_in_items_ptr.tolist(), [0, 1, 0, 1, 0, 1, 0])
        self.assertEqual(
            fb.positions.tolist(),
            [0, 0, 0, 0, 0, 0, 0, 11, 12, 11, 12, 11, 12, 11],
        )

    def test_disabled_returns_empty(self):
        fb = _build([torch.tensor([7, 9, 11, 13])], 14)
        p = _run(fb, enable_mis=False)
        self.assertFalse(p.is_enabled())
        self.assertIsNone(p.prefix_len_ptr)

    def test_decode_mode_returns_empty(self):
        fb = _build([torch.tensor([7, 9, 11, 13])], 14, mode=ForwardMode.DECODE)
        self.assertFalse(_run(fb).is_enabled())

    def test_none_delimiters_returns_empty(self):
        fb = _build(None, 14)
        self.assertFalse(_run(fb).is_enabled())

    def test_empty_delimiters_skipped(self):
        # A sequence with no delimiters is skipped; with only such sequences the
        # builder returns empty params.
        fb = _build([torch.tensor([], dtype=torch.int64)], 14)
        self.assertFalse(_run(fb).is_enabled())


if __name__ == "__main__":
    unittest.main()
