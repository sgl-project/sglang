"""Tests for FlashInfer multi-item scoring metadata and position updates.

The builder runs on the host, so these tests use CPU tensors and require no
model or GPU compute. Importing the backend module still requires the CUDA test
environment.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=15, stage="base-b", runner_config="1-gpu-small-amd")


def _build(
    delimiters,
    total_tokens,
    *,
    seq_lens=None,
    prefix_lens=None,
    mode=ForwardMode.EXTEND,
    mrope_positions=None,
    mm_inputs=None,
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
    if mrope_positions is not None:
        fb.mrope_positions = mrope_positions
    if mm_inputs is not None:
        fb.mm_inputs = mm_inputs
    return fb


def _run(fb, enable_mis=True):
    backend = SimpleNamespace(enable_mis=enable_mis)
    return FlashInferAttnBackend._process_multi_item_scoring(backend, fb)


class TestMISMetadataBuild(CustomTestCase):
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

    def test_empty_prefix_clamps_delimiter_positions_to_zero(self):
        # On BOS-less models with an empty query (first_delim == 0), delimiter
        # positions must clamp at 0 rather than -1 so RoPE does not read OOB.
        fb = _build([torch.tensor([0, 3, 6])], 7, seq_lens=[7])
        p = _run(fb)

        self.assertEqual(p.prefix_len_ptr.tolist(), [0])
        self.assertEqual(p.token_pos_in_items_ptr.tolist(), [0, 1, 2, 0, 1, 2, 0])
        self.assertEqual(p.max_item_len_ptr.tolist(), [2])
        self.assertEqual(fb.positions.tolist(), [0, 0, 1, 0, 0, 1, 0])
        self.assertTrue((fb.positions >= 0).all())

    def test_empty_delimiters_skipped(self):
        # A sequence with no delimiters is skipped; with only such sequences the
        # builder returns empty params.
        fb = _build([torch.tensor([], dtype=torch.int64)], 14)
        self.assertFalse(_run(fb).is_enabled())

    def test_mrope_positions_keep_second_item_invariant_when_first_item_lengthens(
        self,
    ):
        short_mrope = torch.arange(9, dtype=torch.int64).unsqueeze(0).repeat(3, 1)
        long_mrope = torch.arange(13, dtype=torch.int64).unsqueeze(0).repeat(3, 1)
        short_fb = _build(
            [torch.tensor([3, 5, 7], dtype=torch.int64)],
            9,
            mrope_positions=short_mrope,
        )
        short_fb.positions = torch.arange(9, dtype=torch.int64)
        long_fb = _build(
            [torch.tensor([3, 9, 11], dtype=torch.int64)],
            13,
            mrope_positions=long_mrope,
        )
        long_fb.positions = torch.arange(13, dtype=torch.int64)

        self.assertTrue(_run(short_fb).is_enabled())
        self.assertTrue(_run(long_fb).is_enabled())
        torch.testing.assert_close(
            short_fb.mrope_positions,
            short_fb.positions.unsqueeze(0).expand(3, -1),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            long_fb.mrope_positions,
            long_fb.positions.unsqueeze(0).expand(3, -1),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            long_fb.mrope_positions[:, -2:],
            short_fb.mrope_positions[:, -2:],
            rtol=0,
            atol=0,
        )

    def test_mrope_positions_stride0_expand_is_materialized_before_write(self):
        fb = _build(
            [torch.tensor([4, 7, 10], dtype=torch.int64)],
            13,
            mrope_positions=torch.arange(13, dtype=torch.int64)
            .unsqueeze(0)
            .expand(3, -1),
        )
        fb.positions = torch.arange(13, dtype=torch.int64)
        self.assertTrue(_run(fb).is_enabled())
        self.assertNotEqual(fb.mrope_positions.stride(0), 0)
        torch.testing.assert_close(
            fb.mrope_positions,
            fb.positions.unsqueeze(0).expand(3, -1),
            rtol=0,
            atol=0,
        )

    def test_dp_padded_extend_seq_lens_with_trailing_zero_is_ignored(self):
        fb = _build(
            [torch.tensor([3, 5, 7], dtype=torch.int64)],
            9,
            seq_lens=[9, 0],
            mrope_positions=torch.arange(9, dtype=torch.int64)
            .unsqueeze(0)
            .repeat(3, 1),
        )
        p = _run(fb)
        self.assertTrue(p.is_enabled())
        self.assertEqual(p.prefix_len_ptr.tolist(), [3])
        torch.testing.assert_close(
            fb.mrope_positions[:, 3:],
            torch.tensor([2, 3, 2, 3, 2, 3], dtype=torch.int64)
            .unsqueeze(0)
            .expand(3, -1),
            rtol=0,
            atol=0,
        )


if __name__ == "__main__":
    unittest.main()
