"""Tests for FlashInfer multi-item scoring metadata and position updates.

The builder runs on the host, so these tests use CPU tensors and require no
model or GPU compute. Importing the backend module still requires the CUDA test
environment.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.flashinfer_backend import (
    FlashInferAttnBackend,
    _resolve_runner_min_positive_swa_window,
)
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

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


def _run(fb, enable_mis=True, min_positive_swa_window=None):
    backend = SimpleNamespace(
        enable_mis=enable_mis,
        min_positive_swa_window=min_positive_swa_window,
    )
    return FlashInferAttnBackend._process_multi_item_scoring(backend, fb)


def _dummy_radix_module(sliding_window_size: int) -> RadixAttention:
    mod = RadixAttention.__new__(RadixAttention)
    torch.nn.Module.__init__(mod)
    mod.sliding_window_size = sliding_window_size
    return mod


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

    def test_empty_delimiters_skipped(self):
        # A sequence with no delimiters is skipped; with only such sequences the
        # builder returns empty params.
        fb = _build([torch.tensor([], dtype=torch.int64)], 14)
        self.assertFalse(_run(fb).is_enabled())

    def test_resolve_runner_swa_window_paths(self):
        def _model_with(windows):
            model = torch.nn.Module()
            model.layers = torch.nn.ModuleList(
                [_dummy_radix_module(w) for w in windows]
            )
            return model

        hybrid = SimpleNamespace(
            model=_model_with([-1, 31, -1, 63]), sliding_window_size=31
        )
        self.assertEqual(
            _resolve_runner_min_positive_swa_window(model_runner=hybrid), 31
        )

        all_sliding = SimpleNamespace(
            model=_model_with([31, 31]), sliding_window_size=31
        )
        self.assertEqual(
            _resolve_runner_min_positive_swa_window(model_runner=all_sliding), 31
        )

        layer_only = SimpleNamespace(
            model=_model_with([-1, 32, -1]), sliding_window_size=None
        )
        self.assertEqual(
            _resolve_runner_min_positive_swa_window(model_runner=layer_only), 32
        )

        # Negative control: AllFull-Gemma3 still carries runner sliding_window_size
        # while every RadixAttention layer uses -1 (full attention).
        all_full = SimpleNamespace(
            model=_model_with([-1, -1, -1, -1]), sliding_window_size=31
        )
        self.assertIsNone(
            _resolve_runner_min_positive_swa_window(model_runner=all_full)
        )

        # Fallback when model has no RadixAttention modules (unit stubs).
        stub_pos = SimpleNamespace(model=None, sliding_window_size=31)
        self.assertEqual(
            _resolve_runner_min_positive_swa_window(model_runner=stub_pos), 31
        )
        stub_neg = SimpleNamespace(model=None, sliding_window_size=-1)
        self.assertIsNone(
            _resolve_runner_min_positive_swa_window(model_runner=stub_neg)
        )

    def test_wrapper_backend_resolve_min_positive_swa_window(self):
        class _WrapperBackend(AttentionBackend):
            def __init__(self, children):
                self._children = children

            def window_capacity_children(self) -> list:
                return self._children

        class _LeafBackend(AttentionBackend):
            def __init__(self, window):
                self.min_positive_swa_window = window

        wrapper = _WrapperBackend(
            [_LeafBackend(None), _LeafBackend(63), _LeafBackend(31)]
        )
        self.assertEqual(wrapper.resolve_min_positive_swa_window(), 31)

        all_none = _WrapperBackend([_LeafBackend(None), _LeafBackend(-1)])
        self.assertIsNone(all_none.resolve_min_positive_swa_window())

    def test_swa_guard_boundary_and_atomic_rejection(self):
        fb_ok = _build([torch.tensor([31, 33])], 34, seq_lens=[34])
        p_ok = _run(fb_ok, min_positive_swa_window=31)
        self.assertTrue(p_ok.is_enabled())
        self.assertEqual(p_ok.max_item_len_ptr.tolist(), [1])

        for delims, total in (
            ([torch.tensor([31, 34])], 35),
            ([torch.tensor([32, 34])], 35),
        ):
            fb_bad = _build(delims, total, seq_lens=[total])
            with self.assertRaises(ValueError):
                _run(fb_bad, min_positive_swa_window=31)
            self.assertEqual(fb_bad.positions.tolist(), [0] * total)
            p_disabled = _run(fb_bad, enable_mis=False, min_positive_swa_window=31)
            self.assertFalse(p_disabled.is_enabled())
            self.assertEqual(fb_bad.positions.tolist(), [0] * total)

        # Batch atomicity: seq 0 valid, seq 1 overflows -> zero positions mutated.
        fb_batch = _build(
            [torch.tensor([10, 12]), torch.tensor([31, 34])],
            49,
            seq_lens=[14, 35],
        )
        with self.assertRaises(ValueError):
            _run(fb_batch, min_positive_swa_window=31)
        self.assertEqual(fb_batch.positions.tolist(), [0] * 49)

    def test_swa_guard_cached_prefix_overflow(self):
        fb_ok = _build([torch.tensor([27, 29])], 30, seq_lens=[30], prefix_lens=[4])
        p_ok = _run(fb_ok, min_positive_swa_window=31)
        self.assertEqual(p_ok.prefix_len_ptr.tolist(), [31])

        fb_bad = _build([torch.tensor([27, 29])], 30, seq_lens=[30], prefix_lens=[5])
        with self.assertRaises(ValueError):
            _run(fb_bad, min_positive_swa_window=31)
        self.assertEqual(fb_bad.positions.tolist(), [0] * 30)

        fb_zero_item_bad = _build(
            [torch.tensor([27, 28])], 29, seq_lens=[29], prefix_lens=[5]
        )
        with self.assertRaises(ValueError):
            _run(fb_zero_item_bad, min_positive_swa_window=31)
        self.assertEqual(fb_zero_item_bad.positions.tolist(), [0] * 29)

    def test_swa_guard_dp_tp_padded_single_request_invariance(self):
        # query=31, item=1, window_left=31 (unpadded=34, padded to 36).
        for seq_lens in (None, [34]):
            with self.subTest(seq_lens=seq_lens):
                fb = _build([torch.tensor([31, 33])], 36, seq_lens=seq_lens)
                p = _run(fb, min_positive_swa_window=31)
                self.assertTrue(p.is_enabled())
                self.assertEqual(p.prefix_len_ptr.tolist(), [31])

    def test_swa_guard_zero_cuda_sync(self):
        def _forbid_sync(*_args, **_kwargs):
            raise AssertionError("Unexpected torch.cuda.synchronize during SWA guard")

        with patch("torch.cuda.synchronize", side_effect=_forbid_sync):
            fb = _build([torch.tensor([31, 33])], 36, seq_lens=[34])
            if torch.cuda.is_available():
                fb.input_ids = fb.input_ids.cuda()
                fb.positions = fb.positions.cuda()
                prev_mode = torch.cuda.get_sync_debug_mode()
                torch.cuda.set_sync_debug_mode("error")
                try:
                    p = _run(fb, min_positive_swa_window=31)
                finally:
                    torch.cuda.set_sync_debug_mode(prev_mode)
            else:
                p = _run(fb, min_positive_swa_window=31)
            self.assertTrue(p.is_enabled())


if __name__ == "__main__":
    unittest.main()
