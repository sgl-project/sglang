"""Unit tests for QSA's graph speculative row layout -- no server, no weights.

``QwenSparseAttnBackend._graph_speculative_layout`` builds the per-row metadata
every speculative graph replay reads. It expands ``req_pool_indices`` with
``repeat_interleave(..., output_size=actual_rows)``, so the test pins the
contract that makes passing ``output_size`` correct: the expansion has exactly
``row_lengths.numel()`` entries, one run per extend length, in row-major order.
"""

import unittest

import torch

from sglang.srt.layers.attention.qwen_sparse_attn_backend import QwenSparseAttnBackend
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _SpecInfo:
    """The two attributes the layout reads off the speculative input."""

    def __init__(self, draft_token_num=None, extend_seq_lens_cpu=None):
        self.draft_token_num = draft_token_num
        if extend_seq_lens_cpu is not None:
            self.extend_seq_lens_cpu = extend_seq_lens_cpu


def _expected_rows(seq_lens, extend_lens, target_verify):
    """Reference expansion, written out per request."""
    lengths, prefixes, requests = [], [], []
    for i, (seq_len, extend_len) in enumerate(zip(seq_lens, extend_lens)):
        effective = seq_len + extend_len if target_verify else seq_len
        prefix = max(effective - extend_len, 0)
        for k in range(extend_len):
            lengths.append(min(prefix + 1 + k, effective))
            prefixes.append(prefix)
            requests.append(i)
    return lengths, prefixes, requests


class TestQsaGraphSpeculativeLayout(CustomTestCase):
    def _run(self, seq_lens, extend_lens, target_verify, num_padding=0, slack=0):
        bs = len(seq_lens)
        kept = bs - max(0, min(num_padding, bs))
        effective_extend = list(extend_lens[:kept]) + [0] * (bs - kept)
        num_tokens = sum(effective_extend) + slack
        req_pool_indices = torch.arange(bs, dtype=torch.int32)
        if target_verify:
            forward_mode = ForwardMode.TARGET_VERIFY
            spec_info = _SpecInfo(draft_token_num=extend_lens[0])
        else:
            forward_mode = ForwardMode.DRAFT_EXTEND_V2
            spec_info = _SpecInfo(extend_seq_lens_cpu=list(extend_lens))
        row_lengths, row_req_pool_indices, row_prefix_lengths = (
            QwenSparseAttnBackend._graph_speculative_layout(
                bs,
                num_tokens,
                req_pool_indices,
                torch.tensor(seq_lens, dtype=torch.int64),
                forward_mode,
                spec_info,
                num_padding=num_padding,
            )
        )
        exp_lengths, exp_prefixes, exp_requests = _expected_rows(
            seq_lens[:kept], effective_extend[:kept], target_verify
        )
        real = len(exp_lengths)
        self.assertEqual(row_lengths[:real].tolist(), exp_lengths)
        self.assertEqual(row_prefix_lengths[:real].tolist(), exp_prefixes)
        self.assertEqual(row_req_pool_indices[:real].tolist(), exp_requests)
        # Every output is one row per token slot, padding included; this is the
        # identity output_size relies on.
        for name, out in (
            ("row_lengths", row_lengths),
            ("row_req_pool_indices", row_req_pool_indices),
            ("row_prefix_lengths", row_prefix_lengths),
        ):
            self.assertEqual(out.numel(), num_tokens, name)
        self.assertEqual(row_lengths.dtype, torch.int32)
        self.assertEqual(row_prefix_lengths.dtype, torch.int32)

    def test_uniform_extend_draft_extend(self):
        for bs in (1, 2, 8):
            with self.subTest(bs=bs):
                self._run(list(range(10, 10 + bs)), [2] * bs, target_verify=False)

    def test_uniform_extend_target_verify(self):
        for bs in (1, 4):
            with self.subTest(bs=bs):
                self._run([64, 65, 1023, 1024][:bs], [2] * bs, target_verify=False)
                self._run([64, 65, 1023, 1024][:bs], [2] * bs, target_verify=True)

    def test_ragged_extend(self):
        self._run([7, 1000, 33, 2048], [1, 4, 2, 3], target_verify=False)

    def test_zero_length_extends(self):
        self._run([7, 1000, 33, 2048], [0, 4, 0, 3], target_verify=False)

    def test_extend_longer_than_sequence_clamps(self):
        # prefix is floored at 0 and each row is capped at the sequence length
        self._run([1, 2, 3], [4, 4, 4], target_verify=False)

    def test_padded_rows_are_inert(self):
        seq_lens, extend_lens = [12, 34, 56, 78], [2, 2, 2, 2]
        self._run(seq_lens, extend_lens, target_verify=False, num_padding=2)
        bs = len(seq_lens)
        req_pool_indices = torch.arange(bs, dtype=torch.int32)
        row_lengths, row_req_pool_indices, row_prefix_lengths = (
            QwenSparseAttnBackend._graph_speculative_layout(
                bs,
                4,  # capacity for the two unpadded requests
                req_pool_indices,
                torch.tensor(seq_lens, dtype=torch.int64),
                ForwardMode.DRAFT_EXTEND_V2,
                _SpecInfo(extend_seq_lens_cpu=extend_lens),
                num_padding=2,
            )
        )
        self.assertEqual(row_req_pool_indices.tolist(), [0, 0, 1, 1])
        self.assertEqual(row_prefix_lengths.tolist(), [10, 10, 32, 32])
        self.assertEqual(row_lengths.tolist(), [11, 12, 33, 34])

    def test_tail_padding_fills_to_capacity(self):
        # A replay can carry fewer rows than the captured shape: the tail must be
        # inert metadata rather than a short tensor.
        bs = 2
        req_pool_indices = torch.arange(bs, dtype=torch.int32)
        row_lengths, row_req_pool_indices, row_prefix_lengths = (
            QwenSparseAttnBackend._graph_speculative_layout(
                bs,
                6,  # capacity 6, only 4 real rows
                req_pool_indices,
                torch.tensor([12, 34], dtype=torch.int64),
                ForwardMode.DRAFT_EXTEND_V2,
                _SpecInfo(extend_seq_lens_cpu=[2, 2]),
            )
        )
        self.assertEqual(row_lengths.tolist(), [11, 12, 33, 34, 1, 1])
        self.assertEqual(row_prefix_lengths.tolist(), [10, 10, 32, 32, 0, 0])
        self.assertEqual(row_req_pool_indices.tolist(), [0, 0, 1, 1, 0, 0])

    def test_capacity_overflow_is_rejected(self):
        with self.assertRaises(ValueError):
            self._run([12, 34], [4, 4], target_verify=False, slack=-4)


if __name__ == "__main__":
    unittest.main()
