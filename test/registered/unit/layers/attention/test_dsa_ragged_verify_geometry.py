"""CPU contracts for packed DSA query ownership and causal KV lengths."""

import unittest
from types import SimpleNamespace

import torch

from sglang.kernels.ops.speculative.ragged_verify_kernels import PadVerifyLayout
from sglang.srt.layers.attention.dsa.ragged_verify import update_ragged_dsa_query_rows
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _layout(lengths, tier):
    return SimpleNamespace(
        graph_num_tokens=tier,
        qo_indptr_device=torch.tensor(
            [0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32
        ),
    )


class TestDSARaggedVerifyGeometry(CustomTestCase):
    def _check(self, prefix, requests, verify, tier):
        seq = torch.tensor(prefix, dtype=torch.int64)
        req = torch.tensor(requests, dtype=torch.int64)
        out_seq = torch.full((tier,), -100, dtype=torch.int32)
        out_req = torch.full((tier,), -100, dtype=torch.int64)
        pointers = (out_seq.data_ptr(), out_req.data_ptr())
        update_ragged_dsa_query_rows(seq, req, _layout(verify, tier), out_seq, out_req)
        expected_seq, expected_req = [], []
        for row, count in enumerate(verify):
            expected_seq.extend(prefix[row] + j + 1 for j in range(count))
            expected_req.extend([requests[row]] * count)
        padding = tier - len(expected_seq)
        expected_seq.extend([0] * padding)
        expected_req.extend([0] * padding)
        self.assertEqual(out_seq.tolist(), expected_seq)
        self.assertEqual(out_req.tolist(), expected_req)
        self.assertEqual((out_seq.data_ptr(), out_req.data_ptr()), pointers)

    def test_uneven_widths_and_page_boundaries(self):
        self._check([0, 62, 4090], [5, 2, 7], [1, 3, 6], 10)

    def test_graph_ghost_rows_are_not_last_request_continuations(self):
        self._check([0, 62, 4090], [5, 2, 7], [1, 3, 6], 16)

    def test_live_layout_can_be_shorter_than_padded_request_slots(self):
        self._check([62, 127, 251, 999, 888], [7, 3, 5, 99, 98], [1, 3, 6], 16)

    def test_empty_request_slots_own_no_queries(self):
        self._check([62, 999, 127, 888], [7, 99, 3, 98], [1, 0, 3, 0], 16)

    def test_minimum_and_full_width_match_uniform_geometry(self):
        for width in (1, 6):
            with self.subTest(width=width):
                self._check([0, 63, 128], [4, 1, 6], [width] * 3, width * 3)

    def test_empty_layout_does_not_read_request_zero(self):
        self._check([], [], [], 8)

    def test_reused_buffers_follow_changed_lengths_and_batch(self):
        out_seq = torch.empty(16, dtype=torch.int32)
        out_req = torch.empty(16, dtype=torch.int64)
        for prefix, requests, verify in (
            ([62, 127, 251, 999], [7, 3, 5, 99], [1, 3, 6]),
            ([127, 999, 999, 999], [3, 99, 99, 99], [6]),
            ([0, 62, 251, 999], [5, 7, 3, 99], [6, 1, 3]),
        ):
            update_ragged_dsa_query_rows(
                torch.tensor(prefix),
                torch.tensor(requests),
                _layout(verify, 16),
                out_seq,
                out_req,
            )
            expected = [
                prefix[r] + j + 1 for r, n in enumerate(verify) for j in range(n)
            ]
            self.assertEqual(out_seq.tolist(), expected + [0] * (16 - len(expected)))

    def test_rejects_incompatible_buffer_geometry(self):
        seq, req = torch.tensor([62, 127]), torch.tensor([7, 3])
        good_seq, good_req = (
            torch.empty(4, dtype=torch.int32),
            torch.empty(4, dtype=torch.int64),
        )
        cases = (
            (seq, req[:1], _layout([1, 3], 4), good_seq, good_req),
            (seq, req, _layout([1, 1, 2], 4), good_seq, good_req),
            (seq, req, _layout([1, 3], 4), good_seq[:3], good_req[:3]),
            (seq, req, _layout([1, 3], 4), good_seq.to(torch.int64), good_req),
            (seq, req, _layout([1, 3], 4), good_seq, good_req.to(torch.int32)),
            (
                seq,
                req,
                _layout([1, 3], 4),
                torch.empty(8, dtype=torch.int32)[::2],
                good_req,
            ),
        )
        for args in cases:
            with self.subTest(shapes=[getattr(x, "shape", None) for x in args]):
                with self.assertRaises(ValueError):
                    update_ragged_dsa_query_rows(*args)


class TestCappedVerifyLayout(CustomTestCase):
    def test_graph_slack_does_not_extend_the_last_real_request(self):
        """A full request tier can still have unused token rows; these must
        never become extra recurrent steps for its final real request."""
        out = PadVerifyLayout.torch(
            verify_lens=torch.ones(8, dtype=torch.int32),
            graph_num_tokens=16,
            padded_bs=8,
            cap=6,
        )
        pointers = tuple(
            x.data_ptr() for x in (out.verify_lens, out.qo_indptr, out.extend_start_loc)
        )
        for widths in ([1] * 8, [1, 3, 1, 1, 1, 1, 1, 1]):
            with self.subTest(widths=widths):
                PadVerifyLayout.torch(
                    verify_lens=torch.tensor(widths, dtype=torch.int32),
                    graph_num_tokens=16,
                    padded_bs=8,
                    cap=6,
                    out=out,
                )
                self.assertEqual(out.verify_lens.tolist(), widths)
                self.assertEqual(out.qo_indptr[-1].item(), sum(widths))
                self.assertEqual(
                    tuple(
                        x.data_ptr()
                        for x in (out.verify_lens, out.qo_indptr, out.extend_start_loc)
                    ),
                    pointers,
                )

    def test_uncapped_layout_still_covers_the_token_tier(self):
        """Full-coverage consumers retain their explicit padding convention."""
        out = PadVerifyLayout.torch(
            verify_lens=torch.ones(8, dtype=torch.int32),
            graph_num_tokens=16,
            padded_bs=8,
        )
        self.assertEqual(out.verify_lens.tolist(), [1] * 7 + [9])
        self.assertEqual(out.qo_indptr[-1].item(), 16)


if __name__ == "__main__":
    unittest.main()
