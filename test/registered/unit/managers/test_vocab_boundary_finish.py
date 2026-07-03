"""Regression tests for Req._check_vocab_boundary_finish NaN guard boundary."""

import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_batch import FINISH_MATCHED_STR, Req

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

VOCAB_SIZE = 1000


def _make_req(*, output_ids: list, eos_token_ids: set, stop_token_ids: set) -> Req:
    # Build a bare Req without running __init__; only the fields touched by
    # _check_vocab_boundary_finish are populated.
    req = Req.__new__(Req)
    req.output_ids = list(output_ids)
    req.vocab_size = VOCAB_SIZE
    req.eos_token_ids = eos_token_ids
    req.finished_reason = None
    req.finished_len = None

    class _SamplingParams:
        pass

    req.sampling_params = _SamplingParams()
    req.sampling_params.stop_token_ids = stop_token_ids
    return req


class TestVocabBoundaryFinish(CustomTestCase):
    def test_token_equal_to_vocab_size_is_out_of_bounds(self):
        # Valid ids are [0, vocab_size); id == vocab_size must trip the NaN guard.
        req = _make_req(
            output_ids=[5, VOCAB_SIZE], eos_token_ids={2}, stop_token_ids=set()
        )
        self.assertTrue(req._check_vocab_boundary_finish([5, VOCAB_SIZE]))
        self.assertIsInstance(req.finished_reason, FINISH_MATCHED_STR)
        self.assertEqual(req.finished_reason.matched, "NaN happened")
        self.assertEqual(req.finished_len, 2)
        # The offending slot is rewritten to the eos token.
        self.assertEqual(req.output_ids[1], 2)

    def test_token_above_vocab_size_is_out_of_bounds(self):
        # A wildly large garbage id (typical of NaN sampling) is caught.
        req = _make_req(
            output_ids=[5, VOCAB_SIZE + 12345], eos_token_ids={2}, stop_token_ids=set()
        )
        self.assertTrue(req._check_vocab_boundary_finish([5, VOCAB_SIZE + 12345]))
        self.assertEqual(req.finished_len, 2)

    def test_negative_token_is_out_of_bounds(self):
        # Negative ids also indicate corrupted sampling output.
        req = _make_req(output_ids=[5, -1], eos_token_ids={2}, stop_token_ids=set())
        self.assertTrue(req._check_vocab_boundary_finish([5, -1]))
        self.assertEqual(req.output_ids[1], 2)

    def test_max_valid_token_is_in_bounds(self):
        # id == vocab_size - 1 is the largest valid token and must not trip.
        req = _make_req(
            output_ids=[5, VOCAB_SIZE - 1], eos_token_ids={2}, stop_token_ids=set()
        )
        self.assertFalse(req._check_vocab_boundary_finish([5, VOCAB_SIZE - 1]))
        self.assertIsNone(req.finished_reason)
        self.assertIsNone(req.finished_len)
        self.assertEqual(req.output_ids[1], VOCAB_SIZE - 1)

    def test_stop_token_used_when_no_eos(self):
        # Without eos tokens, the slot is rewritten to a stop token instead.
        req = _make_req(
            output_ids=[VOCAB_SIZE], eos_token_ids=set(), stop_token_ids={7}
        )
        self.assertTrue(req._check_vocab_boundary_finish([VOCAB_SIZE]))
        self.assertEqual(req.output_ids[0], 7)


if __name__ == "__main__":
    unittest.main()
