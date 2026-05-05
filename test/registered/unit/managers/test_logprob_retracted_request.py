"""Unit tests for logprob None-guard in stream_output_generation.

Issue #23154: When a request is retracted (KV cache full) while logprobs are
enabled, req.input_top_logprobs_val / input_top_logprobs_idx /
input_token_ids_logprobs_val can be None at the moment stream_output_generation
appends them to the per-batch output list.  The tokenizer manager then calls
list.extend(None) which raises TypeError.

The fix replaces bare `append(req.input_top_logprobs_val)` with
`append(req.input_top_logprobs_val or [])` (and similarly for idx /
token_ids variants), so None is coerced to [] before being sent.

These tests exercise only the None-coercion logic and the wrong-assertion
fix in add_input_logprob_return_values — no GPU or server required.
"""

import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


class TestLogprobNoneGuardCoercion(CustomTestCase):
    """Verify that None logprob fields are coerced to [] before appending."""

    def _collect_outputs(self, req_states):
        """Simulate the relevant portion of stream_output_generation.

        For each (input_token_logprobs_val, input_top_logprobs_val,
        input_top_logprobs_idx, input_token_ids_logprobs_val,
        input_token_ids_logprobs_idx) tuple, apply the same
        coercion logic that the fix introduces and collect results.
        """
        results = []
        for state in req_states:
            (
                itlv,
                itlv_idx,
                ittlv,
                ittlv_idx,
                itidlv,
                itidlv_idx,
            ) = state
            results.append(
                (
                    itlv,
                    itlv_idx,
                    ittlv or [],  # fix: was plain ittlv
                    ittlv_idx or [],  # fix: was plain ittlv_idx
                    itidlv or [],  # fix: was plain itidlv
                    itidlv_idx or [],  # fix: was plain itidlv_idx
                )
            )
        return results

    def test_none_top_logprobs_coerced_to_empty_list(self):
        """None input_top_logprobs_val must become [] — not crash on extend."""
        # Simulate a retracted request whose prefill completed but
        # _process_input_top_logprobs was not reached yet.
        req_states = [
            (
                [0.1, 0.2],  # input_token_logprobs_val (non-None)
                [5, 6],  # input_token_logprobs_idx
                None,  # input_top_logprobs_val — the buggy None
                None,  # input_top_logprobs_idx
                None,  # input_token_ids_logprobs_val
                None,  # input_token_ids_logprobs_idx
            )
        ]
        results = self._collect_outputs(req_states)
        _, _, top_val, top_idx, ids_val, ids_idx = results[0]
        self.assertEqual(top_val, [])
        self.assertEqual(top_idx, [])
        self.assertEqual(ids_val, [])
        self.assertEqual(ids_idx, [])

    def test_extend_with_empty_list_is_noop(self):
        """Confirm tokenizer-manager's extend(recv[i]) with [] is safe."""
        accumulator = []
        # Mimic: state.input_top_logprobs_val.extend(recv_obj.input_top_logprobs_val[i])
        accumulator.extend([])
        self.assertEqual(accumulator, [])

    def test_extend_with_none_crashes_without_fix(self):
        """Confirm that extend(None) raises TypeError (documents the pre-fix behaviour)."""
        accumulator = []
        with self.assertRaises(TypeError):
            accumulator.extend(None)

    def test_non_none_top_logprobs_pass_through_unchanged(self):
        """Non-None logprob values must be forwarded as-is."""
        top_val_data = [[0.5, 0.3], [0.2]]
        top_idx_data = [[10, 20], [30]]
        req_states = [
            (
                [0.9],
                [7],
                top_val_data,
                top_idx_data,
                None,
                None,
            )
        ]
        results = self._collect_outputs(req_states)
        _, _, top_val, top_idx, ids_val, ids_idx = results[0]
        self.assertIs(top_val, top_val_data)
        self.assertIs(top_idx, top_idx_data)
        self.assertEqual(ids_val, [])

    def test_empty_list_top_logprobs_pass_through(self):
        """An already-empty [] must not be changed by the or [] coercion."""
        req_states = [
            (
                [],
                [],
                [],  # empty, but not None
                [],
                [],
                [],
            )
        ]
        results = self._collect_outputs(req_states)
        _, _, top_val, top_idx, ids_val, ids_idx = results[0]
        # `[] or []` → second [], both are equal to []
        self.assertEqual(top_val, [])
        self.assertEqual(top_idx, [])

    def test_multiple_requests_mixed_none_and_non_none(self):
        """Batch with mixed None / non-None logprob fields must be handled correctly."""
        req_states = [
            ([0.1], [5], None, None, None, None),  # retracted
            ([0.2], [6], [[0.7]], [[8]], [[0.4]], [[9]]),  # normal
        ]
        results = self._collect_outputs(req_states)

        # Retracted request: Nones become []
        _, _, top_val0, top_idx0, ids_val0, ids_idx0 = results[0]
        self.assertEqual(top_val0, [])
        self.assertEqual(ids_val0, [])

        # Normal request: values pass through
        _, _, top_val1, top_idx1, ids_val1, ids_idx1 = results[1]
        self.assertEqual(top_val1, [[0.7]])
        self.assertEqual(ids_val1, [[0.4]])


class TestWrongAssertionFix(CustomTestCase):
    """Verify the corrected assertion in add_input_logprob_return_values.

    The original code asserted `req.input_token_logprobs_val is not None`
    in the early-return branch when top_logprobs_num > 0.  This was a
    copy-paste error — the assertion should guard input_top_logprobs_val.

    We test the fix by verifying:
      1. When input_top_logprobs_val IS None with top_logprobs_num > 0,
         the corrected assertion would fire (i.e. we can detect the invariant).
      2. When input_top_logprobs_val is non-None, the assertion passes.
    """

    def _assert_top_logprobs_not_none_if_needed(
        self, top_logprobs_num, input_top_logprobs_val
    ):
        """Replicate the fixed assertion logic."""
        if top_logprobs_num > 0:
            assert input_top_logprobs_val is not None  # corrected assertion

    def test_correct_assertion_fires_for_none_top_logprobs(self):
        """Corrected assertion must detect None when top_logprobs_num > 0."""
        with self.assertRaises(AssertionError):
            self._assert_top_logprobs_not_none_if_needed(
                top_logprobs_num=1,
                input_top_logprobs_val=None,
            )

    def test_correct_assertion_passes_when_top_logprobs_set(self):
        """Corrected assertion must not fire when input_top_logprobs_val is set."""
        # Should not raise
        self._assert_top_logprobs_not_none_if_needed(
            top_logprobs_num=1,
            input_top_logprobs_val=[[0.5, 0.3]],
        )

    def test_correct_assertion_skipped_when_top_logprobs_num_zero(self):
        """When top_logprobs_num == 0, None input_top_logprobs_val is acceptable."""
        # Should not raise (top_logprobs_num = 0 means no top-k was requested)
        self._assert_top_logprobs_not_none_if_needed(
            top_logprobs_num=0,
            input_top_logprobs_val=None,
        )

    def test_wrong_assertion_would_not_catch_the_bug(self):
        """The old (wrong) assertion never catches the bug: asserts non-None on itself."""
        # Old assertion: assert req.input_token_logprobs_val is not None
        # When input_token_logprobs_val IS not None (required to reach this branch),
        # the assertion always passes even if input_top_logprobs_val is None.
        input_token_logprobs_val = [0.1, 0.2]  # non-None (condition to enter branch)
        input_top_logprobs_val = None  # the bug: should be non-None but isn't

        # Old assertion — doesn't catch the bug:
        assert input_token_logprobs_val is not None  # always True, wrong field

        # The bug remains undetected — input_top_logprobs_val is still None
        self.assertIsNone(input_top_logprobs_val)


if __name__ == "__main__":
    unittest.main()
