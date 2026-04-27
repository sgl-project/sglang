"""Test that stop-string checking covers all newly accepted tokens,
not just the tail window — critical for speculative decoding.

This test uses pure mocks so it can run without sglang dependencies."""

import unittest
from unittest.mock import MagicMock

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-test-cpu")


def tail_str_new(self, new_accepted_len: int = 1) -> str:
    """Fixed version of tail_str that expands window for speculative decoding."""
    if (
        len(self.sampling_params.stop_strs) == 0
        and len(self.sampling_params.stop_regex_strs) == 0
    ):
        return ""

    max_len_tail_str = max(
        self.sampling_params.stop_str_max_len + 1,
        self.sampling_params.stop_regex_max_len + 1,
    )

    # Ensure the window covers all newly accepted tokens plus the
    # stop-string length so that multi-token speculative acceptance
    # never pushes an early stop string out of view.
    tail_len = min(
        max(max_len_tail_str, new_accepted_len + max_len_tail_str),
        len(self.output_ids),
    )
    return self.tokenizer.decode(self.output_ids[-tail_len:])


def tail_str_old(self) -> str:
    """Original (buggy) version of tail_str for comparison."""
    if (
        len(self.sampling_params.stop_strs) == 0
        and len(self.sampling_params.stop_regex_strs) == 0
    ):
        return ""

    max_len_tail_str = max(
        self.sampling_params.stop_str_max_len + 1,
        self.sampling_params.stop_regex_max_len + 1,
    )

    tail_len = min(max_len_tail_str, len(self.output_ids))
    return self.tokenizer.decode(self.output_ids[-tail_len:])


class TestStopStrSpeculative(unittest.TestCase):
    """Verify tail_str window expands to cover multi-token acceptance."""

    def _make_req(self, stop_strs, output_ids, stop_str_max_len=None):
        """Create a minimal mock that looks like a Req for tail_str."""
        req = MagicMock()
        req.output_ids = output_ids
        req.sampling_params.stop_strs = stop_strs
        req.sampling_params.stop_regex_strs = []
        req.sampling_params.stop_str_max_len = (
            stop_str_max_len
            if stop_str_max_len is not None
            else max(len(s) for s in stop_strs)
        )
        req.sampling_params.stop_regex_max_len = 0

        # Simple tokenizer mock: decode token ids to characters
        tok = MagicMock()

        def fake_decode(ids):
            return "".join(chr(i) for i in ids)

        tok.decode = fake_decode
        req.tokenizer = tok

        return req

    def test_default_window_near_old_behavior(self):
        """With new_accepted_len=1 (default), window is very close to old logic.

        The new formula adds +1 token to the window when new_accepted_len=1,
        which is negligible and safe for non-speculative paths.
        """
        stop_strs = ["hello"]  # stop_str_max_len = 5
        output_ids = list(range(65, 85))  # 20 tokens (ASCII A-T)
        req = self._make_req(stop_strs, output_ids)

        # Old: tail_len = min(5+1, 20) = 6
        # New with new_accepted_len=1: tail_len = min(max(6, 1+6), 20) = 7
        old_result = tail_str_old(req)
        new_result = tail_str_new(req, new_accepted_len=1)
        # New window is 1 token larger (7 vs 6), which is safe
        self.assertEqual(len(old_result), 6)
        self.assertEqual(len(new_result), 7)
        # New result contains old result as suffix
        self.assertTrue(new_result.endswith(old_result))

    def test_expanded_window_with_speculative(self):
        """With new_accepted_len > max_len_tail_str, window expands."""
        stop_strs = ["hi"]  # stop_str_max_len = 2
        output_ids = list(range(65, 85))  # 20 tokens
        req = self._make_req(stop_strs, output_ids)

        # max_len_tail_str = max(2+1, 0+1) = 3
        # With new_accepted_len=10: tail_len = min(max(3, 10+3), 20) = 13
        result = tail_str_new(req, new_accepted_len=10)
        self.assertEqual(len(result), 13)

    def test_window_clamped_to_output_len(self):
        """Window never exceeds len(output_ids)."""
        stop_strs = ["hi"]
        output_ids = list(range(65, 70))  # only 5 tokens
        req = self._make_req(stop_strs, output_ids)

        # new_accepted_len=100 would want window=103, but clamped to 5
        result = tail_str_new(req, new_accepted_len=100)
        self.assertEqual(len(result), 5)

    def test_stop_str_missed_by_old_window(self):
        """Old window misses stop string at the beginning of a large batch."""
        stop_strs = ["AB"]  # stop_str_max_len = 2
        # "AB" at positions 0-1, then 18 filler chars = 20 total
        output_ids = [65, 66] + list(range(67, 87))
        req = self._make_req(stop_strs, output_ids)

        # Old: tail_len = min(2+1, 20) = 3 → last 3 chars only
        old_tail = tail_str_old(req)
        self.assertNotIn("AB", old_tail, "Old window should miss the stop string")

    def test_stop_str_found_by_expanded_window(self):
        """Expanded window finds stop string at the beginning of a large batch."""
        stop_strs = ["AB"]  # stop_str_max_len = 2
        output_ids = [65, 66] + list(range(67, 87))
        req = self._make_req(stop_strs, output_ids)

        # New: tail_len = min(max(3, 20+3), 20) = 20 → all chars decoded
        new_tail = tail_str_new(req, new_accepted_len=20)
        self.assertIn("AB", new_tail, "Expanded window should find the stop string")

    def test_no_stop_strs_returns_empty(self):
        """Returns empty string when no stop strings configured."""
        output_ids = list(range(65, 85))
        req = self._make_req(["x"], output_ids)  # placeholder for mock setup
        req.sampling_params.stop_strs = []

        result = tail_str_new(req)
        self.assertEqual(result, "")

    def test_regex_max_len_also_expands_window(self):
        """When stop_regex_max_len is larger, it also benefits from expansion."""
        stop_strs = ["hi"]
        output_ids = list(range(65, 85))
        req = self._make_req(stop_strs, output_ids)
        req.sampling_params.stop_regex_max_len = 5

        # max_len_tail_str = max(2+1, 5+1) = 6
        # With new_accepted_len=10: tail_len = min(max(6, 10+6), 20) = 16
        result = tail_str_new(req, new_accepted_len=10)
        self.assertEqual(len(result), 16)


if __name__ == "__main__":
    unittest.main()
