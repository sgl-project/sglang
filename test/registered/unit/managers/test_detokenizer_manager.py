"""Unit tests for DetokenizerManager.trim_matched_stop."""

import unittest

from sglang.srt.managers.detokenizer_manager import DetokenizerManager
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "base-a-test-cpu")


class TestTrimMatchedStop(CustomTestCase):
    def setUp(self):
        self.mgr = DetokenizerManager.__new__(DetokenizerManager)
        self.mgr.is_tool_call_parser_gpt_oss = False

    # --- Single match (original paths, must remain unchanged) ---

    def test_single_str(self):
        result = self.mgr.trim_matched_stop("hello world", {"matched": "world"}, False)
        self.assertEqual(result, "hello ")

    def test_single_int(self):
        result = self.mgr.trim_matched_stop([1, 2, 3], {"matched": 3}, False)
        self.assertEqual(result, [1, 2])

    # --- Multiple stop strings (new path) ---

    def test_multi_strs_trim_at_earliest(self):
        result = self.mgr.trim_matched_stop(
            "hello world foo", {"matched": ["world", "foo"]}, False
        )
        self.assertEqual(result, "hello ")

    def test_multi_strs_none_match(self):
        result = self.mgr.trim_matched_stop(
            "hello", {"matched": ["world", "foo"]}, False
        )
        self.assertEqual(result, "hello")

    # --- Multiple stop tokens (new path) ---

    def test_multi_ints(self):
        result = self.mgr.trim_matched_stop([1, 2, 3, 4], {"matched": [3, 4]}, False)
        self.assertEqual(result, [1, 2])

    def test_multi_ints_overflow(self):
        result = self.mgr.trim_matched_stop([1], {"matched": [1, 2, 3]}, False)
        self.assertEqual(result, [])

    def test_multi_ints_empty_output(self):
        result = self.mgr.trim_matched_stop([], {"matched": [1, 2]}, False)
        self.assertEqual(result, [])

    # --- Edge cases ---

    def test_no_stop_trim(self):
        result = self.mgr.trim_matched_stop("hello world", {"matched": "world"}, True)
        self.assertEqual(result, "hello world")

    def test_empty_finished_reason(self):
        result = self.mgr.trim_matched_stop("hello world", {}, False)
        self.assertEqual(result, "hello world")

    def test_matched_none(self):
        result = self.mgr.trim_matched_stop("hello world", {"matched": None}, False)
        self.assertEqual(result, "hello world")

    def test_empty_matched_list(self):
        result = self.mgr.trim_matched_stop("hello", {"matched": []}, False)
        self.assertEqual(result, "hello")

    # --- gpt-oss tool call token ---

    def test_gpt_oss_single_token_kept(self):
        mgr = DetokenizerManager.__new__(DetokenizerManager)
        mgr.is_tool_call_parser_gpt_oss = True
        result = mgr.trim_matched_stop([100, 200012], {"matched": 200012}, False)
        self.assertEqual(result, [100, 200012])

    def test_gpt_oss_multi_token_kept(self):
        mgr = DetokenizerManager.__new__(DetokenizerManager)
        mgr.is_tool_call_parser_gpt_oss = True
        result = mgr.trim_matched_stop([100, 200012], {"matched": [99, 200012]}, False)
        self.assertEqual(result, [100, 200012])

    def test_gpt_oss_multi_token_not_last(self):
        mgr = DetokenizerManager.__new__(DetokenizerManager)
        mgr.is_tool_call_parser_gpt_oss = True
        result = mgr.trim_matched_stop(
            [100, 200012, 50], {"matched": [50, 200012]}, False
        )
        self.assertEqual(result, [100, 200012, 50])


if __name__ == "__main__":
    unittest.main()
