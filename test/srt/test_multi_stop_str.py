"""Stop at the earliest matching stop_str when several stop_strs match.

Covers the TODO in DetokenizerManager.trim_matched_stop about handling the
case where multiple stop strings are hit. Pure python, no GPU needed.
"""

import unittest
from unittest.mock import MagicMock

from sglang.srt.managers.detokenizer_manager import DetokenizerManager
from sglang.srt.managers.schedule_batch import Req
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5)


def _make_req(stop_strs, decoded_text, tail_str):
    req = Req.__new__(Req)
    req.sampling_params = MagicMock()
    req.sampling_params.stop_strs = stop_strs
    req.sampling_params.stop_regex_strs = []
    req.decoded_text = decoded_text
    req.tail_str = MagicMock(return_value=tail_str)
    req.finished_reason = None
    req.finished_len = None
    req._locate_str_stop_finished_len = MagicMock(return_value=7)
    return req


class TestMultiStopStr(CustomTestCase):
    def test_single_stop_str_unchanged(self):
        req = _make_req(["STOP"], "hello STOP world", "hello STOP world")
        self.assertTrue(req._check_str_based_finish())
        self.assertEqual(req.finished_reason.matched, "STOP")
        req._locate_str_stop_finished_len.assert_called_once_with(1, stop_str="STOP")

    def test_earliest_match_wins_over_list_order(self):
        text = "xx STOP xx END"
        req = _make_req(["END", "STOP"], text, text)
        self.assertTrue(req._check_str_based_finish())
        self.assertEqual(req.finished_reason.matched, "STOP")

    def test_result_independent_of_list_order(self):
        text = "xx STOP xx END"
        req = _make_req(["STOP", "END"], text, text)
        self.assertTrue(req._check_str_based_finish())
        self.assertEqual(req.finished_reason.matched, "STOP")

    def test_no_match(self):
        req = _make_req(["NOPE"], "hello world", "hello world")
        self.assertFalse(req._check_str_based_finish())
        self.assertIsNone(req.finished_reason)

    def test_trim_removes_earliest_stop_str(self):
        out = DetokenizerManager.trim_matched_stop(
            MagicMock(),
            "xx STOP xx END",
            {"type": "stop", "matched": "STOP"},
            False,
        )
        self.assertEqual(out, "xx ")

    def test_trim_no_stop_trim_keeps_stop_str(self):
        out = DetokenizerManager.trim_matched_stop(
            MagicMock(),
            "xx STOP xx END",
            {"type": "stop", "matched": "STOP"},
            True,
        )
        self.assertEqual(out, "xx STOP")


if __name__ == "__main__":
    unittest.main()
