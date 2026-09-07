"""Unit test for DetokenizerManager.trim_matched_stop.

Under speculative decoding the output handed to trim_matched_stop may carry
content after the matched stop. With no_stop_trim the stop string must be kept
but the trailing over-generation still dropped (`output[:end]`); without it the
stop is removed (`output[:pos]`). Pure CPU: calls the method with a stub self,
so no DetokenizerManager.__init__ / IPC / tokenizer."""

import unittest
from types import SimpleNamespace

from sglang.srt.managers.detokenizer_manager import DetokenizerManager
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

GPT_OSS_CALL_TOKEN = 200012


def _trim(output, matched, no_stop_trim, *, gpt_oss=False):
    stub = SimpleNamespace(is_tool_call_parser_gpt_oss=gpt_oss)
    finished_reason = None if matched is None else {"matched": matched}
    return DetokenizerManager.trim_matched_stop(
        stub, output, finished_reason, no_stop_trim
    )


class TestTrimMatchedStop(unittest.TestCase):
    def test_no_finished_reason_returns_output(self):
        self.assertEqual(_trim("abc", None, False), "abc")

    def test_no_matched_returns_output(self):
        stub = SimpleNamespace(is_tool_call_parser_gpt_oss=False)
        self.assertEqual(
            DetokenizerManager.trim_matched_stop(stub, "abc", {}, False), "abc"
        )

    # --- stop string ---
    def test_str_trim_removes_stop(self):
        # no_stop_trim=False: drop the stop string and anything after it.
        self.assertEqual(_trim("ans\n\nQuestion: A", "Question", False), "ans\n\n")

    def test_str_no_trim_keeps_stop_but_drops_trailing(self):
        # no_stop_trim=True: keep through the stop, drop the over-generated tail.
        self.assertEqual(
            _trim("ans\n\nQuestion: A", "Question", True), "ans\n\nQuestion"
        )

    def test_str_not_found_returns_output(self):
        self.assertEqual(_trim("no stop here", "Question", False), "no stop here")

    # --- stop token ---
    def test_token_trim_drops_last(self):
        self.assertEqual(_trim([1, 2, 3], 3, False), [1, 2])

    def test_token_no_trim_keeps_all(self):
        self.assertEqual(_trim([1, 2, 3], 3, True), [1, 2, 3])

    def test_token_gpt_oss_call_kept(self):
        # gpt-oss tool-call token is also an eos; keep it even when trimming.
        self.assertEqual(
            _trim([1, 2, GPT_OSS_CALL_TOKEN], GPT_OSS_CALL_TOKEN, False, gpt_oss=True),
            [1, 2, GPT_OSS_CALL_TOKEN],
        )


if __name__ == "__main__":
    unittest.main()
