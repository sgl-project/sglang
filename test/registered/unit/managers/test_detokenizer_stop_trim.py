import unittest
from types import SimpleNamespace

from sglang.srt.managers.detokenizer_manager import DetokenizerManager
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDetokenizerStopTrim(CustomTestCase):
    def trim(self, output, reason, no_stop_trim=False, gpt_oss=False):
        return DetokenizerManager.trim_matched_stop(
            SimpleNamespace(is_tool_call_parser_gpt_oss=gpt_oss),
            output,
            reason,
            no_stop_trim,
        )

    def test_zero_and_nonzero_stop_token_ids(self):

        for token_id in (0, 30):
            with self.subTest(token_id=token_id):
                reason = {"type": "stop", "matched": token_id}
                self.assertEqual(self.trim([123, token_id], reason), [123])
                self.assertEqual(self.trim([token_id], reason), [])
                self.assertEqual(
                    self.trim([123, token_id], reason, no_stop_trim=True),
                    [123, token_id],
                )

    def test_unmatched_length_and_string_stops(self):
        for reason in (None, {"type": "length"}, {"type": "stop", "matched": None}):
            self.assertEqual(self.trim([123, 0], reason), [123, 0])
        reason = {"type": "stop", "matched": "END"}
        self.assertEqual(self.trim("answerENDtail", reason), "answer")
        self.assertEqual(
            self.trim("answerENDtail", reason, no_stop_trim=True), "answerEND"
        )

    def test_gpt_oss_call_token_is_preserved(self):
        self.assertEqual(
            self.trim([123, 200012], {"matched": 200012}, gpt_oss=True),
            [123, 200012],
        )


if __name__ == "__main__":
    unittest.main()
