"""Unit tests for BaseFormatDetector._ends_with_partial_token — no server, no model."""

from sglang.srt.function_call.qwen25_detector import Qwen25Detector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "base-a-test-cpu")


class TestEndsWithPartialToken(CustomTestCase):
    def setUp(self):
        # _ends_with_partial_token is defined on BaseFormatDetector and takes the
        # bot_token as an argument, so any concrete detector exercises it.
        self.detector = Qwen25Detector()

    def _partial(self, buffer, bot_token):
        return self.detector._ends_with_partial_token(buffer, bot_token)

    def test_returns_longest_partial_for_repeating_prefix_tokens(self):
        # These real bot_tokens repeat "<" internally, so a *shorter* trailing
        # suffix of the buffer is also a prefix of the token. The helper must
        # return the LONGEST partial (the whole trailing partial token), not the
        # shortest, otherwise callers hold back too few characters and leak most
        # of the partial token into the normal-text stream.
        cases = [
            ("<|action_start|> <|plugin|>", "<|action_start|> <", 18),
            ("<|action_start|> <|plugin|>", "<|action_start|> <|", 19),
            ("<|start|>assistant<|channel|>commentary", "<|start|>assistant<", 19),
            ("<|start|>assistant<|channel|>commentary", "<|start|>assistant<|", 20),
        ]
        for bot_token, partial, expected in cases:
            self.assertEqual(
                self._partial("some normal text " + partial, bot_token),
                expected,
                msg=f"bot_token={bot_token!r} partial={partial!r}",
            )

    def test_simple_tokens(self):
        # Tokens without an internal border are unaffected.
        self.assertEqual(self._partial("hi <tool_ca", "<tool_call>"), 8)
        self.assertEqual(self._partial("body <tool_call", "<tool_call>\n"), 10)

    def test_no_partial_returns_zero(self):
        self.assertEqual(self._partial("just some normal text", "<tool_call>"), 0)
        self.assertEqual(self._partial("", "<tool_call>"), 0)
