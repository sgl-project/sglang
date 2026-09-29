import unittest
from types import SimpleNamespace

from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.srt.parser.reasoning_parser import ReasoningParser
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestIQuestQ1ReasoningParser(CustomTestCase):
    def test_thinking_aliases_match_release(self):
        for kwargs, enabled in (
            ({}, True),
            ({"thinking": False}, False),
            ({"enable_thinking": False}, False),
            ({"thinking": False, "enable_thinking": True}, True),
            ({"thinking": True, "enable_thinking": False}, True),
            ({"enable_thinking": None}, True),
            ({"thinking": False, "enable_thinking": None}, False),
            ({"thinking": None, "enable_thinking": False}, False),
        ):
            with self.subTest(kwargs=kwargs):
                request = ChatCompletionRequest(
                    model="iquest-q1",
                    messages=[{"role": "user", "content": "hi"}],
                    chat_template_kwargs=kwargs,
                )
                serving = OpenAIServingChat.__new__(OpenAIServingChat)
                serving.reasoning_parser = "iquest_q1"
                serving.chat_encoding_spec = None
                serving.template_manager = SimpleNamespace(reasoning_config=None)
                serving._reasoning_detector = ReasoningParser("iquest_q1").detector
                self.assertEqual(
                    OpenAIServingChat._get_reasoning_from_request(serving, request),
                    enabled,
                )

    def test_request_thinking_toggle_controls_reasoning(self):
        serving = OpenAIServingChat.__new__(OpenAIServingChat)
        serving.reasoning_parser = "iquest_q1"
        request = ChatCompletionRequest(
            model="iquest-q1",
            messages=[{"role": "user", "content": "hi"}],
            chat_template_kwargs={"thinking": True, "enable_thinking": False},
        )
        for enabled, wire, expected in (
            (False, "answer", ("", "answer")),
            (True, "work</think>answer", ("work", "answer")),
        ):
            with self.subTest(enabled=enabled):
                serving.apply_reasoning_enabled(request, enabled)
                self.assertEqual(serving._get_reasoning_from_request(request), enabled)
                parser = ReasoningParser(
                    "iquest_q1", force_reasoning=not enabled, request=request
                )
                self.assertEqual(parser.parse_non_stream(wire), expected)


if __name__ == "__main__":
    unittest.main()
