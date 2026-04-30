"""Unit tests for OpenAIServingResponses message construction."""

import importlib.abc
import importlib.machinery
import sys
import unittest
from unittest.mock import MagicMock, Mock


class _SglKernelLoader(importlib.abc.Loader):
    def create_module(self, spec):
        return None

    def exec_module(self, module):
        module.__getattr__ = lambda name: MagicMock()


class _SglKernelFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "sgl_kernel" or fullname.startswith("sgl_kernel."):
            return importlib.machinery.ModuleSpec(
                fullname,
                _SglKernelLoader(),
                is_package=True,
            )
        return None


try:
    import sgl_kernel  # noqa: F401
except (ImportError, OSError):
    sys.meta_path.insert(0, _SglKernelFinder())

from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputRefusal,
    ResponseOutputText,
)

from sglang.srt.entrypoints.openai.protocol import (
    ResponsesRequest,
    ResponsesResponse,
)
from sglang.srt.entrypoints.openai.serving_responses import OpenAIServingResponses
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


def _assistant_message(*contents):
    return ResponseOutputMessage(
        id="msg_1",
        type="message",
        role="assistant",
        status="completed",
        content=list(contents),
    )


def _output_text(text: str):
    return ResponseOutputText(type="output_text", text=text, annotations=[])


def _previous_response(output):
    return ResponsesResponse(
        id="resp_prev",
        model="test-model",
        output=output,
        status="completed",
    )


class TestConstructInputMessages(unittest.TestCase):
    def setUp(self):
        self.serving = Mock(spec=OpenAIServingResponses)
        self.serving.msg_store = {
            "resp_prev": [{"role": "user", "content": "first user turn"}],
        }

    def _construct(self, request, prev_response):
        return OpenAIServingResponses._construct_input_messages(
            self.serving, request, prev_response
        )

    def test_previous_output_is_replayed_as_assistant_text(self):
        request = ResponsesRequest(
            model="test-model",
            input="repeat that",
            instructions="You are a creative assistant.",
            previous_response_id="resp_prev",
        )
        prev_response = _previous_response(
            [_assistant_message(_output_text("Echoes of midnight"))]
        )

        messages = self._construct(request, prev_response)

        self.assertEqual(
            messages,
            [
                {"role": "system", "content": "You are a creative assistant."},
                {"role": "user", "content": "first user turn"},
                {"role": "assistant", "content": "Echoes of midnight"},
                {"role": "user", "content": "repeat that"},
            ],
        )

    def test_non_message_output_items_are_skipped(self):
        request = ResponsesRequest(
            model="test-model",
            input="next",
            previous_response_id="resp_prev",
        )
        prev_response = _previous_response(
            [
                ResponseFunctionToolCall(
                    id="fc_1",
                    type="function_call",
                    call_id="call_1",
                    name="lookup",
                    arguments="{}",
                ),
                _assistant_message(_output_text("final answer")),
            ]
        )

        messages = self._construct(request, prev_response)

        self.assertIn({"role": "assistant", "content": "final answer"}, messages)

    def test_refusal_parts_are_skipped(self):
        request = ResponsesRequest(
            model="test-model",
            input="next",
            previous_response_id="resp_prev",
        )
        prev_response = _previous_response(
            [
                _assistant_message(
                    _output_text("visible answer"),
                    ResponseOutputRefusal(type="refusal", refusal="nope"),
                )
            ]
        )

        messages = self._construct(request, prev_response)

        self.assertIn({"role": "assistant", "content": "visible answer"}, messages)
        self.assertNotIn({"role": "assistant", "content": "nope"}, messages)


if __name__ == "__main__":
    unittest.main(verbosity=2)
