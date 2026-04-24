"""Unit tests for the OpenAIServingResponses class from serving_responses.py.

Run with: pytest test/registered/unit/entrypoints/openai/test_serving_responses.py -v
"""

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede any import that pulls in sgl_kernel

import unittest
from unittest.mock import Mock

from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputRefusal,
    ResponseOutputText,
    ResponseReasoningItem,
)
from openai.types.responses.response_reasoning_item import (
    Content as ResponseReasoningTextContent,
)

from sglang.srt.entrypoints.openai.protocol import (
    ResponsesRequest,
    ResponsesResponse,
)
from sglang.srt.entrypoints.openai.serving_responses import OpenAIServingResponses
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


def _make_assistant_message(text: str, msg_id: str = "msg_1") -> ResponseOutputMessage:
    return ResponseOutputMessage(
        id=msg_id,
        type="message",
        role="assistant",
        status="completed",
        content=[
            ResponseOutputText(type="output_text", text=text, annotations=[]),
        ],
    )


def _make_prev_response(output, prev_id: str = "resp_prev") -> ResponsesResponse:
    return ResponsesResponse(
        id=prev_id,
        model="test-model",
        output=output,
        status="completed",
    )


class TestConstructInputMessages(unittest.TestCase):
    """Regression tests for the ``previous_response_id`` message history."""

    def setUp(self):
        # The method only reads ``self.msg_store``; everything else on the
        # serving instance is irrelevant for this codepath.
        self.serving = Mock(spec=OpenAIServingResponses)
        self.serving.msg_store = {
            "resp_prev": [{"role": "user", "content": "first user turn"}],
        }

    def _construct(self, request, prev_response):
        return OpenAIServingResponses._construct_input_messages(
            self.serving, request, prev_response
        )

    def test_previous_assistant_output_injected_as_assistant_role(self):
        """Regression test for #23662.

        Before the fix, the previous assistant output was injected as a
        ``system`` message whose content was the *current* request's
        ``instructions`` (i.e. the system prompt), so the model never saw
        its own prior reply.
        """
        request = ResponsesRequest(
            model="test-model",
            input="repeat what you just said",
            instructions="You are a creative assistant.",
            previous_response_id="resp_prev",
        )
        prev_response = _make_prev_response(
            output=[_make_assistant_message("Echoes of midnight")],
        )

        messages = self._construct(request, prev_response)

        # The assistant's prior turn must be replayed verbatim with the
        # ``assistant`` role — not as a duplicated system prompt.
        self.assertIn({"role": "assistant", "content": "Echoes of midnight"}, messages)

        # Guard against the original bug: the instructions string must not
        # appear under the ``system`` role more than once (the single
        # legitimate system message is at index 0).
        instruction_system_msgs = [
            m
            for m in messages
            if m.get("role") == "system"
            and m.get("content") == "You are a creative assistant."
        ]
        self.assertEqual(
            len(instruction_system_msgs),
            1,
            f"instructions should appear exactly once as a system message, "
            f"got messages={messages!r}",
        )

    def test_message_ordering(self):
        """Expected order: system, prior user turn, prior assistant turn, new user input."""
        request = ResponsesRequest(
            model="test-model",
            input="new user turn",
            instructions="sys prompt",
            previous_response_id="resp_prev",
        )
        prev_response = _make_prev_response(
            output=[_make_assistant_message("prior assistant turn")],
        )

        messages = self._construct(request, prev_response)

        self.assertEqual(
            messages,
            [
                {"role": "system", "content": "sys prompt"},
                {"role": "user", "content": "first user turn"},
                {"role": "assistant", "content": "prior assistant turn"},
                {"role": "user", "content": "new user turn"},
            ],
        )

    def test_multiple_text_parts_become_multiple_assistant_messages(self):
        request = ResponsesRequest(
            model="test-model",
            input="next",
            previous_response_id="resp_prev",
        )
        prev_response = _make_prev_response(
            output=[
                ResponseOutputMessage(
                    id="msg_1",
                    type="message",
                    role="assistant",
                    status="completed",
                    content=[
                        ResponseOutputText(
                            type="output_text", text="part one", annotations=[]
                        ),
                        ResponseOutputText(
                            type="output_text", text="part two", annotations=[]
                        ),
                    ],
                )
            ],
        )

        messages = self._construct(request, prev_response)

        assistant_msgs = [m for m in messages if m["role"] == "assistant"]
        self.assertEqual(
            assistant_msgs,
            [
                {"role": "assistant", "content": "part one"},
                {"role": "assistant", "content": "part two"},
            ],
        )

    def test_refusal_parts_are_skipped(self):
        """Refusal content has no ``.text``; it must not crash or be echoed."""
        request = ResponsesRequest(
            model="test-model",
            input="next",
            previous_response_id="resp_prev",
        )
        prev_response = _make_prev_response(
            output=[
                ResponseOutputMessage(
                    id="msg_1",
                    type="message",
                    role="assistant",
                    status="completed",
                    content=[
                        ResponseOutputText(
                            type="output_text", text="hello", annotations=[]
                        ),
                        ResponseOutputRefusal(type="refusal", refusal="nope"),
                    ],
                )
            ],
        )

        messages = self._construct(request, prev_response)

        assistant_msgs = [m for m in messages if m["role"] == "assistant"]
        self.assertEqual(assistant_msgs, [{"role": "assistant", "content": "hello"}])

    def test_reasoning_items_are_skipped(self):
        request = ResponsesRequest(
            model="test-model",
            input="next",
            previous_response_id="resp_prev",
        )
        prev_response = _make_prev_response(
            output=[
                ResponseReasoningItem(
                    id="rs_1",
                    type="reasoning",
                    summary=[],
                    content=[
                        ResponseReasoningTextContent(
                            type="reasoning_text", text="internal thought"
                        )
                    ],
                ),
                _make_assistant_message("visible answer"),
            ],
        )

        messages = self._construct(request, prev_response)

        # Reasoning text must not leak into the replayed history.
        self.assertNotIn({"role": "assistant", "content": "internal thought"}, messages)
        self.assertIn({"role": "assistant", "content": "visible answer"}, messages)

    def test_non_message_output_items_are_skipped(self):
        """Tool-call output items don't have ``.content``; they must be skipped."""
        request = ResponsesRequest(
            model="test-model",
            input="next",
            previous_response_id="resp_prev",
        )
        prev_response = _make_prev_response(
            output=[
                ResponseFunctionToolCall(
                    id="tc_1",
                    type="function_call",
                    name="do_thing",
                    call_id="call_1",
                    arguments="{}",
                ),
                _make_assistant_message("final answer"),
            ],
        )

        messages = self._construct(request, prev_response)

        self.assertIn({"role": "assistant", "content": "final answer"}, messages)

    def test_no_previous_response(self):
        request = ResponsesRequest(
            model="test-model",
            input="hello",
            instructions="sys prompt",
        )

        messages = self._construct(request, prev_response=None)

        self.assertEqual(
            messages,
            [
                {"role": "system", "content": "sys prompt"},
                {"role": "user", "content": "hello"},
            ],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
