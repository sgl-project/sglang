"""A chat turn can terminate with ``finish_reason == "stop"`` while the model
emitted only thinking tokens: the body is empty (or whitespace) and there is no
tool call. Clients that treat "no visible output, no tool call" as a failed turn
then drop the whole request even though the model was healthy.

``OpenAIServingChat`` repairs that case by retrying the request once with
thinking disabled, which makes the answer deterministic instead of empty.

The tests below pin the detection rules, the single-retry contract, and the
kill switch, all without a GPU or a loaded model.
"""

import asyncio
import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

from sglang.srt.entrypoints.openai.protocol import (  # noqa: E402
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
    FunctionResponse,
    ToolCall,
    UsageInfo,
)
from sglang.srt.entrypoints.openai.serving_chat import (  # noqa: E402
    _EMPTY_TURN_MIN_VISIBLE_CHARS,
    OpenAIServingChat,
)
from sglang.srt.environ import envs  # noqa: E402


class _FakeRequest:
    """The subset of ``ChatCompletionRequest`` the repair path touches."""

    def __init__(self, n: int = 1, reasoning_effort=None):
        self.n = n
        self.stream = False
        self.stream_options = None
        self.model = "test-model"
        self.reasoning_effort = reasoning_effort
        self.chat_template_kwargs = {}
        self.epoch = 0

    def model_copy(self, deep: bool = False):
        clone = _FakeRequest(n=self.n, reasoning_effort=self.reasoning_effort)
        clone.stream = self.stream
        clone.stream_options = self.stream_options
        clone.chat_template_kwargs = (
            dict(self.chat_template_kwargs) if deep else self.chat_template_kwargs
        )
        clone.epoch = self.epoch
        return clone


def _response(content=None, reasoning=None, tool_calls=None, reasoning_tokens=0):
    usage = UsageInfo(
        prompt_tokens=10,
        total_tokens=20,
        completion_tokens=10,
        reasoning_tokens=reasoning_tokens,
    )
    message = ChatMessage(
        role="assistant",
        content=content,
        reasoning_content=reasoning,
        tool_calls=tool_calls,
    )
    choice = ChatCompletionResponseChoice(
        index=0, message=message, finish_reason="stop"
    )
    return ChatCompletionResponse(
        id="chatcmpl-test",
        created=0,
        model="test-model",
        choices=[choice],
        usage=usage,
    )


def _ret_item(finish_type="stop", reasoning_tokens=0):
    return {
        "meta_info": {
            "finish_reason": {"type": finish_type},
            "reasoning_tokens": reasoning_tokens,
        }
    }


class _ServerHarness:
    """Drives ``_handle_non_streaming_request`` with scripted responses."""

    def __init__(self, responses, reasoning_tokens=0):
        self.server = OpenAIServingChat.__new__(OpenAIServingChat)
        self.requests = []
        self.responses = list(responses)
        self.reasoning_tokens = reasoning_tokens

        async def _generate(adapted_request, raw_request=None):
            self.requests.append(adapted_request)
            yield _ret_item(reasoning_tokens=self.reasoning_tokens)

        self.server.tokenizer_manager = SimpleNamespace(generate_request=_generate)
        self.server._convert_to_internal_request = (
            lambda request, raw_request=None: (request, request)
        )
        self.server._build_chat_response = (
            lambda request, ret, created: self.responses.pop(0)
        )

    def run(self, request):
        return asyncio.run(
            self.server._handle_non_streaming_request(request, request, None)
        )


class TestEmptyTurnDetection(unittest.TestCase):
    def setUp(self):
        self.server = OpenAIServingChat.__new__(OpenAIServingChat)

    def test_whitespace_after_thinking_is_empty(self):
        self.assertTrue(
            self.server._is_empty_terminal_turn(
                finish_reason="stop",
                content="\n\n",
                tool_call_count=0,
                reasoning_tokens=221,
            )
        )

    def test_no_reasoning_evidence_is_left_alone(self):
        # An empty turn with no reasoning at all is not the pathology this
        # repair targets: the model never started thinking.
        self.assertFalse(
            self.server._is_empty_terminal_turn(
                finish_reason="stop",
                content="",
                tool_call_count=0,
                reasoning_tokens=0,
            )
        )

    def test_reasoning_text_also_counts_as_evidence(self):
        self.assertTrue(
            self.server._is_empty_terminal_turn(
                finish_reason="stop",
                content="  ",
                tool_call_count=0,
                reasoning_tokens=0,
                reasoning_text="let me think about it",
            )
        )

    def test_visible_text_is_not_empty(self):
        self.assertFalse(
            self.server._is_empty_terminal_turn(
                finish_reason="stop",
                content="Hello",
                tool_call_count=0,
                reasoning_tokens=500,
            )
        )

    def test_tool_call_turn_is_not_empty(self):
        self.assertFalse(
            self.server._is_empty_terminal_turn(
                finish_reason="stop",
                content="",
                tool_call_count=1,
                reasoning_tokens=500,
            )
        )

    def test_length_finish_is_not_repaired(self):
        self.assertFalse(
            self.server._is_empty_terminal_turn(
                finish_reason="length",
                content="",
                tool_call_count=0,
                reasoning_tokens=500,
            )
        )

    def test_min_visible_chars_constant(self):
        self.assertEqual(_EMPTY_TURN_MIN_VISIBLE_CHARS, 1)

    def test_response_detection_reads_usage_and_message(self):
        response = _response(content="\n", reasoning="a long thought", reasoning_tokens=9)
        self.assertTrue(
            self.server._response_is_empty_turn(response, _ret_item(reasoning_tokens=9))
        )

    def test_response_detection_accepts_healthy_answer(self):
        response = _response(content="pong", reasoning_tokens=9)
        self.assertFalse(
            self.server._response_is_empty_turn(response, _ret_item(reasoning_tokens=9))
        )

    def test_response_detection_accepts_tool_call(self):
        tool_call = ToolCall(id="call_1", function=FunctionResponse(name="f", arguments="{}"))
        response = _response(content=None, tool_calls=[tool_call], reasoning_tokens=9)
        self.assertFalse(
            self.server._response_is_empty_turn(response, _ret_item(reasoning_tokens=9))
        )


class TestEmptyTurnRepair(unittest.TestCase):
    def setUp(self):
        self._repair_on = envs.SGLANG_DISABLE_EMPTY_TURN_REPAIR.override(False)
        self._repair_on.__enter__()

    def tearDown(self):
        self._repair_on.__exit__(None, None, None)

    def test_empty_turn_is_retried_once_with_thinking_off(self):
        harness = _ServerHarness(
            [_response(content="\n\n", reasoning="hmm", reasoning_tokens=221),
             _response(content="pong")],
            reasoning_tokens=221,
        )
        request = _FakeRequest()

        response = harness.run(request)
        self.assertEqual(response.choices[0].message.content, "pong")
        self.assertEqual(len(harness.requests), 2)

        retry = harness.requests[1]
        self.assertFalse(retry.stream)
        self.assertIsNone(retry.stream_options)
        self.assertIs(retry.chat_template_kwargs["enable_thinking"], False)
        # The caller's request object must not be mutated by the retry.
        self.assertFalse(request.stream)
        self.assertEqual(request.chat_template_kwargs, {})

    def test_deep_copied_retry_does_not_alias_caller_kwargs(self):
        harness = _ServerHarness(
            [_response(content="", reasoning="hmm", reasoning_tokens=5),
             _response(content="pong")],
            reasoning_tokens=5,
        )
        request = _FakeRequest()
        request.chat_template_kwargs = {"custom": 1}

        harness.run(request)
        self.assertEqual(harness.requests[1].chat_template_kwargs["custom"], 1)
        self.assertIs(harness.requests[1].chat_template_kwargs["enable_thinking"], False)
        self.assertEqual(request.chat_template_kwargs, {"custom": 1})

    def test_healthy_turn_is_not_retried(self):
        harness = _ServerHarness([_response(content="pong")], reasoning_tokens=0)
        response = harness.run(_FakeRequest())
        self.assertEqual(response.choices[0].message.content, "pong")
        self.assertEqual(len(harness.requests), 1)

    def test_retry_that_is_still_empty_returns_the_original(self):
        harness = _ServerHarness(
            [_response(content="\n\n", reasoning="hmm", reasoning_tokens=7),
             _response(content="  ", reasoning="hmm", reasoning_tokens=7)],
            reasoning_tokens=7,
        )
        response = harness.run(_FakeRequest())
        self.assertEqual(response.choices[0].message.content, "\n\n")
        self.assertEqual(len(harness.requests), 2)

    def test_kill_switch_disables_the_repair(self):
        with envs.SGLANG_DISABLE_EMPTY_TURN_REPAIR.override(True):
            harness = _ServerHarness(
                [_response(content="\n\n", reasoning="hmm", reasoning_tokens=7)],
                reasoning_tokens=7,
            )
            response = harness.run(_FakeRequest())
        self.assertEqual(response.choices[0].message.content, "\n\n")
        self.assertEqual(len(harness.requests), 1)

    def test_batched_n_is_not_retried(self):
        harness = _ServerHarness(
            [_response(content="\n\n", reasoning="hmm", reasoning_tokens=7)],
            reasoning_tokens=7,
        )
        harness.run(_FakeRequest(n=3))
        self.assertEqual(len(harness.requests), 1)


if __name__ == "__main__":
    unittest.main()
