"""Regression test: _process_tool_call_stream now yields ``(sse_chunk,
normal_text_delta)`` tuples so the caller can feed the content accumulator
used by the streaming Fallback B (unclosed-reasoning recovery).

Bug scenario this protects against:
  - request has tools + tool_choice="auto"
  - model emits: "<think>reasoning</think>real answer." (no tool call)
  - stream loop enters the tool branch and yields "real answer." as a content
    chunk via ``_process_tool_call_stream``
  - BEFORE the fix: the tool branch never called
    ``update_stream_state_with_content`` so ``state["content_text"]`` stayed
    empty, ``should_recover_stream_state`` returned True, and Fallback B
    issued a redundant continuation (double-response + wasted compute).
  - AFTER the fix: each content chunk is co-yielded with its text delta so
    the caller can fold it into ``state["content_text"]`` → recovery skipped.

We bypass the full streaming machinery by calling ``_process_tool_call_stream``
directly with a pre-populated ``parser_dict`` so no real FunctionCallParser /
JsonArrayParser is constructed.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")

import asyncio
import unittest
from types import SimpleNamespace
from typing import Tuple

from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sglang.srt.entrypoints.openai.recover_unclosed_reasoning import (
    new_stream_recovery_state,
    should_recover_stream_state,
    update_stream_state_with_content,
    update_stream_state_with_reasoning,
)
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.test.test_utils import CustomTestCase


class _FakeToolCallParser:
    """Drop-in for FunctionCallParser.parse_stream_chunk with scripted output."""

    def __init__(self, scripted: list):
        # Each item: (normal_text, calls_list). Consumed in order.
        self._scripted = list(scripted)

    def parse_stream_chunk(self, chunk_text: str) -> Tuple[str, list]:
        if not self._scripted:
            return "", []
        return self._scripted.pop(0)


def _basic_request(**overrides):
    base = {
        "model": "test",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "f",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        "tool_choice": "auto",
    }
    base.update(overrides)
    return ChatCompletionRequest(**base)


def _make_handler():
    handler = OpenAIServingChat.__new__(OpenAIServingChat)
    handler.tool_call_parser = "qwen25"
    handler.tokenizer_manager = SimpleNamespace(
        server_args=SimpleNamespace(incremental_streaming_output=False)
    )
    return handler


def _meta_content(token_id: str = "chatcmpl-test"):
    return {
        "meta_info": {
            "id": token_id,
            "prompt_tokens": 0,
            "completion_tokens": 1,
            "reasoning_tokens": 0,
        }
    }


def _drain(agen):
    async def _run():
        return [item async for item in agen]

    return asyncio.new_event_loop().run_until_complete(_run())


class TestToolCallStreamYieldsTuple(CustomTestCase):
    """Contract: each yield is a ``(str, Optional[str])`` tuple where the
    second element is the content text delta (or None for tool-call chunks).
    """

    def test_yields_normal_text_delta_for_content_chunk(self):
        handler = _make_handler()
        parser_dict = {0: _FakeToolCallParser([("real answer.", [])])}

        out = _drain(
            handler._process_tool_call_stream(
                index=0,
                delta="real answer.",
                parser_dict=parser_dict,
                content=_meta_content(),
                request=_basic_request(),
                has_tool_calls={},
            )
        )
        self.assertEqual(len(out), 1)
        sse_chunk, normal_text_delta = out[0]
        self.assertIsInstance(sse_chunk, str)
        self.assertTrue(sse_chunk.startswith("data: "))
        self.assertEqual(normal_text_delta, "real answer.")

    def test_yields_none_delta_for_tool_call_chunk(self):
        handler = _make_handler()

        class _Call:
            name = "f"
            tool_index = 0
            parameters = "{}"

        parser_dict = {0: _FakeToolCallParser([("", [_Call()])])}
        has_tool_calls: dict = {}

        out = _drain(
            handler._process_tool_call_stream(
                index=0,
                delta='<tool_call>{"name":"f"}',
                parser_dict=parser_dict,
                content=_meta_content(),
                request=_basic_request(),
                has_tool_calls=has_tool_calls,
            )
        )
        self.assertEqual(len(out), 1)
        sse_chunk, normal_text_delta = out[0]
        self.assertIsInstance(sse_chunk, str)
        self.assertIsNone(normal_text_delta)
        self.assertTrue(has_tool_calls[0])

    def test_no_normal_text_and_no_calls_yields_nothing(self):
        handler = _make_handler()
        parser_dict = {0: _FakeToolCallParser([("", [])])}
        out = _drain(
            handler._process_tool_call_stream(
                index=0,
                delta="partial",
                parser_dict=parser_dict,
                content=_meta_content(),
                request=_basic_request(),
                has_tool_calls={},
            )
        )
        self.assertEqual(out, [])


class TestToolBranchFeedsContentAccumulator(CustomTestCase):
    """End-to-end: simulate the fixed caller glue.

    Reasoning produced via the reasoning parser stream → then tool branch
    yields real content (no tool call) → caller folds both into the recovery
    state → ``should_recover_stream_state`` must return False.
    """

    def test_tool_branch_content_suppresses_recovery(self):
        handler = _make_handler()
        state = new_stream_recovery_state()

        # 1) Reasoning phase (parser returned reasoning_text).
        update_stream_state_with_reasoning(state, "thinking about it")

        # 2) Tool branch yields real content (no tool call). Caller glue in
        #    serving_chat.py now does exactly this: drain the (chunk,
        #    normal_text_delta) tuples and feed delta into the accumulator.
        parser_dict = {0: _FakeToolCallParser([("real answer.", [])])}
        for _chunk, normal_text_delta in _drain(
            handler._process_tool_call_stream(
                index=0,
                delta="real answer.",
                parser_dict=parser_dict,
                content=_meta_content(),
                request=_basic_request(),
                has_tool_calls={},
            )
        ):
            if normal_text_delta:
                update_stream_state_with_content(state, normal_text_delta)

        # 3) Stream finished normally.
        state["finish_reason"] = {"type": "stop"}

        # With the fix: content_text is populated → recovery MUST NOT fire.
        self.assertEqual(state["content_text"], "real answer.")
        self.assertFalse(should_recover_stream_state(state))

    def test_tool_branch_with_tool_call_and_no_content_still_recovers_correctly(
        self,
    ):
        # Sanity: if the tool branch yields ONLY a tool-call chunk (no
        # normal_text), and the stream state is then marked via
        # mark_stream_state_tool_call, recovery still skips because
        # has_tool_calls is True — independent of content_text.
        from sglang.srt.entrypoints.openai.recover_unclosed_reasoning import (
            mark_stream_state_tool_call,
        )

        handler = _make_handler()
        state = new_stream_recovery_state()
        update_stream_state_with_reasoning(state, "thinking")

        class _Call:
            name = "f"
            tool_index = 0
            parameters = "{}"

        parser_dict = {0: _FakeToolCallParser([("", [_Call()])])}
        has_tool_calls: dict = {}
        for _chunk, normal_text_delta in _drain(
            handler._process_tool_call_stream(
                index=0,
                delta='<tool_call>{"name":"f"}',
                parser_dict=parser_dict,
                content=_meta_content(),
                request=_basic_request(),
                has_tool_calls=has_tool_calls,
            )
        ):
            if normal_text_delta:
                update_stream_state_with_content(state, normal_text_delta)
        if has_tool_calls.get(0):
            mark_stream_state_tool_call(state)

        state["finish_reason"] = {"type": "stop"}
        self.assertEqual(state["content_text"], "")
        self.assertTrue(state["has_tool_calls"])
        self.assertFalse(should_recover_stream_state(state))


if __name__ == "__main__":
    unittest.main()
