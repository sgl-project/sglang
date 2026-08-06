import asyncio
import unittest
from unittest.mock import Mock, patch

from utils import (
    collect_stream_events,
    event_payloads,
    event_types,
    find_completed_event,
    make_serving,
)

from sglang.srt.entrypoints.openai.protocol import (
    RequestResponseMetadata,
    ResponsesRequest,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


class _StreamFixture:
    def __init__(self, serving, request, *, require_reasoning=False):
        self.serving = serving
        self.request = request
        self.require_reasoning = require_reasoning
        self.request_metadata = RequestResponseMetadata(request_id=request.request_id)

    def run(self, chunks):
        async def gen():
            for ch in chunks:
                yield ch

        async def collect():
            return await collect_stream_events(
                self.serving.responses_stream_generator_non_harmony(
                    self.request,
                    sampling_params={},
                    result_generator=gen(),
                    model_name="x",
                    tokenizer=Mock(),
                    request_metadata=self.request_metadata,
                    require_reasoning=self.require_reasoning,
                )
            )

        return asyncio.run(collect())


def _engine_chunk(text, completion_tokens, *, finish=False):
    return {
        "text": text,
        "meta_info": {
            "id": "rid",
            "prompt_tokens": 5,
            "completion_tokens": completion_tokens,
            "cached_tokens": 0,
            "reasoning_tokens": 0,
            "finish_reason": {"type": "stop"} if finish else None,
        },
    }


class NonHarmonyStreamTestCase(unittest.TestCase):
    def test_reasoning_parser_uses_processed_reasoning_state(self):
        serving = make_serving()
        serving.reasoning_parser = "deepseek-r1"
        request = ResponsesRequest(model="x", input="hi", stream=True, store=False)

        with patch(
            "sglang.srt.entrypoints.openai.serving_responses.ReasoningParser"
        ) as parser_cls:
            parser_cls.return_value.parse_stream_chunk.return_value = (None, "done")
            fixture = _StreamFixture(serving, request, require_reasoning=True)
            fixture.run([_engine_chunk("done", 1, finish=True)])

        self.assertTrue(parser_cls.call_args.kwargs["force_reasoning"])

    def test_emits_typed_sse_events_in_order(self):
        serving = make_serving()
        serving.reasoning_parser = None
        serving.tool_call_parser = None

        request = ResponsesRequest(model="x", input="hi", stream=True, store=False)
        fixture = _StreamFixture(serving, request)
        events = fixture.run(
            [
                _engine_chunk("Hel", 1),
                _engine_chunk("Hello", 2),
                _engine_chunk("Hello world", 4, finish=True),
            ]
        )

        types = event_types(events)
        self.assertEqual(types[0], "response.created")
        self.assertEqual(types[1], "response.in_progress")
        for ev in (
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
        ):
            self.assertIn(ev, types)
        self.assertEqual(types[-1], "response.completed")

        seqs = [p["sequence_number"] for p in event_payloads(events)]
        self.assertEqual(seqs, list(range(len(seqs))))

    def test_required_tool_choice_emits_function_call_events(self):
        serving = make_serving()
        serving.reasoning_parser = None
        serving.tool_call_parser = None

        request = ResponsesRequest(
            model="x",
            input="hi",
            stream=True,
            store=False,
            tool_choice="required",
            tools=[
                {
                    "type": "function",
                    "name": "get_weather",
                    "parameters": {"type": "object"},
                }
            ],
        )
        payload = '[{"name": "get_weather", "parameters": {"city": "Beijing"}}]'

        chunks = []
        sent = 0
        while sent < len(payload):
            sent += min(8, len(payload) - sent)
            chunks.append(
                _engine_chunk(payload[:sent], sent, finish=sent == len(payload))
            )

        fixture = _StreamFixture(serving, request)
        events = fixture.run(chunks)
        types = event_types(events)

        self.assertIn("response.function_call_arguments.delta", types)
        self.assertIn("response.function_call_arguments.done", types)
        self.assertIn("response.output_item.added", types)
        self.assertIn("response.output_item.done", types)
        self.assertNotIn("response.output_text.delta", types)

        added_kinds = [
            payload["item"]["type"]
            for payload in event_payloads(events)
            if payload.get("type") == "response.output_item.added"
        ]
        self.assertIn("function_call", added_kinds)

    def test_final_output_preserves_text_tool_text_order(self):
        from sglang.srt.function_call.core_types import (
            StreamingParseResult,
            ToolCallItem,
        )

        serving = make_serving()
        serving.reasoning_parser = None
        serving.tool_call_parser = "qwen3_coder"

        request = ResponsesRequest(
            model="x",
            input="hi",
            stream=True,
            store=False,
            tools=[
                {
                    "type": "function",
                    "name": "get_weather",
                    "parameters": {"type": "object"},
                }
            ],
        )

        scripted = [
            StreamingParseResult(normal_text="I'll check.", calls=[]),
            StreamingParseResult(
                normal_text="",
                calls=[
                    ToolCallItem(
                        tool_index=0,
                        name="get_weather",
                        parameters='{"city": "Beijing"}',
                    )
                ],
            ),
            StreamingParseResult(normal_text="It's sunny.", calls=[]),
        ]
        chunks = [
            _engine_chunk(" " * 3, 3),
            _engine_chunk(" " * 10, 10),
            _engine_chunk(" " * 14, 14, finish=True),
        ]

        script_iter = iter(scripted)

        def fake_parse_stream_chunk(delta):
            sp = next(script_iter)
            return sp.normal_text, sp.calls

        with patch(
            "sglang.srt.entrypoints.openai.serving_responses.FunctionCallParser"
        ) as parser_cls:
            parser_cls.return_value.detector.supports_structural_tag.return_value = True
            parser_cls.return_value.parse_stream_chunk.side_effect = (
                fake_parse_stream_chunk
            )
            fixture = _StreamFixture(serving, request)
            events = fixture.run(chunks)

        completed = find_completed_event(events)
        output = completed["response"]["output"]
        kinds = [item["type"] for item in output]
        self.assertEqual(kinds, ["message", "function_call", "message"])
        self.assertEqual(output[0]["content"][0]["text"], "I'll check.")
        self.assertEqual(output[1]["name"], "get_weather")
        self.assertEqual(output[2]["content"][0]["text"], "It's sunny.")


class ReasoningParserFinalizationTestCase(unittest.TestCase):
    """Test that the Responses streaming path calls parse_stream_end() and
    emits any buffered visible text returned by the reasoning parser's
    finish() method.

    Chat Completions calls parse_stream_end() when a stream finishes.
    The non-Harmony Responses streaming path must do the same to avoid
    silently dropping visible text that some reasoning detectors buffer
    until finalization.
    """

    def test_final_reasoning_text_emitted(self):
        """parse_stream_end() returns final reasoning text -> must appear
        as a reasoning delta event and in the completed response."""
        serving = make_serving()
        serving.reasoning_parser = "deepseek-r1"
        serving.tool_call_parser = None

        request = ResponsesRequest(
            model="x",
            input="hi",
            stream=True,
            store=False,
        )
        request_metadata = RequestResponseMetadata(request_id=request.request_id)

        chunks = [
            _engine_chunk("partial", 1),
            _engine_chunk("partial more", 3, finish=True),
        ]

        async def gen():
            for ch in chunks:
                yield ch

        with patch(
            "sglang.srt.entrypoints.openai.serving_responses.ReasoningParser"
        ) as parser_cls:
            # Per-chunk: no visible output (parser is buffering)
            parser_cls.return_value.parse_stream_chunk.return_value = (None, None)
            # Finalization: flush buffered reasoning
            parser_cls.return_value.parse_stream_end.return_value = (
                "final reasoning",
                None,
            )

            async def collect():
                return await collect_stream_events(
                    serving.responses_stream_generator_non_harmony(
                        request,
                        sampling_params={},
                        result_generator=gen(),
                        model_name="x",
                        tokenizer=Mock(),
                        request_metadata=request_metadata,
                    )
                )

            events = asyncio.run(collect())

        types = event_types(events)
        # Must have a reasoning text delta event for the final text
        self.assertIn("response.reasoning_text.delta", types)

        # The final reasoning text must appear exactly once as a delta
        deltas = [
            p.get("delta", "")
            for p in event_payloads(events)
            if p.get("type") == "response.reasoning_text.delta"
        ]
        self.assertEqual(deltas, ["final reasoning"])

        # Must also appear in the completed response output
        completed = find_completed_event(events)
        output = completed["response"]["output"]
        reasoning_items = [item for item in output if item.get("type") == "reasoning"]
        self.assertEqual(len(reasoning_items), 1)
        self.assertEqual(reasoning_items[0]["content"][0]["text"], "final reasoning")

    def test_final_normal_text_emitted(self):
        """parse_stream_end() returns final normal text -> must appear
        as an output_text.delta event and in the completed response."""
        serving = make_serving()
        serving.reasoning_parser = "deepseek-r1"
        serving.tool_call_parser = None

        request = ResponsesRequest(
            model="x",
            input="hi",
            stream=True,
            store=False,
        )
        request_metadata = RequestResponseMetadata(request_id=request.request_id)

        chunks = [
            _engine_chunk("partial", 1),
            _engine_chunk("partial more", 3, finish=True),
        ]

        async def gen():
            for ch in chunks:
                yield ch

        with patch(
            "sglang.srt.entrypoints.openai.serving_responses.ReasoningParser"
        ) as parser_cls:
            # Per-chunk: no visible output (parser is buffering)
            parser_cls.return_value.parse_stream_chunk.return_value = (None, None)
            # Finalization: flush buffered normal text
            parser_cls.return_value.parse_stream_end.return_value = (
                None,
                "final answer",
            )

            async def collect():
                return await collect_stream_events(
                    serving.responses_stream_generator_non_harmony(
                        request,
                        sampling_params={},
                        result_generator=gen(),
                        model_name="x",
                        tokenizer=Mock(),
                        request_metadata=request_metadata,
                    )
                )

            events = asyncio.run(collect())

        types = event_types(events)
        # Must have an output_text.delta event for the final text
        self.assertIn("response.output_text.delta", types)

        # The final normal text must appear exactly once as a delta
        deltas = [
            p.get("delta", "")
            for p in event_payloads(events)
            if p.get("type") == "response.output_text.delta"
        ]
        self.assertEqual(deltas, ["final answer"])

        # Must also appear in the completed response output
        completed = find_completed_event(events)
        output = completed["response"]["output"]
        message_items = [item for item in output if item.get("type") == "message"]
        self.assertEqual(len(message_items), 1)
        self.assertEqual(message_items[0]["content"][0]["text"], "final answer")

    def test_empty_finalization_emits_no_extra_events(self):
        """parse_stream_end() returns (None, None) -> no extra delta events."""
        serving = make_serving()
        serving.reasoning_parser = "deepseek-r1"
        serving.tool_call_parser = None

        request = ResponsesRequest(
            model="x",
            input="hi",
            stream=True,
            store=False,
        )
        request_metadata = RequestResponseMetadata(request_id=request.request_id)

        chunks = [
            _engine_chunk("hello", 1, finish=True),
        ]

        async def gen():
            for ch in chunks:
                yield ch

        with patch(
            "sglang.srt.entrypoints.openai.serving_responses.ReasoningParser"
        ) as parser_cls:
            # Per-chunk: emit reasoning immediately
            parser_cls.return_value.parse_stream_chunk.return_value = (
                "hello",
                None,
            )
            # Finalization: nothing buffered
            parser_cls.return_value.parse_stream_end.return_value = (None, None)

            async def collect():
                return await collect_stream_events(
                    serving.responses_stream_generator_non_harmony(
                        request,
                        sampling_params={},
                        result_generator=gen(),
                        model_name="x",
                        tokenizer=Mock(),
                        request_metadata=request_metadata,
                    )
                )

            events = asyncio.run(collect())

        types = event_types(events)
        # Exactly one reasoning delta from the chunk, none from finalization
        reasoning_deltas = [t for t in types if t == "response.reasoning_text.delta"]
        self.assertEqual(len(reasoning_deltas), 1)

    def test_abort_skips_finalization(self):
        """When finish_reason is 'abort', parse_stream_end() must not be called."""
        serving = make_serving()
        serving.reasoning_parser = "deepseek-r1"
        serving.tool_call_parser = None

        request = ResponsesRequest(
            model="x",
            input="hi",
            stream=True,
            store=False,
        )
        request_metadata = RequestResponseMetadata(request_id=request.request_id)

        chunks = [
            {
                "text": "partial",
                "meta_info": {
                    "id": "rid",
                    "prompt_tokens": 5,
                    "completion_tokens": 1,
                    "cached_tokens": 0,
                    "reasoning_tokens": 0,
                    "finish_reason": {"type": "abort"},
                },
            },
        ]

        async def gen():
            for ch in chunks:
                yield ch

        with patch(
            "sglang.srt.entrypoints.openai.serving_responses.ReasoningParser"
        ) as parser_cls:
            parser_cls.return_value.parse_stream_chunk.return_value = (
                "partial",
                None,
            )
            parser_cls.return_value.parse_stream_end.return_value = (
                "should not appear",
                None,
            )

            async def collect():
                return await collect_stream_events(
                    serving.responses_stream_generator_non_harmony(
                        request,
                        sampling_params={},
                        result_generator=gen(),
                        model_name="x",
                        tokenizer=Mock(),
                        request_metadata=request_metadata,
                    )
                )

            events = asyncio.run(collect())

        # parse_stream_end must NOT have been called
        parser_cls.return_value.parse_stream_end.assert_not_called()

        # "should not appear" must not be in any delta
        payloads = event_payloads(events)
        for p in payloads:
            self.assertNotIn("should not appear", str(p))


if __name__ == "__main__":
    unittest.main()
