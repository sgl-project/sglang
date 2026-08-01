import unittest
from unittest.mock import patch

from utils import (
    StreamFixture,
    engine_chunk,
    event_payloads,
    event_types,
    find_completed_event,
    make_serving,
)

from sglang.srt.entrypoints.openai.protocol import ResponsesRequest
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


class NonHarmonyStreamTestCase(unittest.TestCase):
    def test_emits_typed_sse_events_in_order(self):
        serving = make_serving()
        serving.reasoning_parser = None
        serving.tool_call_parser = None

        request = ResponsesRequest(model="x", input="hi", stream=True, store=False)
        fixture = StreamFixture(serving, request)
        events = fixture.run(
            [
                engine_chunk("Hel", 1),
                engine_chunk("Hello", 2),
                engine_chunk("Hello world", 4, finish=True),
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
                engine_chunk(payload[:sent], sent, finish=sent == len(payload))
            )

        fixture = StreamFixture(serving, request)
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
            engine_chunk(" " * 3, 3),
            engine_chunk(" " * 10, 10),
            engine_chunk(" " * 14, 14, finish=True),
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
            fixture = StreamFixture(serving, request)
            events = fixture.run(chunks)

        completed = find_completed_event(events)
        output = completed["response"]["output"]
        kinds = [item["type"] for item in output]
        self.assertEqual(kinds, ["message", "function_call", "message"])
        self.assertEqual(output[0]["content"][0]["text"], "I'll check.")
        self.assertEqual(output[1]["name"], "get_weather")
        self.assertEqual(output[2]["content"][0]["text"], "It's sunny.")


if __name__ == "__main__":
    unittest.main()
