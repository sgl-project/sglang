import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from utils import (
    StreamFixture,
    collect_stream_events,
    engine_chunk,
    event_payloads,
    event_types,
    find_completed_event,
    make_serving,
)

from sglang.srt.entrypoints.openai.protocol import (
    RequestResponseMetadata,
    ResponsesRequest,
)
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class NonHarmonyStreamTestCase(CustomTestCase):
    def test_reasoning_parser_uses_processed_reasoning_state(self):
        serving = make_serving()
        serving.reasoning_parser = "deepseek-r1"
        request = ResponsesRequest(model="x", input="hi", stream=True, store=False)

        with patch(
            "sglang.srt.entrypoints.openai.serving_responses.ReasoningParser"
        ) as parser_cls:
            parser_cls.return_value.parse_stream_chunk.return_value = (None, "done")
            fixture = StreamFixture(serving, request, require_reasoning=True)
            fixture.run([engine_chunk("done", 1, finish=True)])

        self.assertTrue(parser_cls.call_args.kwargs["force_reasoning"])

    def test_k2_nested_effort_selects_streaming_reasoning_delimiter(self):
        serving = make_serving()
        serving.reasoning_parser = "k2_horizon"
        serving.tool_call_parser = None
        request = ResponsesRequest(
            model="IFM/K2-Horizon-7B",
            input="hi",
            reasoning={"effort": "medium"},
            stream=True,
            store=False,
        )

        events = StreamFixture(serving, request, require_reasoning=True).run(
            [engine_chunk("work</ifm|think_fast>\nanswer", 4, finish=True)]
        )
        types = event_types(events)
        payloads = event_payloads(events)
        reasoning = "".join(
            payload["delta"]
            for event_type, payload in zip(types, payloads)
            if event_type == "response.reasoning_text.delta"
        )
        answer = "".join(
            payload["delta"]
            for event_type, payload in zip(types, payloads)
            if event_type == "response.output_text.delta"
        )

        self.assertEqual(reasoning, "work")
        self.assertEqual(answer, "\nanswer")

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

        for payload in event_payloads(events):
            if payload["type"] in (
                "response.output_item.added",
                "response.output_item.done",
            ):
                self.assertEqual(payload["item"]["phase"], "final_answer")
        self.assertEqual(
            find_completed_event(events)["response"]["output"][0]["phase"],
            "final_answer",
        )

    def test_truncated_and_aborted_streams_have_matching_terminal_events(self):
        serving = make_serving()
        for finish_reason, status in (
            ({"type": "length"}, "incomplete"),
            (
                {"type": "abort", "status_code": 503, "message": "Worker unavailable"},
                "failed",
            ),
        ):
            with self.subTest(status=status):
                request = ResponsesRequest(
                    model="x", input="hi", stream=True, store=True
                )
                chunk = engine_chunk("partial answer", finish=True)
                chunk["meta_info"]["finish_reason"] = finish_reason
                events = StreamFixture(serving, request).run([chunk])
                terminal = event_payloads(events)[-1]
                self.assertEqual(terminal["type"], f"response.{status}")
                self.assertEqual(terminal["response"]["status"], status)
                self.assertNotIn("response.completed", event_types(events))
                stored = serving.response_store[request.request_id]
                self.assertEqual(stored.status, status)
                if status == "incomplete":
                    self.assertEqual(
                        terminal["response"]["incomplete_details"],
                        {"reason": "max_output_tokens"},
                    )
                else:
                    self.assertEqual(
                        terminal["response"]["error"]["message"], "Worker unavailable"
                    )
                self.assertEqual(
                    [p["sequence_number"] for p in event_payloads(events)],
                    list(range(len(events))),
                )

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

    def test_required_native_parser_matches_full_response(self):
        serving = make_serving()
        serving.reasoning_parser = None
        serving.tool_call_parser = "hunyuan"
        serving.tokenizer_manager.tokenizer.get_vocab.return_value = {"<tool_sep>": 1}
        raw = (
            "<tool_calls><tool_call>get_weather<tool_sep>"
            "<arg_key>city</arg_key><arg_value>Beijing</arg_value>"
            "</tool_call></tool_calls>"
        )
        for choice in ("required", {"type": "function", "name": "get_weather"}):
            with self.subTest(choice=choice):
                request = ResponsesRequest(
                    model="x",
                    input="hi",
                    stream=True,
                    store=False,
                    tool_choice=choice,
                    tools=[
                        {
                            "type": "function",
                            "name": "get_weather",
                            "parameters": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                            },
                        }
                    ],
                )
                (full_item,) = serving._make_response_output_items(
                    request,
                    raw,
                    serving.tokenizer_manager.tokenizer,
                    require_reasoning=False,
                )
                events = StreamFixture(serving, request).run(
                    [
                        engine_chunk(raw[:i], i, finish=i == len(raw))
                        for i in range(1, len(raw) + 1)
                    ]
                )
                (stream_item,) = find_completed_event(events)["response"]["output"]
                self.assertEqual(stream_item["type"], "function_call")
                self.assertEqual(stream_item["name"], full_item.name)
                self.assertEqual(stream_item["arguments"], full_item.arguments)
                deltas = "".join(
                    p["delta"]
                    for p in event_payloads(events)
                    if p["type"] == "response.function_call_arguments.delta"
                )
                self.assertEqual(deltas, full_item.arguments)

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
            parser_cls.return_value.parse_stream_end.return_value = ("", [])
            fixture = StreamFixture(serving, request)
            events = fixture.run(chunks)

        completed = find_completed_event(events)
        output = completed["response"]["output"]
        kinds = [item["type"] for item in output]
        self.assertEqual(kinds, ["message", "function_call", "message"])
        self.assertEqual(output[0]["content"][0]["text"], "I'll check.")
        self.assertEqual(output[1]["name"], "get_weather")
        self.assertEqual(output[2]["content"][0]["text"], "It's sunny.")
        self.assertEqual(output[0]["phase"], "commentary")
        self.assertEqual(output[2]["phase"], "final_answer")
        for payload in event_payloads(events):
            if (
                payload["type"]
                in ("response.output_item.added", "response.output_item.done")
                and payload["item"]["type"] == "message"
            ):
                self.assertEqual(
                    payload["item"]["phase"], output[payload["output_index"]]["phase"]
                )

    def test_reasoning_parser_flushed_at_stream_end(self):
        """Bug regression: the stream loop never drained text the reasoning
        parser held back as a possible marker prefix, so a response whose text
        genuinely ends with e.g. "<|e" lost that tail on /v1/responses (chat
        flushes via parse_stream_end; responses did not)."""
        serving = make_serving()
        serving.reasoning_parser = "muse"
        serving.tool_call_parser = None

        request = ResponsesRequest(model="x", input="hi", stream=True, store=False)
        text = (
            " to=self<|message|>think<|eom|>"
            "<|start|>assistant to=user<|message|>Answer<|e"
        )
        fixture = StreamFixture(serving, request)
        events = fixture.run(
            [
                engine_chunk(text[:30], 4),
                engine_chunk(text, 9, finish=True),
            ]
        )

        streamed = "".join(
            p["delta"]
            for ev, p in zip(event_types(events), event_payloads(events))
            if ev == "response.output_text.delta"
        )
        self.assertEqual(streamed, "Answer<|e")


class HarmonyStreamLifecycleTestCase(CustomTestCase):
    def test_truncated_harmony_arguments_close_the_emitted_item(self):
        from openai_harmony import Role

        from sglang.srt.entrypoints.context import StreamingHarmonyContext

        serving = make_serving()
        serving.use_harmony = True
        request = ResponsesRequest(model="x", input="hi", stream=True, store=False)
        context = Mock(spec=StreamingHarmonyContext)
        context.messages = []
        context.parser = SimpleNamespace(
            current_content='{"city":',
            current_role=Role.ASSISTANT,
            current_channel="commentary",
            current_recipient="functions.lookup",
        )
        context.num_prompt_tokens = 5
        context.num_output_tokens = 3
        context.num_cached_tokens = 0
        context.num_reasoning_tokens = 0
        context.finish_reason = {"type": "length"}

        async def generate():
            yield context

        events = asyncio.run(
            collect_stream_events(
                serving.responses_stream_generator(
                    request,
                    {},
                    generate(),
                    context,
                    "x",
                    Mock(),
                    RequestResponseMetadata(request_id=request.request_id),
                    require_reasoning=False,
                )
            )
        )
        payloads = event_payloads(events)
        self.assertEqual(payloads[-1]["type"], "response.incomplete")
        self.assertEqual(
            payloads[-1]["response"]["incomplete_details"],
            {"reason": "max_output_tokens"},
        )
        output = payloads[-1]["response"]["output"]
        self.assertEqual(output[0]["arguments"], '{"city":')
        self.assertEqual(output[0]["status"], "incomplete")
        added = next(
            p["item"] for p in payloads if p["type"] == "response.output_item.added"
        )
        done = next(
            p["item"] for p in payloads if p["type"] == "response.output_item.done"
        )
        self.assertEqual(added["id"], done["id"])
        self.assertEqual(done, output[0])
        self.assertEqual(added["call_id"], done["call_id"])

    def test_split_and_coalesced_messages_preserve_stream_items(self):
        from openai_harmony import Message, Role, StreamState

        from sglang.srt.entrypoints.context import StreamingHarmonyContext

        reasoning = Message.from_role_and_content(Role.ASSISTANT, "plan").with_channel(
            "analysis"
        )
        commentary = Message.from_role_and_content(
            Role.ASSISTANT, "checking"
        ).with_channel("commentary")
        call = (
            Message.from_role_and_content(Role.ASSISTANT, '{"city":"Beijing"}')
            .with_channel("commentary")
            .with_recipient("functions.lookup")
        )
        code = (
            Message.from_role_and_content(Role.ASSISTANT, "print(42)")
            .with_channel("commentary")
            .with_recipient("python")
        )
        search = (
            Message.from_role_and_content(Role.ASSISTANT, '{"query":"weather"}')
            .with_channel("commentary")
            .with_recipient("browser.search")
        )
        final = Message.from_role_and_content(Role.ASSISTANT, "answer").with_channel(
            "final"
        )
        snapshots = [
            ([], "analysis", None, "pl"),
            ([reasoning], "commentary", None, "checking"),
            ([reasoning, commentary], "commentary", "functions.lookup", '{"city":'),
            ([reasoning, commentary, call], "commentary", "python", "print("),
            ([reasoning, commentary, call, code, search], "final", None, "ans"),
            ([reasoning, commentary, call, code, search, final], None, None, ""),
        ]
        for chunks in (snapshots, [snapshots[0], snapshots[-1]]):
            with self.subTest(chunk_count=len(chunks)):
                serving = make_serving()
                serving.use_harmony = True
                request = ResponsesRequest(
                    model="x",
                    input="hi",
                    stream=True,
                    store=True,
                    include=["reasoning.encrypted_content"],
                    reasoning={"summary": "auto"},
                )
                context = StreamingHarmonyContext.__new__(StreamingHarmonyContext)
                context.num_init_messages = 2
                context.num_prompt_tokens = 5
                context.num_output_tokens = 10
                context.num_cached_tokens = 0
                context.num_reasoning_tokens = 0
                context.finish_reason = {"type": "stop"}
                context.last_tok = None
                context.encoding = Mock()
                context.encoding.stop_tokens_for_assistant_actions.return_value = []

                async def generate():
                    for messages, channel, recipient, text in chunks:
                        context.parser = SimpleNamespace(
                            messages=messages,
                            current_role=Role.ASSISTANT,
                            current_channel=channel,
                            current_recipient=recipient,
                            current_content=text,
                            last_content_delta=text,
                            state=StreamState.CONTENT,
                        )
                        yield context

                events = asyncio.run(
                    collect_stream_events(
                        serving.responses_stream_generator(
                            request,
                            {},
                            generate(),
                            context,
                            "x",
                            Mock(),
                            RequestResponseMetadata(request_id=request.request_id),
                            require_reasoning=False,
                        )
                    )
                )
                payloads = event_payloads(events)
                output = find_completed_event(events)["response"]["output"]
                self.assertEqual(
                    [item["type"] for item in output],
                    [
                        "reasoning",
                        "message",
                        "function_call",
                        "code_interpreter_call",
                        "web_search_call",
                        "message",
                    ],
                )
                added = [
                    p for p in payloads if p["type"] == "response.output_item.added"
                ]
                done = [p for p in payloads if p["type"] == "response.output_item.done"]
                self.assertEqual([p["output_index"] for p in added], list(range(6)))
                self.assertEqual([p["output_index"] for p in done], list(range(6)))
                self.assertEqual(
                    [p["item"]["id"] for p in added], [i["id"] for i in output]
                )
                self.assertEqual(len({i["id"] for i in output}), 6)
                self.assertEqual([p["item"] for p in done], output)
                self.assertEqual(
                    [p["sequence_number"] for p in payloads], list(range(len(payloads)))
                )
                for index, field, event_type in (
                    (0, "content", "response.reasoning_text.delta"),
                    (1, "content", "response.output_text.delta"),
                    (2, "arguments", "response.function_call_arguments.delta"),
                    (5, "content", "response.output_text.delta"),
                ):
                    text = "".join(
                        p["delta"]
                        for p in payloads
                        if p["type"] == event_type and p["output_index"] == index
                    )
                    expected = output[index][field]
                    self.assertEqual(
                        text, expected[0]["text"] if field == "content" else expected
                    )
                self.assertEqual(output[1]["phase"], "commentary")
                self.assertEqual(output[5]["phase"], "final_answer")
                self.assertTrue(output[0]["encrypted_content"])
                self.assertEqual(output[3]["code"], "print(42)")
                for event_type in (
                    "response.code_interpreter_call_code.done",
                    "response.code_interpreter_call.completed",
                    "response.web_search_call.completed",
                ):
                    self.assertIn(event_type, event_types(events))
                self.assertEqual(
                    serving.response_store[request.request_id].model_dump()["output"],
                    output,
                )


class MultiToolCallStreamingOrderTestCase(CustomTestCase):
    """The wire order of message / function_call items across tool-call deltas."""

    def setUp(self):
        from sglang.srt.function_call.qwen3_coder_detector import Qwen3CoderDetector

        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="tokenizer")

        self.serving = make_serving()
        self.serving.tool_call_parser = "qwen3_coder"
        self.serving.reasoning_parser = None

        det = Qwen3CoderDetector()
        s, e = det.tool_call_start_token, det.tool_call_end_token
        fp, fe = det.tool_call_prefix, det.function_end_token
        pp, pe = det.parameter_prefix, det.parameter_end_token
        self.weather = f"{s}{fp}get_weather>{pp}city>Beijing{pe}{fe}{e}"
        self.time = f"{s}{fp}get_time>{pp}tz>UTC{pe}{fe}{e}"
        # a prefix of ``weather`` that stops mid-arguments
        self.weather_head = f"{s}{fp}get_weather>{pp}city>Beij"

    def _seq(self, texts, *names):
        """Stream cumulative ``texts`` (last one final) and return (type, payload)."""
        request = ResponsesRequest(
            model="x",
            input="weather and time",
            store=False,
            tools=[
                {"type": "function", "name": n, "parameters": {"type": "object"}}
                for n in names
            ],
        )
        chunks = [engine_chunk(t) for t in texts]
        chunks.append(engine_chunk(texts[-1], finish=True))
        return StreamFixture(self.serving, request).run_seq(chunks)

    @staticmethod
    def _added(seq):
        return [
            (p["output_index"], p["item"].get("type"))
            for t, p in seq
            if t == "response.output_item.added"
        ]

    @staticmethod
    def _done_calls(seq):
        return [
            p["item"]
            for t, p in seq
            if t == "response.output_item.done"
            and p["item"].get("type") == "function_call"
        ]

    def test_prior_tool_call_done_before_next_added(self):
        full = self.weather + "\n" + self.time
        seq = self._seq(
            [self.weather, self.weather + "\n", full], "get_weather", "get_time"
        )

        def position(pred):
            return next(i for i, (t, p) in enumerate(seq) if pred(t, p))

        done0 = position(
            lambda t, p: t == "response.output_item.done" and p["output_index"] == 0
        )
        added1 = position(
            lambda t, p: t == "response.output_item.added" and p["output_index"] == 1
        )
        self.assertLess(done0, added1)

        items = self._done_calls(seq)
        self.assertEqual(sorted(i["name"] for i in items), ["get_time", "get_weather"])

    def test_prose_before_tool_call_keeps_message_first(self):
        """Prose and a tool-call start in one delta: the message item must come
        first, since the prose preceded the call."""
        # One delta spanning prose + the whole call, as spec decoding or
        # --stream-interval > 1 produces.
        seq = self._seq(["Let me check." + self.weather], "get_weather")

        added = self._added(seq)
        message_index = next(i for i, kind in added if kind == "message")
        call_index = next(i for i, kind in added if kind == "function_call")
        self.assertLess(message_index, call_index)

        # The call must not be split across two items by the reordering.
        self.assertEqual(len([k for _, k in added if k == "function_call"]), 1)

    def test_call_tail_prose_and_next_call_in_one_delta(self):
        """One delta closing tool1, carrying prose, and opening tool2 needs both
        orders at once: tool1's trailing "}" must be drained before the prose
        closes every open item, and tool2 must land after the message."""
        seq = self._seq(
            [self.weather_head, self.weather + "Here you go." + self.time],
            "get_weather",
            "get_time",
        )

        items = self._done_calls(seq)
        # No duplicate item invented for the already-closed call, and no call
        # left nameless by being reopened from an args-only fragment.
        self.assertEqual(len(items), 2)
        self.assertTrue(all(i["name"] for i in items))
        self.assertEqual(items[0]["arguments"], '{"city": "Beijing"}')


if __name__ == "__main__":
    unittest.main()
