"""Harmony usage bookkeeping with prepared engine chunks and parser/encoding mocks."""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from utils import (
    StreamFixture,
    collect_stream_events,
    engine_chunk,
    event_payloads,
    make_serving,
)

from sglang.srt.entrypoints.openai.protocol import (
    RequestResponseMetadata,
    ResponsesRequest,
)
from sglang.srt.runtime_context import reset_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class HarmonyStreamUsageTestCase(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        self.processed_tokens = []
        parser = SimpleNamespace(
            messages=[], current_content="", process=self.processed_tokens.append
        )
        for target, value in (
            ("get_streamable_parser_for_assistant", parser),
            ("get_encoding", SimpleNamespace(render=lambda msg: [90, 91])),
        ):
            patcher = patch(
                f"sglang.srt.entrypoints.context.{target}", return_value=value
            )
            patcher.start()
            self.addCleanup(patcher.stop)

    def _context(self):
        from sglang.srt.entrypoints.context import StreamingHarmonyContext

        return StreamingHarmonyContext(messages=[], tool_sessions={})

    @staticmethod
    def _chunk(ids, completion_tokens, *, prompt=5, cached=1, finish=None):
        return {
            "output_ids": ids,
            "meta_info": {
                "prompt_tokens": prompt,
                "cached_tokens": cached,
                "completion_tokens": completion_tokens,
                "finish_reason": finish,
            },
        }

    @staticmethod
    def _tool_output():
        from openai_harmony import Message, Role

        return [Message.from_role_and_content(Role.TOOL, "tool result")]

    def _run_stream(self, context, chunks):
        serving = make_serving()
        serving.use_harmony = True
        serving.enable_prompt_tokens_details = True
        request = ResponsesRequest(model="x", input="hi", stream=True, store=False)
        metadata = RequestResponseMetadata(request_id=request.request_id)

        async def generate():
            for chunk in chunks:
                context.append_output(chunk)
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
                    metadata,
                    require_reasoning=False,
                )
            )
        )
        response = event_payloads(events)[-1]["response"]
        self.assertEqual(
            response["usage"]["output_tokens"],
            metadata.final_usage_info.completion_tokens,
        )
        return response

    def _assert_usage(self, response, prompt, cached, completion):
        usage = response["usage"]
        self.assertEqual(usage["input_tokens"], prompt)
        self.assertEqual(usage["output_tokens"], completion)
        self.assertEqual(usage["total_tokens"], prompt + completion)
        self.assertEqual(usage["input_tokens_details"]["cached_tokens"], cached)

    def test_cumulative_usage(self):
        """Running-total snapshots must not be summed or re-parsed."""
        context = self._context()
        response = self._run_stream(
            context,
            [
                self._chunk([11], 1),
                self._chunk([11, 12, 13], 3),
                self._chunk(
                    [11, 12, 13], 3, prompt=8, cached=2, finish={"type": "stop"}
                ),
            ],
        )
        self._assert_usage(response, 8, 2, 3)
        self.assertEqual(self.processed_tokens, [11, 12, 13])

    def test_incremental_usage(self):
        """Delta IDs still carry cumulative completion metadata."""
        response = self._run_stream(
            self._context(),
            [
                self._chunk([11, 12], 2),
                self._chunk([13], 3, prompt=8, cached=2, finish={"type": "stop"}),
            ],
        )
        self._assert_usage(response, 8, 2, 3)
        self.assertEqual(self.processed_tokens, [11, 12, 13])

    def test_metadata_only_terminal_usage(self):
        """Empty terminal IDs must still update usage, including zero cache."""
        response = self._run_stream(
            self._context(),
            [
                self._chunk([11, 12], 2),
                self._chunk([], 3, prompt=8, cached=0, finish={"type": "stop"}),
            ],
        )
        self._assert_usage(response, 8, 0, 3)
        self.assertEqual(self.processed_tokens, [11, 12])

    def test_tool_continuation_usage(self):
        """Restarted running totals add to prior generations, excluding tool IDs."""
        for incremental in (False, True):
            with self.subTest(incremental=incremental):
                response = self._run_stream(
                    self._context(),
                    [
                        self._chunk([11, 12, 13], 3, finish={"type": "stop"}),
                        self._tool_output(),
                        self._chunk([21], 1, prompt=10, cached=2),
                        self._chunk(
                            [22] if incremental else [21, 22],
                            2,
                            prompt=12,
                            cached=4,
                            finish={"type": "stop"},
                        ),
                    ],
                )
                self._assert_usage(response, 12, 4, 5)

    def test_tool_continuation_resets_cursor(self):
        """A shorter second cumulative generation must not lose its first IDs."""
        context = self._context()
        context.append_output(self._chunk([11, 12, 13], 3))
        context.append_output(self._tool_output())
        context.append_output(self._chunk([21], 1))
        context.append_output(self._chunk([21, 22], 2))
        self.assertEqual(self.processed_tokens, [11, 12, 13, 90, 91, 21, 22])

    def test_aborted_or_incomplete_usage(self):
        """Terminal failure/length metadata preserves usage, even before decoding."""
        for reason, status in (("abort", "failed"), ("length", "incomplete")):
            for generated in (0, 2):
                with self.subTest(reason=reason, generated=generated):
                    chunks = [self._chunk([11, 12], 2)] if generated else []
                    chunks.append(
                        {
                            "output_ids": [],
                            "meta_info": {
                                "completion_tokens": generated,
                                "finish_reason": {"type": reason},
                            },
                        }
                    )
                    response = self._run_stream(self._context(), chunks)
                    self.assertEqual(response["status"], status)
                    self._assert_usage(
                        response,
                        5 if generated else 0,
                        1 if generated else 0,
                        generated,
                    )

    def test_buffered_completion_usage_unchanged(self):
        """Buffered generations accumulate completions independently of streaming."""
        from sglang.srt.entrypoints.context import HarmonyContext

        context = HarmonyContext(messages=[], tool_sessions={})
        context.append_output(self._chunk([11, 12, 13], 3))
        context.append_output(self._tool_output())
        context.append_output(self._chunk([21, 22], 2))
        self.assertEqual(context.num_output_tokens, 5)
        self.assertEqual(context.num_cached_tokens, 1)

    def test_simple_context_usage_unchanged(self):
        """Non-Harmony streams retain the last engine chunk's usage."""
        from sglang.srt.entrypoints.context import SimpleContext

        context = SimpleContext()
        serving = make_serving()
        serving.enable_prompt_tokens_details = True
        request = ResponsesRequest(model="x", input="hi", stream=True, store=False)
        chunks = [
            engine_chunk("a", 1),
            engine_chunk("answer", 3, finish=True),
        ]
        chunks[-1]["meta_info"].update(prompt_tokens=8, cached_tokens=2)
        for chunk in chunks:
            context.append_output(chunk)
        response = event_payloads(StreamFixture(serving, request).run(chunks))[-1][
            "response"
        ]
        self._assert_usage(response, 8, 2, 3)
        self.assertIs(context.last_output, chunks[-1])


if __name__ == "__main__":
    unittest.main()
