"""Unit tests for the streaming Fallback B helpers and async orchestration.

Mocks tokenizer + tokenizer_manager.generate_request.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="stage-a-test-cpu")

import asyncio
import json
import unittest
from types import SimpleNamespace

from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sglang.srt.entrypoints.openai.recover_unclosed_reasoning import (
    is_stream_recovery_enabled_for_request,
    mark_stream_state_tool_call,
    new_stream_recovery_state,
    should_recover_stream_state,
    stream_recovery_chunks,
    update_stream_state_meta,
    update_stream_state_with_content,
    update_stream_state_with_reasoning,
)
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.test.test_utils import CustomTestCase


def _basic_request(**overrides):
    base = {
        "model": "test",
        "messages": [{"role": "user", "content": "hi"}],
    }
    base.update(overrides)
    return ChatCompletionRequest(**base)


class TestStreamStateHelpers(CustomTestCase):
    def test_initial_state(self):
        s = new_stream_recovery_state()
        self.assertEqual(s["reasoning_text"], "")
        self.assertEqual(s["content_text"], "")
        self.assertFalse(s["has_tool_calls"])
        self.assertIsNone(s["finish_reason"])
        self.assertEqual(s["last_output_ids"], [])

    def test_update_reasoning_concatenates(self):
        s = new_stream_recovery_state()
        update_stream_state_with_reasoning(s, "abc")
        update_stream_state_with_reasoning(s, "def")
        update_stream_state_with_reasoning(s, "")
        update_stream_state_with_reasoning(s, None)
        self.assertEqual(s["reasoning_text"], "abcdef")

    def test_update_content_concatenates(self):
        s = new_stream_recovery_state()
        update_stream_state_with_content(s, "x")
        update_stream_state_with_content(s, "y")
        self.assertEqual(s["content_text"], "xy")

    def test_mark_tool_call_flag(self):
        s = new_stream_recovery_state()
        mark_stream_state_tool_call(s)
        self.assertTrue(s["has_tool_calls"])

    def test_update_meta_updates_finish_and_output_ids(self):
        s = new_stream_recovery_state()
        update_stream_state_meta(
            s,
            {"finish_reason": {"type": "stop"}, "completion_tokens": 5},
            output_ids=[1, 2, 3],
        )
        self.assertEqual(s["finish_reason"], {"type": "stop"})
        self.assertEqual(s["last_output_ids"], [1, 2, 3])

    def test_update_meta_non_incremental_overwrites(self):
        # Default (cumulative) mode: each chunk snapshots the full list so far.
        s = new_stream_recovery_state()
        update_stream_state_meta(s, {}, output_ids=[1, 2])
        update_stream_state_meta(s, {}, output_ids=[1, 2, 3, 4, 5])
        self.assertEqual(s["last_output_ids"], [1, 2, 3, 4, 5])

    def test_update_meta_incremental_extends(self):
        # Incremental mode (server flag incremental_streaming_output=True):
        # each chunk carries only the delta and must be concatenated.
        s = new_stream_recovery_state()
        update_stream_state_meta(s, {}, output_ids=[1, 2], incremental=True)
        update_stream_state_meta(s, {}, output_ids=[3], incremental=True)
        update_stream_state_meta(s, {}, output_ids=[4, 5], incremental=True)
        self.assertEqual(s["last_output_ids"], [1, 2, 3, 4, 5])


class TestShouldRecoverStreamState(CustomTestCase):
    def _state(self, **overrides):
        s = new_stream_recovery_state()
        s.update(overrides)
        return s

    def test_triggers_with_reasoning_and_no_content_and_stop(self):
        s = self._state(
            reasoning_text="thinking",
            finish_reason={"type": "stop"},
        )
        self.assertTrue(should_recover_stream_state(s))

    def test_no_trigger_when_content_present(self):
        s = self._state(
            reasoning_text="thinking",
            content_text="answer",
            finish_reason={"type": "stop"},
        )
        self.assertFalse(should_recover_stream_state(s))

    def test_no_trigger_when_only_whitespace_content(self):
        # Aggressive Scene B: whitespace-only content still triggers recovery.
        s = self._state(
            reasoning_text="thinking",
            content_text="   \n  ",
            finish_reason={"type": "stop"},
        )
        self.assertTrue(should_recover_stream_state(s))

    def test_no_trigger_when_finish_length(self):
        s = self._state(
            reasoning_text="thinking",
            finish_reason={"type": "length"},
        )
        self.assertFalse(should_recover_stream_state(s))

    def test_no_trigger_when_tool_calls(self):
        s = self._state(
            reasoning_text="thinking",
            has_tool_calls=True,
            finish_reason={"type": "stop"},
        )
        self.assertFalse(should_recover_stream_state(s))

    def test_no_trigger_when_no_reasoning(self):
        s = self._state(finish_reason={"type": "stop"})
        self.assertFalse(should_recover_stream_state(s))

    def test_no_trigger_when_tool_branch_yielded_content(self):
        # Regression for: tools offered with tool_choice="auto", reasoning
        # produced, then the model produced real content (no tool call). The
        # content delta is yielded via _process_tool_call_stream, and the
        # caller MUST feed it back into state["content_text"] via
        # update_stream_state_with_content — otherwise Fallback B would
        # spuriously trigger a follow-up generation and the client would see
        # a double-response. This test encodes the contract that updating
        # content_text from the tool branch is sufficient to suppress
        # recovery.
        s = self._state(
            reasoning_text="thinking about it",
            finish_reason={"type": "stop"},
        )
        update_stream_state_with_content(s, "real answer part 1 ")
        update_stream_state_with_content(s, "part 2")
        self.assertEqual(s["content_text"], "real answer part 1 part 2")
        self.assertFalse(should_recover_stream_state(s))


class TestIsStreamRecoveryEnabledForRequest(CustomTestCase):
    def _handler(self, server_default, parser_name="qwen3"):
        h = SimpleNamespace()
        h.tokenizer_manager = SimpleNamespace(
            server_args=SimpleNamespace(
                auto_recover_unclosed_reasoning=server_default,
            )
        )
        h.reasoning_parser = parser_name
        return h

    def test_off_by_default(self):
        self.assertFalse(
            is_stream_recovery_enabled_for_request(
                self._handler(False), _basic_request()
            )
        )

    def test_on_by_server_default(self):
        self.assertTrue(
            is_stream_recovery_enabled_for_request(
                self._handler(True), _basic_request()
            )
        )

    def test_per_request_override(self):
        h = self._handler(False)
        self.assertTrue(
            is_stream_recovery_enabled_for_request(
                h, _basic_request(recover_unclosed_reasoning=True)
            )
        )

    def test_disabled_when_no_parser(self):
        h = self._handler(True, parser_name=None)
        self.assertFalse(
            is_stream_recovery_enabled_for_request(h, _basic_request())
        )


# ----- streaming generator ----------------------------------------------


class _FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        if text == "</think>":
            return [99]
        return [9999, 9998]


def _make_handler():
    h = SimpleNamespace()
    h.tokenizer_manager = SimpleNamespace(
        tokenizer=_FakeTokenizer(),
        server_args=SimpleNamespace(
            auto_recover_unclosed_reasoning=True,
            incremental_streaming_output=False,
        ),
    )
    h.template_manager = SimpleNamespace(force_reasoning=False)
    h.reasoning_parser = "qwen3"
    h._get_reasoning_from_request = lambda req: None
    return h


def _parse_sse_line(line: str):
    assert line.startswith("data: "), line
    body = line[len("data: ") :]
    if body.endswith("\n\n"):
        body = body[:-2]
    return json.loads(body)


class TestStreamRecoveryChunks(CustomTestCase):
    def _run(self, agen):
        async def _drain():
            return [c async for c in agen]

        return asyncio.new_event_loop().run_until_complete(_drain())

    def _state(self):
        s = new_stream_recovery_state()
        s["reasoning_text"] = "thinking only"
        s["content_text"] = ""
        s["finish_reason"] = {"type": "stop"}
        s["last_output_ids"] = [1, 2, 3]
        s["last_meta_info"] = {"finish_reason": {"type": "stop"}}
        return s

    def _adapted(self):
        return GenerateReqInput(input_ids=[10, 11], sampling_params={})

    def test_emits_content_delta_and_finish_chunk(self):
        handler = _make_handler()
        captured = {}

        def fake_generate(follow_req, raw):
            captured["follow_req"] = follow_req

            async def _gen():
                # Two deltas (cumulative text mode), then finish.
                yield {
                    "text": "the ",
                    "meta_info": {"completion_tokens": 1},
                }
                yield {
                    "text": "the answer",
                    "meta_info": {"completion_tokens": 2},
                }
                yield {
                    "text": "the answer.",
                    "meta_info": {
                        "completion_tokens": 3,
                        "finish_reason": {"type": "stop"},
                    },
                }

            return _gen()

        agen = stream_recovery_chunks(
            handler=handler,
            request=_basic_request(recover_unclosed_reasoning=True),
            raw_request=None,
            adapted_request=self._adapted(),
            state=self._state(),
            request_id="chatcmpl-test",
            request_model="m",
            continuous_usage_stats=False,
            prompt_tokens=10,
            reasoning_tokens_acc=3,
            completion_tokens_acc=3,
            generate_request_fn=fake_generate,
        )
        chunks = self._run(agen)
        # 3 content deltas + 1 final finish chunk
        self.assertEqual(len(chunks), 4)
        deltas = [_parse_sse_line(c) for c in chunks[:-1]]
        self.assertEqual(deltas[0]["choices"][0]["delta"]["content"], "the ")
        self.assertEqual(
            deltas[1]["choices"][0]["delta"]["content"], "answer"
        )
        self.assertEqual(deltas[2]["choices"][0]["delta"]["content"], ".")
        # All content deltas have no finish_reason
        for d in deltas:
            self.assertIsNone(d["choices"][0]["finish_reason"])

        final = _parse_sse_line(chunks[-1])
        self.assertEqual(final["choices"][0]["finish_reason"], "stop")
        # Continuation request must NOT carry the redirect processor.
        follow = captured["follow_req"]
        self.assertIsNone(follow.custom_logit_processor)
        # input_ids = original + last_output_ids + think_end_ids
        self.assertEqual(follow.input_ids, [10, 11, 1, 2, 3, 99])

    def test_no_yield_when_followup_returns_no_text(self):
        handler = _make_handler()

        def fake_generate(follow_req, raw):
            async def _gen():
                yield {
                    "text": "",
                    "meta_info": {"finish_reason": {"type": "stop"}},
                }

            return _gen()

        agen = stream_recovery_chunks(
            handler=handler,
            request=_basic_request(recover_unclosed_reasoning=True),
            raw_request=None,
            adapted_request=self._adapted(),
            state=self._state(),
            request_id="chatcmpl-test",
            request_model="m",
            continuous_usage_stats=False,
            prompt_tokens=10,
            reasoning_tokens_acc=3,
            completion_tokens_acc=3,
            generate_request_fn=fake_generate,
        )
        chunks = self._run(agen)
        # Only the final finish_reason chunk; no content delta.
        self.assertEqual(len(chunks), 1)
        final = _parse_sse_line(chunks[0])
        self.assertEqual(final["choices"][0]["finish_reason"], "stop")
        # Final finish chunk may include a `content` key but it must be empty/None.
        self.assertFalse((final["choices"][0]["delta"] or {}).get("content"))

    def test_yields_fallback_finish_when_followup_request_cannot_be_built(self):
        # text-mode adapted_request → build_continuation_request returns None.
        # Even on this early-return path, the helper MUST still yield a final
        # finish_reason chunk so the caller's ``continue`` does not swallow
        # the SSE stream's content-termination marker.
        handler = _make_handler()
        agen = stream_recovery_chunks(
            handler=handler,
            request=_basic_request(recover_unclosed_reasoning=True),
            raw_request=None,
            adapted_request=GenerateReqInput(text="hi", sampling_params={}),
            state=self._state(),
            request_id="x",
            request_model="m",
            continuous_usage_stats=False,
            prompt_tokens=0,
            reasoning_tokens_acc=0,
            completion_tokens_acc=0,
            generate_request_fn=lambda *a, **kw: (_ for _ in ()),
        )
        chunks = self._run(agen)
        self.assertEqual(len(chunks), 1)
        final = _parse_sse_line(chunks[0])
        self.assertEqual(final["choices"][0]["finish_reason"], "stop")

    def test_yields_fallback_finish_when_tokenizer_encode_returns_empty(self):
        # Tokenizer returns empty list for </think> → early-return must still
        # yield a finish chunk.
        handler = _make_handler()

        class _EmptyEncodeTok:
            def encode(self, text, add_special_tokens=False):
                return []

        handler.tokenizer_manager.tokenizer = _EmptyEncodeTok()

        agen = stream_recovery_chunks(
            handler=handler,
            request=_basic_request(recover_unclosed_reasoning=True),
            raw_request=None,
            adapted_request=self._adapted(),
            state=self._state(),
            request_id="x",
            request_model="m",
            continuous_usage_stats=False,
            prompt_tokens=0,
            reasoning_tokens_acc=0,
            completion_tokens_acc=0,
            generate_request_fn=lambda *a, **kw: (_ for _ in ()),
        )
        chunks = self._run(agen)
        self.assertEqual(len(chunks), 1)
        final = _parse_sse_line(chunks[0])
        self.assertEqual(final["choices"][0]["finish_reason"], "stop")

    def test_on_meta_update_called_with_final_followup_meta(self):
        # The callback must be invoked exactly once, with the LAST meta seen
        # in the continuation stream (usage fields from the follow-up are
        # cumulative, so only the final snapshot matters for accounting).
        handler = _make_handler()

        def fake_generate(follow_req, raw):
            async def _gen():
                yield {
                    "text": "ans",
                    "meta_info": {"completion_tokens": 1, "prompt_tokens": 20},
                }
                yield {
                    "text": "answer",
                    "meta_info": {
                        "completion_tokens": 7,
                        "cached_tokens": 4,
                        "prompt_tokens": 20,
                        "finish_reason": {"type": "stop"},
                    },
                }

            return _gen()

        captured = []

        def on_meta(meta):
            captured.append(dict(meta))

        agen = stream_recovery_chunks(
            handler=handler,
            request=_basic_request(recover_unclosed_reasoning=True),
            raw_request=None,
            adapted_request=self._adapted(),
            state=self._state(),
            request_id="chatcmpl-test",
            request_model="m",
            continuous_usage_stats=False,
            prompt_tokens=10,
            reasoning_tokens_acc=3,
            completion_tokens_acc=3,
            generate_request_fn=fake_generate,
            on_meta_update=on_meta,
        )
        _ = self._run(agen)
        self.assertEqual(len(captured), 1)
        final_meta = captured[0]
        self.assertEqual(final_meta["completion_tokens"], 7)
        self.assertEqual(final_meta["cached_tokens"], 4)
        self.assertEqual(final_meta["prompt_tokens"], 20)


if __name__ == "__main__":
    unittest.main()
