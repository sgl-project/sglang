from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede any import that pulls in sgl_kernel

import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.srt.entrypoints.openai import serving_inkling
from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.srt.environ import envs
from sglang.srt.parser.inkling_renderer import load_tml_renderers
from sglang.srt.parser.inkling_tokenizer import INKLING_SPECIAL_TOKEN_IDS
from sglang.srt.runtime_context import get_context, publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


def _inkling_serving() -> OpenAIServingChat:
    serving = object.__new__(OpenAIServingChat)
    serving.chat_encoding_spec = "inkling"
    serving.reasoning_parser = "inkling"
    serving.tool_call_parser = "inkling"
    serving.tokenizer_manager = Mock(tokenizer=None)
    serving._encoding_adapter = serving._build_chat_encoding_adapter()
    return serving


class InklingReasoningEffortTest(unittest.TestCase):
    def test_parse_reasoning_effort(self):
        """Named levels are external literals; scalars are range-checked."""
        cases = [
            ("none", 0.0),
            ("minimal", 0.1),
            ("low", 0.2),
            ("medium", 0.7),
            ("high", 0.9),
            # "xhigh" and "max" are aliases for the same 0.99 ceiling
            ("xhigh", 0.99),
            ("max", 0.99),
            (0.5, 0.5),
            (0.99, 0.99),
            (None, None),
        ]
        for value, expected in cases:
            with self.subTest(value=value):
                self.assertEqual(
                    serving_inkling.parse_reasoning_effort(value), expected
                )
        for value in (1.0, "1.0", 2.0, "1.5", -1.0, float("nan"), True, "garbage"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                serving_inkling.parse_reasoning_effort(value)

    def test_env_default(self):
        get = serving_inkling.get_default_reasoning_effort
        env = envs.SGLANG_INKLING_DEFAULT_REASONING_EFFORT
        try:
            env.clear()  # unset -> EnvStr default "0.9"
            self.assertEqual(get(), 0.9)
            env.set("")  # explicit empty still uses the protocol default
            self.assertEqual(get(), 0.9)
            env.set("0.7")
            self.assertEqual(get(), 0.7)
            for value in ("1.0", "1.1", "garbage"):
                env.set(value)
                with self.subTest(value=value), self.assertRaises(ValueError):
                    get()
        finally:
            env.clear()

    def test_thinking_disabled_maps_to_no_thinking_effort(self):
        """Bug regression: Inkling is an always-on parser, so Anthropic
        thinking={"type": "disabled"} was rejected outright even though effort
        "none" (0.0) expresses exactly that."""
        serving = object.__new__(OpenAIServingChat)
        serving.reasoning_parser = "inkling"
        serving.template_manager = Mock(reasoning_config=None)
        serving._reasoning_detector = Mock(reasoning_default="always")
        request = ChatCompletionRequest(
            model="test-model", messages=[{"role": "user", "content": "hi"}]
        )

        serving.apply_reasoning_enabled(request, False)
        self.assertEqual(request.reasoning_effort, "none")

        # Enabling leaves an effort set via output_config.effort alone.
        request.reasoning_effort = "low"
        serving.apply_reasoning_enabled(request, True)
        self.assertEqual(request.reasoning_effort, "low")


class InklingPromptTest(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="tokenizer")
        self.serving = _inkling_serving()

    def _encode(self, messages, **request_kwargs) -> list[int]:
        request = ChatCompletionRequest(
            model="test-model",
            messages=messages,
            reasoning_effort=0.5,
            **request_kwargs,
        )
        return self.serving._encode_messages(
            [message.model_dump() for message in request.messages],
            request,
            thinking_mode=None,
            image_data=[],
        )

    def test_prompt_tail(self):
        """Ordinary requests end at the last closed message; continue_final_message
        opens a model text block unless the trailing turn carries tool calls."""
        special = INKLING_SPECIAL_TOKEN_IDS
        end_sampling = special["<|content_model_end_sampling|>"]
        tool_call_turn = {
            "role": "assistant",
            "content": "calling",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "weather", "arguments": "{}"},
                }
            ],
        }
        open_block = [
            special["<|message_model|>"],
            special["<|content_text|>"],
            *load_tml_renderers().tokenizer.encode_ordinary("The answer"),
        ]
        user = {"role": "user", "content": "hello"}
        cases = [
            ("no model prefill", [user], False, [special["<|end_message|>"]]),
            (
                "continuation opens a text block",
                [user, {"role": "assistant", "content": "The answer"}],
                True,
                open_block,
            ),
            (
                "tool-call turn stays closed",
                [user, tool_call_turn],
                True,
                [end_sampling],
            ),
        ]
        for name, messages, continue_final_message, tail in cases:
            with self.subTest(case=name):
                prompt_ids = self._encode(
                    messages, continue_final_message=continue_final_message
                )
                self.assertEqual(prompt_ids[-len(tail) :], tail)
                if name == "continuation opens a text block":
                    self.assertNotIn(end_sampling, prompt_ids)

    def test_reasoning_content_history_renders_as_thinking(self):
        """Prior thinking sent as reasoning_content (Chat Completions, and the
        Anthropic and Responses converters) must render like a thinking part."""

        def encode(assistant):
            return self._encode(
                [
                    {"role": "user", "content": "hi"},
                    assistant,
                    {"role": "user", "content": "again"},
                ]
            )

        self.assertEqual(
            encode(
                {"role": "assistant", "reasoning_content": "ponder", "content": "ok"}
            ),
            encode(
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "ponder"},
                        {"type": "text", "text": "ok"},
                    ],
                }
            ),
        )


class InklingTokenOutputTest(CustomTestCase):
    """Inkling chat responses are parsed from output token IDs, not text."""

    def setUp(self):
        super().setUp()
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="tokenizer")
        self.tokenizer = load_tml_renderers().tokenizer
        self.inkling = _inkling_serving()._encoding_adapter
        self.request = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "weather?"}],
            tools=[
                {
                    "type": "function",
                    "function": {"name": "weather", "parameters": {"type": "object"}},
                }
            ],
        )

    def _output_ids(self) -> list[int]:
        special = self.tokenizer.encode_special
        return [
            special("message_model"),
            special("content_thinking"),
            *self.tokenizer.encode_ordinary("plan"),
            special("end_message"),
            special("message_model"),
            *self.tokenizer.encode_ordinary("weather"),
            special("content_invoke_tool_json"),
            *self.tokenizer.encode_ordinary('{"name":"weather","args":{"city":"SF"}}'),
            special("end_message"),
            special("content_model_end_sampling"),
        ]

    def _stream_deltas(
        self,
        request,
        output_ids,
        finish_reason,
        *,
        incremental=False,
        split=4,
        has_tool_calls=None,
    ) -> list[dict]:
        parser_dict, chunks = {}, []
        steps = (
            (output_ids[:split], None),
            (output_ids[split:] if incremental else output_ids, finish_reason),
        )
        with get_context().override_server_args(
            incremental_streaming_output=incremental
        ):
            for ids, finish in steps:
                chunks += self.inkling.stream_chat_chunks(
                    content={
                        "output_ids": ids,
                        "meta_info": {
                            "id": "chatcmpl-1",
                            "completion_tokens": len(output_ids),
                            "finish_reason": finish,
                        },
                    },
                    index=0,
                    request=request,
                    created=0,
                    parser_dict=parser_dict,
                    has_tool_calls={} if has_tool_calls is None else has_tool_calls,
                    choice_logprobs=None,
                    finish_reason_type=finish and finish["type"],
                    usage=None,
                )
        return [
            json.loads(chunk[len("data: ") :])["choices"][0]["delta"]
            for chunk in chunks
        ]

    def _reasoning_and_content(
        self, request, output_ids, finish_reason, *, split=2
    ) -> dict:
        reasoning, content, _, _ = self.inkling.parse_chat_response(
            request=request,
            output_ids=output_ids,
            finish_reason=finish_reason,
        )
        results = {"non-stream": (reasoning or "", content)}
        for incremental in (False, True):
            deltas = self._stream_deltas(
                request,
                output_ids,
                finish_reason,
                incremental=incremental,
                split=split,
            )
            results[f"stream incremental={incremental}"] = (
                "".join(d.get("reasoning_content") or "" for d in deltas),
                "".join(d.get("content") or "" for d in deltas),
            )
        return results

    def test_tool_call_output(self):
        """Calls map to OpenAI tool calls in bulk and as one complete streamed call;
        tool_choice="none" and separate_reasoning=False route them back to text."""
        finish = {"type": "stop", "matched": 200006}
        with self.subTest(case="non-stream"):
            reasoning, content, tool_calls, finish_reason = (
                self.inkling.parse_chat_response(
                    request=self.request,
                    output_ids=self._output_ids(),
                    finish_reason=finish,
                )
            )
            self.assertEqual(reasoning, "plan")
            self.assertEqual(content, "")
            self.assertEqual(finish_reason, {"type": "tool_calls", "matched": None})
            self.assertEqual(len(tool_calls), 1)
            self.assertEqual(tool_calls[0].index, 0)
            self.assertEqual(tool_calls[0].function.name, "weather")
            self.assertEqual(
                json.loads(tool_calls[0].function.arguments), {"city": "SF"}
            )
            self.assertTrue(tool_calls[0].id.startswith("call_"))

        with self.subTest(case="stream"):
            has_tool_calls = {}
            deltas = self._stream_deltas(
                self.request,
                self._output_ids(),
                finish,
                has_tool_calls=has_tool_calls,
            )
            self.assertEqual(
                "".join(d.get("reasoning_content") or "" for d in deltas), "plan"
            )
            calls = [call for d in deltas for call in d.get("tool_calls") or []]
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["function"]["name"], "weather")
            self.assertEqual(
                json.loads(calls[0]["function"]["arguments"]), {"city": "SF"}
            )
            self.assertEqual(has_tool_calls, {0: True})

        payload = '{"name":"weather","args":{"city":"SF"}}'
        switches = [
            ({"tool_choice": "none"}, ("plan", payload), "stop"),
            ({"separate_reasoning": False}, ("", "plan"), "tool_calls"),
        ]
        for update, expected, finish_type in switches:
            request = self.request.model_copy(update=update)
            _, _, tool_calls, finish_reason = self.inkling.parse_chat_response(
                request=request,
                output_ids=self._output_ids(),
                finish_reason=finish,
            )
            results = self._reasoning_and_content(request, self._output_ids(), finish)
            with self.subTest(update=update):
                self.assertEqual(finish_reason["type"], finish_type)
                self.assertEqual(tool_calls is None, finish_type == "stop")
                for mode, result in results.items():
                    with self.subTest(mode=mode):
                        self.assertEqual(result, expected)

    def test_each_output_token_is_parsed_once(self):
        """Non-incremental chunks carry the cumulative ids and share the live output
        list, and an incremental abort chunk re-sends the last id; a token parsed
        twice, or before its own chunk, would duplicate or leak text."""
        cases = [
            ([1, 2, 3, 4], 4, None, False, [4]),
            ([3, 4], 4, None, True, [3, 4]),
            ([4], 3, "abort", True, []),
            ([4], 4, "abort", True, [4]),
        ]
        for output_ids, completion_tokens, finish, incremental, expected in cases:
            with self.subTest(output_ids=output_ids, finish=finish):
                self.assertEqual(
                    serving_inkling.select_new_output_ids(
                        output_ids=output_ids,
                        num_consumed_tokens=3,
                        completion_tokens=completion_tokens,
                        finish_reason_type=finish,
                        incremental=incremental,
                    ),
                    expected,
                )

        # Tokens appended after a chunk was built (here the matched stop)
        # must wait for their own chunk.
        special = self.tokenizer.encode_special
        request = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "Yes?"}],
            stop=["."],
        )
        live = [
            special("message_model"),
            special("content_text"),
            *self.tokenizer.encode_ordinary("Yes"),
        ]
        first = {"completion_tokens": len(live), "finish_reason": None}
        live.extend(self.tokenizer.encode_ordinary(" ."))
        finish = {"type": "stop", "matched": "."}
        final = {"completion_tokens": len(live), "finish_reason": finish}
        chunks, parser_dict = [], {}
        with get_context().override_server_args(incremental_streaming_output=False):
            for meta_info, ids in ((first, live), (final, list(live))):
                chunks += self.inkling.stream_chat_chunks(
                    content={
                        "output_ids": ids,
                        "meta_info": {"id": "chatcmpl-1", **meta_info},
                    },
                    index=0,
                    request=request,
                    created=0,
                    parser_dict=parser_dict,
                    has_tool_calls={},
                    choice_logprobs=None,
                    finish_reason_type=meta_info["finish_reason"]
                    and meta_info["finish_reason"]["type"],
                    usage=None,
                )
        streamed = "".join(
            json.loads(chunk[len("data: ") :])["choices"][0]["delta"].get("content")
            or ""
            for chunk in chunks
        )
        _, content, _, _ = self.inkling.parse_chat_response(
            request=request, output_ids=list(live), finish_reason=finish
        )
        self.assertEqual(content, "Yes ")
        self.assertEqual(streamed, content)

    def test_matched_stop_is_trimmed(self):
        """Matched stops are trimmed where the sampler matched them: inside a token,
        across tokens, at the earliest of overlapping matches, never across message
        framing, and for special stops inside an open text block."""
        encode = self.tokenizer.encode_ordinary
        special = self.tokenizer.encode_special
        text_block = [special("message_model"), special("content_text")]
        thinking_block = [special("message_model"), special("content_thinking")]
        bang = encode("!")
        self.assertEqual(len(encode(" hello")), 1)
        first_block = [*text_block, *encode("EN"), special("end_message")]
        split_end = [*first_block, *text_block, *encode("D and actual END")]
        stop_id = special("content_thinking")
        hi = [{"role": "user", "content": "hi"}]
        continued = [
            {"role": "user", "content": "First five primes?"},
            {"role": "assistant", "content": "2, 3,"},
        ]
        # (name, messages, output_ids, matched, split, {no_stop_trim: expected})
        cases = [
            (
                "string spanning tokens",
                hi,
                text_block + encode("hello") + encode("EN") + encode("D"),
                "END",
                2,
                {False: ("", "hello")},
            ),
            (
                "string inside a token",
                hi,
                text_block + encode("say") + encode(" hello"),
                "hel",
                2,
                {False: ("", "say "), True: ("", "say hel")},
            ),
            (
                "string in reasoning",
                hi,
                thinking_block + encode("plan") + encode("END"),
                "END",
                2,
                {False: ("plan", "")},
            ),
            (
                "ordinary stop token",
                hi,
                text_block + encode("hello") + bang,
                bang[0],
                2,
                {False: ("", "hello"), True: ("", "hello!")},
            ),
            (
                "stop split by message framing",
                hi,
                split_end,
                "END",
                len(first_block),
                {False: ("", "END and actual "), True: ("", "END and actual END")},
            ),
            (
                "special stop in an explicit block",
                hi,
                [*text_block, *encode(" 5, 7"), stop_id],
                stop_id,
                3,
                {False: ("", " 5, 7"), True: ("", " 5, 7<|content_thinking|>")},
            ),
            (
                "special stop in a continued block",
                continued,
                [*encode(" 5, 7"), stop_id],
                stop_id,
                1,
                {False: ("", " 5, 7"), True: ("", " 5, 7<|content_thinking|>")},
            ),
        ]
        for previous, final, stop in (("\n", "\n\n", "\n\n"), ("ab", "aba", "aba")):
            sampled = [*encode(previous), *encode(final)]
            self.assertEqual(len(sampled), 2)
            cases.append(
                (
                    f"overlapping stop {stop!r}",
                    hi,
                    text_block + sampled,
                    stop,
                    2,
                    {False: ("", ""), True: ("", stop)},
                )
            )
        for name, messages, output_ids, matched, split, by_trim in cases:
            for no_stop_trim, expected in by_trim.items():
                request = ChatCompletionRequest(
                    model="test-model",
                    messages=messages,
                    continue_final_message=messages is continued,
                    no_stop_trim=no_stop_trim,
                )
                results = self._reasoning_and_content(
                    request,
                    output_ids,
                    {"type": "stop", "matched": matched},
                    split=split,
                )
                for mode, result in results.items():
                    with self.subTest(case=name, no_stop_trim=no_stop_trim, mode=mode):
                        self.assertEqual(result, expected)

    def test_headerless_output(self):
        """Headerless fallback payloads remain content, whether they start the
        response or follow thinking, and a stray ordinary prefix must not erase
        the following thinking or tool-call discriminator."""
        encode = self.tokenizer.encode_ordinary
        special = self.tokenizer.encode_special
        eos = special("content_model_end_sampling")
        payload = '{"marker": "<|end_message|>", "ok": true}'
        thinking = [
            special("message_model"),
            special("content_thinking"),
            *encode("plan"),
            special("end_message"),
        ]
        # Recovery can reopen text after a content-kind token;
        # a terminal EOS must still stay invisible.
        with_kind_token = [*encode('{"a": "'), special("content_text"), *encode('x"}')]
        unframed_thinking = [
            *encode("\n\n"),
            special("content_thinking"),
            *encode("plan"),
            special("end_message"),
            special("message_model"),
            special("content_text"),
            *encode("answer"),
            special("end_message"),
        ]
        cases = [
            ("after thinking", [*thinking, *encode(payload)], ("plan", payload)),
            ("from first token", encode(payload), ("", payload)),
            ("content kind in payload", with_kind_token, ("", '{"a": "x"}')),
            ("prefix before thinking", unframed_thinking, ("plan", "\n\nanswer")),
        ]
        for name, sampled, expected in cases:
            results = self._reasoning_and_content(
                self.request, [*sampled, eos], {"type": "stop", "matched": eos}
            )
            for mode, result in results.items():
                with self.subTest(case=name, mode=mode):
                    self.assertEqual(result, expected)

        call = [
            *encode(" "),
            special("content_invoke_tool_json"),
            *encode('{"name":"weather","args":{"city":"SF"}}'),
            special("end_message"),
            eos,
        ]
        _, _, tool_calls, finish_reason = self.inkling.parse_chat_response(
            request=self.request,
            output_ids=call,
            finish_reason={"type": "stop", "matched": eos},
        )
        self.assertEqual(finish_reason["type"], "tool_calls")
        self.assertEqual(
            [(c.function.name, json.loads(c.function.arguments)) for c in tool_calls],
            [("weather", {"city": "SF"})],
        )

    def test_constrained_continuation_stays_in_the_open_text_block(self):
        """The rendered continuation must select an open-answer grammar
        and parse its sampled suffix without leaking a header."""
        from xgrammar import TokenizerInfo, VocabType

        from sglang.srt.constrained.reasoner_grammar_backend import (
            ReasonerGrammarBackend,
        )
        from sglang.srt.constrained.xgrammar_backend import XGrammarGrammarBackend
        from sglang.srt.parser.reasoning_parser import InklingDetector

        special = self.tokenizer.encode_special
        eos = special("content_model_end_sampling")
        vocab = []
        for token_id in range(201088):
            try:
                vocab.append(self.tokenizer.decode_bytes([token_id]))
            except ValueError:
                vocab.append(b"")
        info = TokenizerInfo(vocab, VocabType.RAW, stop_token_ids=[eos])
        backend = XGrammarGrammarBackend(
            SimpleNamespace(init_xgrammar=lambda: (info, None)), len(vocab)
        )
        reasoner = ReasonerGrammarBackend(
            backend, SimpleNamespace(detector=InklingDetector()), None
        )
        request = ChatCompletionRequest(
            model="test-model",
            messages=[
                {"role": "user", "content": "Answer yes."},
                {"role": "assistant", "content": "Answer: "},
            ],
            continue_final_message=True,
            regex="yes",
        )
        prompt_ids = self.inkling.encode_messages(
            messages=[message.model_dump() for message in request.messages],
            request=request,
            tools=None,
            thinking=False,
            image_data=[],
        )
        key = reasoner.get_request_grammar_key(
            ("regex", "yes"), input_ids=prompt_ids, thinking_budget=None
        )
        grammar = reasoner._init_value_dispatch(key, reasoning=True)
        output_ids = [
            *self.tokenizer.encode_ordinary("yes"),
            special("end_message"),
            eos,
        ]
        mask = backend.allocate_vocab_mask(len(vocab), 1, "cpu")
        for token_id in [special("message_model"), *output_ids]:
            grammar.fill_vocab_mask(mask, 0)
            allowed = bool(int(mask[0, token_id // 32]) >> (token_id % 32) & 1)
            self.assertEqual(allowed, token_id != special("message_model"))
            if allowed:
                grammar.accept_token(token_id)
        self.assertTrue(grammar.is_terminated())

        results = self._reasoning_and_content(
            request, output_ids, {"type": "stop", "matched": eos}, split=1
        )
        for mode, result in results.items():
            with self.subTest(mode=mode):
                self.assertEqual(result, ("", "yes"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
