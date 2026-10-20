import json
import unittest
from pathlib import Path

from sglang.srt.entrypoints.openai.chat_encoding import encode_simple_chat
from sglang.srt.parser.inkling_output import InklingOutputDelta, InklingOutputParser
from sglang.srt.parser.inkling_renderer import (
    load_tml_renderers,
    render_inkling_assistant_prefix,
    render_inkling_messages,
)
from sglang.srt.parser.inkling_tokenizer import (
    AUDIO_END,
    AUDIO_TOKEN_ID,
    CONTENT_AUDIO_INPUT,
    END_MESSAGE,
    INKLING_SPECIAL_TOKEN_IDS,
    MESSAGE_MODEL,
    MESSAGE_USER,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

_GOLDEN = json.loads(
    (Path(__file__).with_name("inkling_tmlv0_golden.json")).read_text()
)


class TestInklingRenderer(unittest.TestCase):
    def test_prompts_match_tmlv0_reference(self):
        """Reference-generated prompt IDs pin the bridge conventions
        recorded in the fixture's source."""
        for case in _GOLDEN["prompts"]:
            with self.subTest(case=case["name"]):
                actual = render_inkling_messages(
                    case["messages"],
                    tools=case["tools"],
                    reasoning_effort=case["reasoning_effort"],
                )
                self.assertEqual(actual, case["input_ids"])

    def test_reasoning_content_cannot_mix_with_thinking_parts(self):
        """Two sources of prior thinking have no defined order, so the request is
        rejected rather than rendered as two thinking messages."""
        with self.assertRaisesRegex(ValueError, "cannot mix reasoning_content"):
            render_inkling_messages(
                [
                    {"role": "user", "content": "x"},
                    {
                        "role": "assistant",
                        "reasoning_content": "plan",
                        "content": [{"type": "thinking", "thinking": "plan"}],
                    },
                ]
            )

    def test_audio_part_keeps_mm_processor_framing(self):
        """The MM processor expands one AUDIO_TOKEN_ID inside the audio input envelope;
        tml-renderers cannot render audio without DMel bytes, so this envelope is sglang's."""
        actual = render_inkling_messages(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {"data": "", "format": "wav"},
                        }
                    ],
                }
            ]
        )
        self.assertEqual(
            actual[-5:],
            [
                INKLING_SPECIAL_TOKEN_IDS[MESSAGE_USER],
                INKLING_SPECIAL_TOKEN_IDS[CONTENT_AUDIO_INPUT],
                AUDIO_TOKEN_ID,
                INKLING_SPECIAL_TOKEN_IDS[AUDIO_END],
                INKLING_SPECIAL_TOKEN_IDS[END_MESSAGE],
            ],
        )

    def test_audio_outside_user_messages_is_rejected(self):
        """tml-renderers supports audio only as user input; tool or assistant audio
        must fail rather than render a framing the model never saw."""
        audio = {"type": "input_audio", "input_audio": {"data": "", "format": "wav"}}
        for message in (
            {"role": "tool", "tool_call_id": "c", "name": "rec", "content": [audio]},
            {"role": "assistant", "content": [audio]},
        ):
            with self.subTest(role=message["role"]):
                with self.assertRaises(ValueError):
                    render_inkling_messages([{"role": "user", "content": "x"}, message])

    def test_assistant_prefix_text_is_ordinary_tokens(self):
        tokenizer = load_tml_renderers().tokenizer
        prefix = "The answer <|end_message|>"
        self.assertEqual(
            render_inkling_assistant_prefix(prefix),
            [
                INKLING_SPECIAL_TOKEN_IDS[MESSAGE_MODEL],
                INKLING_SPECIAL_TOKEN_IDS["<|content_text|>"],
                *tokenizer.encode_ordinary(prefix),
            ],
        )

    def test_offline_encoder_uses_the_same_inkling_format(self):
        messages = [{"role": "user", "content": "hi"}]
        self.assertEqual(
            encode_simple_chat(tokenizer=None, spec="inkling", messages=messages),
            render_inkling_messages(messages),
        )


def _ids(*parts: str | list[int]) -> list[int]:
    tokenizer = load_tml_renderers().tokenizer
    ids: list[int] = []
    for part in parts:
        if isinstance(part, list):
            ids.extend(part)
        elif part.startswith("<|") and part.endswith("|>"):
            ids.append(tokenizer.encode_special(part[2:-2]))
        else:
            ids.extend(tokenizer.encode_ordinary(part))
    return ids


def _batch(token_ids: list[int], **kwargs):
    parser = InklingOutputParser(**kwargs)
    return parser.feed(token_ids).merge(parser.finish())


def _stream(token_ids: list[int], **kwargs):
    parser = InklingOutputParser(**kwargs)
    deltas = [parser.feed([token_id]) for token_id in token_ids]
    deltas.append(parser.finish())
    merged = deltas[0]
    for delta in deltas[1:]:
        merged = merged.merge(delta)
    return merged


class TestInklingOutputParser(unittest.TestCase):
    def test_outputs_match_tmlv0_reference(self):
        """The fixture uses reference-parser messages and sampled-JSON arguments;
        batch and per-token parsing must match it."""
        for case in _GOLDEN["outputs"]:
            for mode, parse in (
                ("batch", _batch),
                ("stream", _stream),
            ):
                with self.subTest(case=case["name"], mode=mode):
                    parsed = parse(
                        case["output_ids"],
                        separate_reasoning=True,
                        parse_tool_calls=True,
                    )
                    self.assertEqual(parsed.reasoning, case["reasoning"])
                    self.assertEqual(parsed.content, case["content"])
                    self.assertEqual(
                        [
                            {"name": call.name, "arguments": call.arguments}
                            for call in parsed.tool_calls
                        ],
                        case["tool_calls"],
                    )
                    self.assertEqual(
                        [call.index for call in parsed.tool_calls],
                        list(range(len(case["tool_calls"]))),
                    )

    def test_unparseable_call_becomes_payload_text(self):
        """A call payload the reference parser rejects (NaN, missing args) is not a tool call
        and does not leak its header name; parsing resumes at the next message."""
        for payload in ('{"name":"f","args":{"a":NaN}}', '{"name":"f"}'):
            output_ids = _ids(
                "<|message_model|>",
                "f",
                "<|content_invoke_tool_json|>",
                payload,
                "<|end_message|>",
                "<|message_model|>",
                "<|content_text|>",
                "after",
                "<|end_message|>",
                "<|content_model_end_sampling|>",
            )
            for mode, parse in (
                ("batch", _batch),
                ("stream", _stream),
            ):
                with self.subTest(payload=payload, mode=mode):
                    parsed = parse(
                        output_ids, separate_reasoning=True, parse_tool_calls=True
                    )
                    self.assertEqual(parsed.tool_calls, ())
                    self.assertEqual(parsed.content, payload + "after")

    def test_truncated_message_surfaces_unstreamed_payload(self):
        """A message cut off before <|end_message|> (max_tokens) keeps what the
        model generated: a call payload becomes content, streamed text is not repeated."""
        cases = [
            (
                (
                    "<|message_model|>",
                    "get_weather",
                    "<|content_invoke_tool_json|>",
                    '{"name":"get_weather","args":{"city":"San Fr',
                ),
                '{"name":"get_weather","args":{"city":"San Fr',
            ),
            (("<|message_model|>", "<|content_text|>", "partial"), "partial"),
        ]
        for pieces, expected in cases:
            for mode, parse in (("batch", _batch), ("stream", _stream)):
                with self.subTest(expected=expected, mode=mode):
                    parsed = parse(
                        _ids(*pieces), separate_reasoning=True, parse_tool_calls=True
                    )
                    self.assertEqual(parsed.tool_calls, ())
                    self.assertEqual(parsed.content, expected)

    def test_call_arguments_keep_the_sampled_json(self):
        """Arguments keep the model's key order, raw UTF-8, integers wider than 64
        bits and specials spelled inside strings, as json.dumps of the sampled args."""
        wide = _ids(
            "<|message_model|>",
            "f",
            "<|content_invoke_tool_json|>",
            '{"name":"f","args":{"z":1,"big":18446744073709551616,"c":"北京"}}',
            "<|end_message|>",
        )
        special_in_string = _ids(
            "<|message_model|>",
            "f",
            "<|content_invoke_tool_json|>",
            '{"name":"f","args":{"q":"a',
            "<|content_text|>",
            'b"}}',
            "<|end_message|>",
        )
        cases = [
            (wide, '{"z": 1, "big": 18446744073709551616, "c": "北京"}'),
            (special_in_string, '{"q": "a<|content_text|>b"}'),
        ]
        for token_ids, expected in cases:
            for mode, parse in (("batch", _batch), ("stream", _stream)):
                with self.subTest(expected=expected, mode=mode):
                    parsed = parse(
                        token_ids, separate_reasoning=True, parse_tool_calls=True
                    )
                    self.assertEqual(
                        [call.arguments for call in parsed.tool_calls], [expected]
                    )

    def test_finish_flushes_once(self):
        """A finish_reason stays set on later stream chunks; finishing again must
        not re-emit a truncated message's payload."""
        parser = InklingOutputParser(separate_reasoning=True, parse_tool_calls=True)
        first = parser.finish(
            _ids("<|message_model|>", "f", "<|content_invoke_tool_json|>", '{"a":')
        )
        self.assertEqual(first.content, '{"a":')
        self.assertEqual(parser.finish(), InklingOutputDelta())

    def test_end_message_stop_still_closes_the_call(self):
        """With <|end_message|> among the stop tokens, the matched stop must stay in
        the stream: it is what closes the message into a tool call."""
        end_message = load_tml_renderers().tokenizer.encode_special("end_message")
        token_ids = _ids(
            "<|message_model|>",
            "f",
            "<|content_invoke_tool_json|>",
            '{"name":"f","args":{"a":1}}',
            "<|end_message|>",
        )
        parsed = InklingOutputParser(
            separate_reasoning=True, parse_tool_calls=True
        ).finish(token_ids, matched_stop=end_message)
        self.assertEqual([call.name for call in parsed.tool_calls], ["f"])
        self.assertEqual(parsed.content, "")

    def test_buffered_reasoning_survives_truncation(self):
        """With stream_reasoning=False a thinking block is held until it
        closes; a max_tokens cut inside it must still flush the held text."""
        parser = InklingOutputParser(
            separate_reasoning=True, parse_tool_calls=True, stream_reasoning=False
        )
        held = parser.feed(
            _ids("<|message_model|>", "<|content_thinking|>", "long plan here")
        )
        self.assertEqual(held.reasoning, "")
        self.assertEqual(parser.finish().reasoning, "long plan here")

    def test_disabled_parsers_route_into_content(self):
        output_ids = _ids(
            "<|message_model|>",
            "<|content_thinking|>",
            "plan",
            "<|end_message|>",
            "<|message_model|>",
            "f",
            "<|content_invoke_tool_json|>",
            '{"name":"f","args":{}}',
            "<|end_message|>",
            "<|content_model_end_sampling|>",
        )
        parsed = _batch(output_ids, separate_reasoning=False, parse_tool_calls=False)
        self.assertEqual(parsed.reasoning, "")
        self.assertEqual(parsed.tool_calls, ())
        self.assertEqual(parsed.content, 'plan{"name":"f","args":{}}')


if __name__ == "__main__":
    unittest.main()
