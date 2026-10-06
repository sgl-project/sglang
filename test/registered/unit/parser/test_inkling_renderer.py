import json
import sys
import unittest
from pathlib import Path
from unittest import mock

from sglang.srt.entrypoints.openai.chat_encoding import encode_simple_chat
from sglang.srt.parser.inkling_output import InklingOutputParser
from sglang.srt.parser.inkling_renderer import (
    TML_RENDERERS_INSTALL_HINT,
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
        """Golden input_ids were produced by tml-renderers itself (see the
        fixture's ``source``); sglang's message adaptation must not change a
        single token."""
        for case in _GOLDEN["prompts"]:
            with self.subTest(case=case["name"]):
                actual = render_inkling_messages(
                    case["messages"],
                    tools=case["tools"],
                    reasoning_effort=case["reasoning_effort"],
                )
                self.assertEqual(actual, case["input_ids"])

    def test_audio_part_keeps_mm_processor_framing(self):
        """The MM processor expands one AUDIO_TOKEN_ID inside
        <|content_audio_input|> ... <|audio_end|>; tml-renderers cannot render
        audio without DMel-encoding the bytes, so this framing is sglang's."""
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

    def test_missing_tml_renderers_names_the_package(self):
        load_tml_renderers.cache_clear()
        self.addCleanup(load_tml_renderers.cache_clear)
        with mock.patch.dict(sys.modules, {"tml_renderers": None}):
            with self.assertRaises(ImportError) as ctx:
                load_tml_renderers()
        self.assertEqual(str(ctx.exception), TML_RENDERERS_INSTALL_HINT)


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
        """Expected reasoning/content/tool_calls come from tml-renderers'
        own parser; one-shot and per-token parsing must both reproduce them."""
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
        """A call payload the reference parser rejects (NaN, missing args) must
        not surface as a tool call, and the header tool name must not leak
        into content; parsing resumes at the next message."""
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
