import unittest

from sglang.srt.entrypoints.openai.chat_encoding import encode_simple_chat
from sglang.srt.parser.inkling_renderer import render_inkling_messages
from sglang.srt.parser.inkling_tokenizer import (
    CONTENT_INVOKE_TOOL_JSON,
    CONTENT_MODEL_END_SAMPLING,
    CONTENT_TEXT,
    CONTENT_THINKING,
    CONTENT_XML,
    END_MESSAGE,
    INKLING_SPECIAL_TOKEN_IDS,
    MESSAGE_MODEL,
    MESSAGE_SYSTEM,
    MESSAGE_USER,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _text(value: str) -> list[int]:
    return list(value.encode())


class _InklingTokenizer:
    def encode_special(self, token: str) -> int:
        return INKLING_SPECIAL_TOKEN_IDS[token]

    def encode_text(self, text: str) -> list[int]:
        return _text(text)


class _BaseTokenizer:
    chat_template = None

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return _text(text)


def _block(role: str, kind: str, payload: str, author: str = "") -> list[int]:
    return (
        [INKLING_SPECIAL_TOKEN_IDS[role]]
        + _text(author)
        + [
            INKLING_SPECIAL_TOKEN_IDS[kind],
            *_text(payload),
            INKLING_SPECIAL_TOKEN_IDS[END_MESSAGE],
        ]
    )


class TestInklingRenderer(unittest.TestCase):
    def setUp(self):
        self.tokenizer = _InklingTokenizer()

    def test_generation_prompt_is_not_prefilled(self):
        actual = render_inkling_messages(
            [{"role": "user", "content": "hello"}], self.tokenizer
        )
        self.assertEqual(
            actual,
            _block(MESSAGE_SYSTEM, CONTENT_TEXT, "Thinking effort level: 0.9")
            + _block(MESSAGE_USER, CONTENT_TEXT, "hello"),
        )
        self.assertNotEqual(actual[-1], INKLING_SPECIAL_TOKEN_IDS[MESSAGE_MODEL])

    def test_tool_system_and_effort_have_canonical_prefix_order(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "weather",
                    "description": "Lookup weather",
                    "parameters": {"type": "object"},
                },
            }
        ]
        actual = render_inkling_messages(
            [
                {"role": "system", "content": "original"},
                {"role": "user", "content": "question"},
            ],
            self.tokenizer,
            tools=tools,
            reasoning_effort=0.8764,
        )
        tool_json = (
            '[{"description":"Lookup weather","name":"weather",'
            '"parameters":{"type":"object"},"type":"function"}]'
        )
        expected = (
            _block(
                MESSAGE_SYSTEM,
                CONTENT_XML,
                tool_json,
                author="tool_declare",
            )
            + _block(MESSAGE_SYSTEM, CONTENT_TEXT, "original")
            + _block(MESSAGE_SYSTEM, CONTENT_TEXT, "Thinking effort level: 0.88")
            + _block(MESSAGE_USER, CONTENT_TEXT, "question")
        )
        self.assertEqual(actual, expected)

    def test_multiturn_conversation_has_one_fixed_effort_directive(self):
        system = {"role": "system", "content": "system"}
        user1 = {"role": "user", "content": "user1"}
        assistant1 = {"role": "assistant", "content": "assistant1"}
        user2 = {"role": "user", "content": "user2"}
        prefix = (
            _block(MESSAGE_SYSTEM, CONTENT_TEXT, "system")
            + _block(MESSAGE_SYSTEM, CONTENT_TEXT, "Thinking effort level: 0.2")
            + _block(MESSAGE_USER, CONTENT_TEXT, "user1")
        )

        turn1 = render_inkling_messages(
            [system, user1], self.tokenizer, reasoning_effort=0.2
        )
        turn2 = render_inkling_messages(
            [system, user1, assistant1, user2],
            self.tokenizer,
            reasoning_effort=0.2,
        )

        self.assertEqual(turn1, prefix)
        self.assertEqual(
            turn2,
            prefix
            + _block(MESSAGE_MODEL, CONTENT_TEXT, "assistant1")
            + [INKLING_SPECIAL_TOKEN_IDS[CONTENT_MODEL_END_SAMPLING]]
            + _block(MESSAGE_USER, CONTENT_TEXT, "user2"),
        )

    def test_historical_assistant_preserves_parts_and_ends_sampling(self):
        actual = render_inkling_messages(
            [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "first"},
                        {"type": "text", "text": "visible"},
                        {"type": "reasoning", "text": "second"},
                    ],
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "function": {
                                "name": "weather",
                                "arguments": '{"city":"SF"}',
                            },
                        }
                    ],
                }
            ],
            self.tokenizer,
        )
        expected = (
            _block(MESSAGE_SYSTEM, CONTENT_TEXT, "Thinking effort level: 0.9")
            + _block(MESSAGE_MODEL, CONTENT_THINKING, "first")
            + _block(MESSAGE_MODEL, CONTENT_TEXT, "visible")
            + _block(MESSAGE_MODEL, CONTENT_THINKING, "second")
            + _block(
                MESSAGE_MODEL,
                CONTENT_INVOKE_TOOL_JSON,
                '{"name":"weather","args":{"city":"SF"}}',
                author="weather",
            )
            + [INKLING_SPECIAL_TOKEN_IDS[CONTENT_MODEL_END_SAMPLING]]
        )
        self.assertEqual(actual, expected)

    def test_empty_assistant_message_does_not_emit_bare_terminator(self):
        """Bug regression: an assistant message that renders zero blocks
        (content None, no reasoning, no tool calls) appended a bare
        <|content_model_end_sampling|> with no preceding model block —
        injecting a malformed turn terminator into the prompt."""
        actual = render_inkling_messages(
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": None},
                {"role": "user", "content": "again"},
            ],
            self.tokenizer,
        )
        self.assertNotIn(INKLING_SPECIAL_TOKEN_IDS[CONTENT_MODEL_END_SAMPLING], actual)

    def test_reasoning_content_cannot_reorder_thinking_parts(self):
        with self.assertRaisesRegex(ValueError, "cannot mix"):
            render_inkling_messages(
                [
                    {
                        "role": "assistant",
                        "reasoning_content": "legacy",
                        "content": [{"type": "thinking", "thinking": "ordered"}],
                    }
                ],
                self.tokenizer,
            )

    def test_reasoning_effort_is_two_decimal_quantized_and_validated(self):
        for value, expected in ((0.8766, "0.88"), (0.0, "0"), (0.99, "0.99")):
            with self.subTest(value=value):
                actual = render_inkling_messages(
                    [{"role": "user", "content": "q"}],
                    self.tokenizer,
                    reasoning_effort=value,
                )
                directive = _block(
                    MESSAGE_SYSTEM,
                    CONTENT_TEXT,
                    f"Thinking effort level: {expected}",
                )
                self.assertEqual(actual[: len(directive)], directive)
        for value in (-0.1, 1.0, 1.1, float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                render_inkling_messages(
                    [{"role": "user", "content": "q"}],
                    self.tokenizer,
                    reasoning_effort=value,
                )

    def test_offline_encoder_uses_the_same_inkling_format(self):
        actual = encode_simple_chat(
            tokenizer=_BaseTokenizer(),
            spec="inkling",
            messages=[{"role": "user", "content": "hello"}],
        )
        self.assertEqual(
            actual,
            _block(MESSAGE_SYSTEM, CONTENT_TEXT, "Thinking effort level: 0.9")
            + _block(MESSAGE_USER, CONTENT_TEXT, "hello"),
        )


if __name__ == "__main__":
    unittest.main()
