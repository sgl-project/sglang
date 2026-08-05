"""Unit tests for the chat-encoding dispatch helpers."""

from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.entrypoints.openai.chat_encoding import (
    encode_simple_chat,
    resolve_chat_encoding_spec,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class RecordingTokenizer:
    def __init__(self, *, chat_template=None, name_or_path="test-model"):
        self.chat_template = chat_template
        self.name_or_path = name_or_path
        self.encoded_text = None
        self.applied_messages = None

    def encode(self, text):
        self.encoded_text = text
        return [11, 22]

    def apply_chat_template(self, messages, **kwargs):
        self.applied_messages = (messages, kwargs)
        return [33, 44]


class TestResolveChatEncodingSpec(CustomTestCase):
    def test_explicit_tool_call_parser_takes_precedence(self):
        config = SimpleNamespace(architectures=["DeepseekV4ForCausalLM"])
        tokenizer = SimpleNamespace(chat_template="template")

        self.assertEqual(
            resolve_chat_encoding_spec(
                hf_config=config,
                tokenizer=tokenizer,
                tool_call_parser="deepseekv4",
            ),
            "dsv4",
        )
        self.assertEqual(
            resolve_chat_encoding_spec(
                hf_config=config,
                tokenizer=tokenizer,
                tool_call_parser="deepseekv32",
            ),
            "dsv32",
        )

    def test_deepseek_v4_architecture_uses_dsv4(self):
        config = SimpleNamespace(architectures=["DeepseekV4ForCausalLM"])
        tokenizer = SimpleNamespace(chat_template="template")

        self.assertEqual(
            resolve_chat_encoding_spec(hf_config=config, tokenizer=tokenizer),
            "dsv4",
        )

    def test_deepseek_v3_without_chat_template_uses_dsv32(self):
        config = SimpleNamespace(architectures=["DeepseekV3ForCausalLM"])
        tokenizer = SimpleNamespace(chat_template=None)

        self.assertEqual(
            resolve_chat_encoding_spec(hf_config=config, tokenizer=tokenizer),
            "dsv32",
        )

    def test_deepseek_v3_with_chat_template_uses_default_path(self):
        config = SimpleNamespace(architectures=["DeepseekV3ForCausalLM"])
        tokenizer = SimpleNamespace(chat_template="template")

        self.assertIsNone(
            resolve_chat_encoding_spec(hf_config=config, tokenizer=tokenizer)
        )

    def test_unrelated_architecture_uses_default_path(self):
        config = SimpleNamespace(architectures=["LlamaForCausalLM"])
        tokenizer = SimpleNamespace(chat_template=None)

        self.assertIsNone(
            resolve_chat_encoding_spec(hf_config=config, tokenizer=tokenizer)
        )

    def test_missing_architecture_uses_default_path(self):
        config = SimpleNamespace(architectures=[])
        tokenizer = SimpleNamespace(chat_template=None)

        self.assertIsNone(
            resolve_chat_encoding_spec(hf_config=config, tokenizer=tokenizer)
        )

    def test_none_config_uses_default_path(self):
        self.assertIsNone(
            resolve_chat_encoding_spec(
                hf_config=None,
                tokenizer=SimpleNamespace(chat_template=None),
            )
        )

    def test_dict_config_uses_architecture(self):
        self.assertEqual(
            resolve_chat_encoding_spec(
                hf_config={"architectures": ["DeepseekV4ForCausalLM"]},
                tokenizer=SimpleNamespace(chat_template="template"),
            ),
            "dsv4",
        )

    def test_config_without_architectures_uses_default_path(self):
        self.assertIsNone(
            resolve_chat_encoding_spec(
                hf_config=SimpleNamespace(),
                tokenizer=SimpleNamespace(chat_template=None),
            )
        )


class TestEncodeSimpleChat(CustomTestCase):
    def test_custom_encoder_prepends_empty_system_message(self):
        tokenizer = RecordingTokenizer()
        messages = [{"role": "user", "content": "hello"}]

        with patch(
            "sglang.srt.entrypoints.openai.encoding_dsv4.encode_messages",
            return_value="encoded-dsv4",
        ) as encode_messages:
            result = encode_simple_chat(
                tokenizer=tokenizer,
                spec="dsv4",
                messages=messages,
                thinking_mode="thinking",
            )

        self.assertEqual(result, [11, 22])
        encode_messages.assert_called_once_with(
            [
                {"role": "system", "content": ""},
                {"role": "user", "content": "hello"},
            ],
            thinking_mode="thinking",
        )
        self.assertEqual(messages, [{"role": "user", "content": "hello"}])

    def test_custom_encoder_preserves_existing_system_message(self):
        tokenizer = RecordingTokenizer()
        messages = [
            {"role": "system", "content": "be concise"},
            {"role": "user", "content": "hello"},
        ]

        with patch(
            "sglang.srt.entrypoints.openai.encoding_dsv32.encode_messages",
            return_value="encoded-dsv32",
        ) as encode_messages:
            result = encode_simple_chat(
                tokenizer=tokenizer,
                spec="dsv32",
                messages=messages,
            )

        self.assertEqual(result, [11, 22])
        encode_messages.assert_called_once_with(messages, thinking_mode="chat")

    def test_default_path_uses_tokenizer_chat_template(self):
        tokenizer = RecordingTokenizer(chat_template="template")
        messages = [{"role": "user", "content": "hello"}]

        result = encode_simple_chat(tokenizer=tokenizer, spec=None, messages=messages)

        self.assertEqual(result, [33, 44])
        self.assertEqual(
            tokenizer.applied_messages,
            (
                messages,
                {
                    "add_generation_prompt": True,
                    "tokenize": True,
                },
            ),
        )
        self.assertIsNone(tokenizer.encoded_text)

    def test_default_path_without_chat_template_explains_failure(self):
        tokenizer = RecordingTokenizer(chat_template=None, name_or_path="my-model")

        with self.assertRaisesRegex(ValueError, "my-model"):
            encode_simple_chat(
                tokenizer=tokenizer,
                spec=None,
                messages=[{"role": "user", "content": "hello"}],
            )


if __name__ == "__main__":
    import unittest

    unittest.main()
