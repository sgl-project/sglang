# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Unit tests for chat_encoding dispatch helpers — no server, no model loading."""

import unittest
from unittest.mock import Mock, patch

from sglang.srt.entrypoints.openai.chat_encoding import (
    encode_simple_chat,
    resolve_chat_encoding_spec,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestResolveChatEncodingSpec(unittest.TestCase):
    def test_tool_call_parser_deepseekv4(self):
        self.assertEqual(
            resolve_chat_encoding_spec(
                hf_config=None, tokenizer=None, tool_call_parser="deepseekv4"
            ),
            "dsv4",
        )

    def test_tool_call_parser_deepseekv32(self):
        self.assertEqual(
            resolve_chat_encoding_spec(
                hf_config=None, tokenizer=None, tool_call_parser="deepseekv32"
            ),
            "dsv32",
        )

    def test_deepseekv4_architecture(self):
        hf_config = Mock(architectures=["DeepseekV4ForCausalLM"])
        self.assertEqual(
            resolve_chat_encoding_spec(hf_config=hf_config, tokenizer=Mock()), "dsv4"
        )

    def test_inkling_architecture_no_chat_template(self):
        hf_config = Mock(architectures=["InklingForConditionalGeneration"])
        tokenizer = Mock(chat_template=None)
        self.assertEqual(
            resolve_chat_encoding_spec(hf_config=hf_config, tokenizer=tokenizer),
            "inkling",
        )

    def test_deepseekv3_no_chat_template(self):
        hf_config = Mock(architectures=["DeepseekV3ForCausalLM"])
        tokenizer = Mock(chat_template=None)
        self.assertEqual(
            resolve_chat_encoding_spec(hf_config=hf_config, tokenizer=tokenizer),
            "dsv32",
        )

    def test_deepseekv3_with_chat_template_defaults_to_none(self):
        hf_config = Mock(architectures=["DeepseekV3ForCausalLM"])
        tokenizer = Mock(chat_template="{{ messages }}")
        self.assertIsNone(
            resolve_chat_encoding_spec(hf_config=hf_config, tokenizer=tokenizer)
        )

    def test_llama_architecture_with_chat_template_defaults_to_none(self):
        hf_config = Mock(architectures=["LlamaForCausalLM"])
        tokenizer = Mock(chat_template="{{ messages }}")
        self.assertIsNone(
            resolve_chat_encoding_spec(hf_config=hf_config, tokenizer=tokenizer)
        )

    def test_empty_architectures_defaults_to_none(self):
        hf_config = Mock(architectures=[])
        tokenizer = Mock(chat_template=None)
        self.assertIsNone(
            resolve_chat_encoding_spec(hf_config=hf_config, tokenizer=tokenizer)
        )


class TestEncodeSimpleChat(unittest.TestCase):
    def test_default_hf_chat_template(self):
        tokenizer = Mock()
        tokenizer.chat_template = "{{ messages }}"
        tokenizer.apply_chat_template.return_value = [1, 2, 3]

        messages = [{"role": "user", "content": "hello"}]
        result = encode_simple_chat(tokenizer=tokenizer, spec=None, messages=messages)

        self.assertEqual(result, [1, 2, 3])
        tokenizer.apply_chat_template.assert_called_once_with(
            messages, add_generation_prompt=True, tokenize=True
        )

    def test_no_chat_template_and_no_spec_raises(self):
        tokenizer = Mock()
        tokenizer.chat_template = None
        tokenizer.name_or_path = "test/model"

        with self.assertRaises(ValueError) as ctx:
            encode_simple_chat(
                tokenizer=tokenizer,
                spec=None,
                messages=[{"role": "user", "content": "hi"}],
            )
        self.assertIn("no HF chat template", str(ctx.exception))
        self.assertIn("test/model", str(ctx.exception))

    def test_dsv4_prepends_empty_system_message(self):
        tokenizer = Mock()
        tokenizer.encode.return_value = [10, 20]

        messages = [{"role": "user", "content": "hi"}]
        with patch(
            "sglang.srt.entrypoints.openai.encoding_dsv4.encode_messages",
            return_value="encoded_text",
        ) as mock_encode:
            result = encode_simple_chat(
                tokenizer=tokenizer,
                spec="dsv4",
                messages=messages,
                thinking_mode="chat",
            )

        self.assertEqual(result, [10, 20])
        tokenizer.encode.assert_called_once_with("encoded_text")
        mock_encode.assert_called_once()
        passed_messages = mock_encode.call_args[0][0]
        self.assertEqual(passed_messages[0]["role"], "system")
        self.assertEqual(passed_messages[0]["content"], "")
        self.assertEqual(passed_messages[1], {"role": "user", "content": "hi"})
        self.assertEqual(mock_encode.call_args.kwargs["thinking_mode"], "chat")

    def test_dsv32_uses_dsv32_encoder(self):
        tokenizer = Mock()
        tokenizer.encode.return_value = [30, 40]

        messages = [{"role": "user", "content": "hi"}]
        with patch(
            "sglang.srt.entrypoints.openai.encoding_dsv32.encode_messages",
            return_value="encoded_text",
        ) as mock_encode:
            result = encode_simple_chat(
                tokenizer=tokenizer, spec="dsv32", messages=messages
            )

        self.assertEqual(result, [30, 40])
        mock_encode.assert_called_once()
        passed_messages = mock_encode.call_args[0][0]
        self.assertEqual(passed_messages[0]["role"], "system")


if __name__ == "__main__":
    unittest.main()
