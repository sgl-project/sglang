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
"""Tests for OpenAI API protocol models"""

import unittest
from typing import List, Optional

from pydantic import BaseModel, Field, ValidationError

from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
    CompletionRequest,
    ModelCard,
    ModelList,
    UsageInfo,
)


class TestModelCard(unittest.TestCase):
    """Test ModelCard protocol model"""

    def test_model_card_serialization(self):
        """Test model card JSON serialization"""
        card = ModelCard(id="test-model", max_model_len=4096)
        data = card.model_dump()
        self.assertEqual(data["id"], "test-model")
        self.assertEqual(data["object"], "model")
        self.assertEqual(data["max_model_len"], 4096)


class TestModelList(unittest.TestCase):
    """Test ModelList protocol model"""

    def test_empty_model_list(self):
        """Test empty model list creation"""
        model_list = ModelList()
        self.assertEqual(model_list.object, "list")
        self.assertEqual(len(model_list.data), 0)

    def test_model_list_with_cards(self):
        """Test model list with model cards"""
        cards = [
            ModelCard(id="model-1"),
            ModelCard(id="model-2", max_model_len=2048),
        ]
        model_list = ModelList(data=cards)
        self.assertEqual(len(model_list.data), 2)
        self.assertEqual(model_list.data[0].id, "model-1")
        self.assertEqual(model_list.data[1].id, "model-2")


class TestCompletionRequest(unittest.TestCase):
    """Test CompletionRequest protocol model"""

    def test_basic_completion_request(self):
        """Test basic completion request"""
        request = CompletionRequest(model="test-model", prompt="Hello world")
        self.assertEqual(request.model, "test-model")
        self.assertEqual(request.prompt, "Hello world")
        self.assertEqual(request.max_tokens, 16)  # default
        self.assertEqual(request.temperature, 1.0)  # default
        self.assertEqual(request.n, 1)  # default
        self.assertFalse(request.stream)  # default
        self.assertFalse(request.echo)  # default

    def test_completion_request_sglang_extensions(self):
        """Test completion request with SGLang-specific extensions"""
        request = CompletionRequest(
            model="test-model",
            prompt="Hello",
            top_k=50,
            min_p=0.1,
            repetition_penalty=1.1,
            regex=r"\d+",
            json_schema='{"type": "object"}',
            lora_path="/path/to/lora",
        )
        self.assertEqual(request.top_k, 50)
        self.assertEqual(request.min_p, 0.1)
        self.assertEqual(request.repetition_penalty, 1.1)
        self.assertEqual(request.regex, r"\d+")
        self.assertEqual(request.json_schema, '{"type": "object"}')
        self.assertEqual(request.lora_path, "/path/to/lora")

    def test_completion_request_validation_errors(self):
        """Test completion request validation errors"""
        with self.assertRaises(ValidationError):
            CompletionRequest()  # missing required fields

        with self.assertRaises(ValidationError):
            CompletionRequest(model="test-model")  # missing prompt


class TestChatCompletionRequest(unittest.TestCase):
    """Test ChatCompletionRequest protocol model"""

    def test_basic_chat_completion_request(self):
        """Test basic chat completion request"""
        messages = [{"role": "user", "content": "Hello"}]
        request = ChatCompletionRequest(model="test-model", messages=messages)
        self.assertEqual(request.model, "test-model")
        self.assertEqual(len(request.messages), 1)
        self.assertEqual(request.messages[0].role, "user")
        self.assertEqual(request.messages[0].content, "Hello")
        self.assertEqual(request.temperature, None)  # default
        self.assertFalse(request.stream)  # default
        self.assertEqual(request.tool_choice, "none")  # default when no tools

    def test_sampling_param_build(self):
        req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi"}],
            temperature=0.8,
            max_tokens=150,
            min_tokens=5,
            top_p=0.9,
            stop=["</s>"],
        )
        params = req.to_sampling_params(["</s>"], {}, None)
        self.assertEqual(params["temperature"], 0.8)
        self.assertEqual(params["max_new_tokens"], 150)
        self.assertEqual(params["min_new_tokens"], 5)
        self.assertEqual(params["stop"], ["</s>"])

    def test_chat_completion_tool_choice_validation(self):
        """Test tool choice validation logic"""
        messages = [{"role": "user", "content": "Hello"}]

        # No tools, tool_choice should default to "none"
        request1 = ChatCompletionRequest(model="test-model", messages=messages)
        self.assertEqual(request1.tool_choice, "none")

        # With tools, tool_choice should default to "auto"
        tools = [
            {
                "type": "function",
                "function": {"name": "test_func", "description": "Test function"},
            }
        ]
        request2 = ChatCompletionRequest(
            model="test-model", messages=messages, tools=tools
        )
        self.assertEqual(request2.tool_choice, "auto")

    def test_chat_completion_sglang_extensions(self):
        """Test chat completion with SGLang extensions"""
        messages = [{"role": "user", "content": "Hello"}]
        request = ChatCompletionRequest(
            model="test-model",
            messages=messages,
            top_k=40,
            min_p=0.05,
            separate_reasoning=False,
            stream_reasoning=False,
            chat_template_kwargs={"custom_param": "value"},
        )
        self.assertEqual(request.top_k, 40)
        self.assertEqual(request.min_p, 0.05)
        self.assertFalse(request.separate_reasoning)
        self.assertFalse(request.stream_reasoning)
        self.assertEqual(request.chat_template_kwargs, {"custom_param": "value"})

    def test_chat_completion_reasoning_effort(self):
        """Test chat completion with reasoning effort"""
        messages = [{"role": "user", "content": "Hello"}]
        request = ChatCompletionRequest(
            model="test-model",
            messages=messages,
            reasoning={
                "enabled": True,
                "reasoning_effort": "high",
            },
        )
        self.assertEqual(request.reasoning_effort, "high")
        self.assertEqual(request.chat_template_kwargs, {"thinking": True})

    def test_chat_completion_json_format(self):
        """Test chat completion json format"""
        transcript = "Good morning! It's 7:00 AM, and I'm just waking up. Today is going to be a busy day, "
        "so let's get started. First, I need to make a quick breakfast. I think I'll have some "
        "scrambled eggs and toast with a cup of coffee. While I'm cooking, I'll also check my "
        "emails to see if there's anything urgent."

        messages = [
            {
                "role": "system",
                "content": "The following is a voice message transcript. Only answer in JSON.",
            },
            {
                "role": "user",
                "content": transcript,
            },
        ]

        class VoiceNote(BaseModel):
            title: str = Field(description="A title for the voice note")
            summary: str = Field(
                description="A short one sentence summary of the voice note."
            )
            strict: Optional[bool] = True
            actionItems: List[str] = Field(
                description="A list of action items from the voice note"
            )

        request = ChatCompletionRequest(
            model="test-model",
            messages=messages,
            top_k=40,
            min_p=0.05,
            separate_reasoning=False,
            stream_reasoning=False,
            chat_template_kwargs={"custom_param": "value"},
            response_format={
                "type": "json_schema",
                "schema": VoiceNote.model_json_schema(),
            },
        )
        res_format = request.response_format
        json_format = res_format.json_schema
        name = json_format.name
        schema = json_format.schema_
        strict = json_format.strict
        self.assertEqual(name, "VoiceNote")
        self.assertEqual(strict, True)
        self.assertNotIn("strict", schema["properties"])

        request = ChatCompletionRequest(
            model="test-model",
            messages=messages,
            top_k=40,
            min_p=0.05,
            separate_reasoning=False,
            stream_reasoning=False,
            chat_template_kwargs={"custom_param": "value"},
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "VoiceNote",
                    "schema": VoiceNote.model_json_schema(),
                    "strict": True,
                },
            },
        )
        res_format = request.response_format
        json_format = res_format.json_schema
        name = json_format.name
        schema = json_format.schema_
        strict = json_format.strict
        self.assertEqual(name, "VoiceNote")
        self.assertEqual(strict, True)


class TestModelSerialization(unittest.TestCase):
    """Test model serialization with hidden states"""

    def test_hidden_states_excluded_when_none(self):
        """Test that None hidden_states are excluded with exclude_none=True"""
        choice = ChatCompletionResponseChoice(
            index=0,
            message=ChatMessage(role="assistant", content="Hello"),
            finish_reason="stop",
            hidden_states=None,
        )

        response = ChatCompletionResponse(
            id="test-id",
            model="test-model",
            choices=[choice],
            usage=UsageInfo(prompt_tokens=5, completion_tokens=1, total_tokens=6),
        )

        # Test exclude_none serialization (should exclude None hidden_states)
        data = response.model_dump(exclude_none=True)
        self.assertNotIn("hidden_states", data["choices"][0])

    def test_hidden_states_included_when_not_none(self):
        """Test that non-None hidden_states are included"""
        choice = ChatCompletionResponseChoice(
            index=0,
            message=ChatMessage(role="assistant", content="Hello"),
            finish_reason="stop",
            hidden_states=[0.1, 0.2, 0.3],
        )

        response = ChatCompletionResponse(
            id="test-id",
            model="test-model",
            choices=[choice],
            usage=UsageInfo(prompt_tokens=5, completion_tokens=1, total_tokens=6),
        )

        # Test exclude_none serialization (should include non-None hidden_states)
        data = response.model_dump(exclude_none=True)
        self.assertIn("hidden_states", data["choices"][0])
        self.assertEqual(data["choices"][0]["hidden_states"], [0.1, 0.2, 0.3])


class TestValidationEdgeCases(unittest.TestCase):
    """Test edge cases and validation scenarios"""

    def test_invalid_tool_choice_type(self):
        """Test invalid tool choice type"""
        messages = [{"role": "user", "content": "Hello"}]
        with self.assertRaises(ValidationError):
            ChatCompletionRequest(
                model="test-model", messages=messages, tool_choice=123
            )

    def test_negative_token_limits(self):
        """Test negative token limits"""
        with self.assertRaises(ValidationError):
            CompletionRequest(model="test-model", prompt="Hello", max_tokens=-1)

    def test_model_serialization_roundtrip(self):
        """Test that models can be serialized and deserialized"""
        original_request = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.7,
            max_tokens=100,
        )

        # Serialize to dict
        data = original_request.model_dump()

        # Deserialize back
        restored_request = ChatCompletionRequest(**data)

        self.assertEqual(restored_request.model, original_request.model)
        self.assertEqual(restored_request.temperature, original_request.temperature)
        self.assertEqual(restored_request.max_tokens, original_request.max_tokens)
        self.assertEqual(len(restored_request.messages), len(original_request.messages))


class TestSamplingParameterValidation(unittest.TestCase):
    """Test sampling parameter validation for OpenAI API compliance"""

    def test_completion_n_validation(self):
        """Test n parameter validation for CompletionRequest"""
        # Valid n
        request = CompletionRequest(model="test-model", prompt="Hello", n=1)
        self.assertEqual(request.n, 1)

        request = CompletionRequest(model="test-model", prompt="Hello", n=5)
        self.assertEqual(request.n, 5)

        # Invalid n (less than 1)
        with self.assertRaises(ValidationError):
            CompletionRequest(model="test-model", prompt="Hello", n=0)

        with self.assertRaises(ValidationError):
            CompletionRequest(model="test-model", prompt="Hello", n=-1)

    def test_completion_logprobs_validation(self):
        """Test logprobs parameter validation for CompletionRequest"""
        # Valid logprobs
        request = CompletionRequest(model="test-model", prompt="Hello", logprobs=0)
        self.assertEqual(request.logprobs, 0)

        request = CompletionRequest(model="test-model", prompt="Hello", logprobs=5)
        self.assertEqual(request.logprobs, 5)

        # Invalid logprobs (out of range 0-5)
        with self.assertRaises(ValidationError):
            CompletionRequest(model="test-model", prompt="Hello", logprobs=-1)

        with self.assertRaises(ValidationError):
            CompletionRequest(model="test-model", prompt="Hello", logprobs=6)

    def test_completion_temperature_validation(self):
        """Test temperature parameter validation for CompletionRequest"""
        # Valid temperature
        request = CompletionRequest(model="test-model", prompt="Hello", temperature=0)
        self.assertEqual(request.temperature, 0)

        request = CompletionRequest(model="test-model", prompt="Hello", temperature=2.0)
        self.assertEqual(request.temperature, 2.0)

        # Invalid temperature (negative)
        with self.assertRaises(ValidationError):
            CompletionRequest(model="test-model", prompt="Hello", temperature=-0.1)

    def test_completion_top_p_validation(self):
        """Test top_p parameter validation for CompletionRequest"""
        # Valid top_p
        request = CompletionRequest(model="test-model", prompt="Hello", top_p=0)
        self.assertEqual(request.top_p, 0)

        request = CompletionRequest(model="test-model", prompt="Hello", top_p=1.0)
        self.assertEqual(request.top_p, 1.0)

        request = CompletionRequest(model="test-model", prompt="Hello", top_p=0.5)
        self.assertEqual(request.top_p, 0.5)

        # Invalid top_p (out of range 0-1)
        with self.assertRaises(ValidationError):
            CompletionRequest(model="test-model", prompt="Hello", top_p=-0.1)

        with self.assertRaises(ValidationError):
            CompletionRequest(model="test-model", prompt="Hello", top_p=1.1)

    def test_completion_frequency_penalty_validation(self):
        """Test frequency_penalty parameter validation for CompletionRequest"""
        # Valid frequency_penalty
        request = CompletionRequest(
            model="test-model", prompt="Hello", frequency_penalty=-2.0
        )
        self.assertEqual(request.frequency_penalty, -2.0)

        request = CompletionRequest(
            model="test-model", prompt="Hello", frequency_penalty=2.0
        )
        self.assertEqual(request.frequency_penalty, 2.0)

        request = CompletionRequest(
            model="test-model", prompt="Hello", frequency_penalty=0
        )
        self.assertEqual(request.frequency_penalty, 0)

        # Invalid frequency_penalty (out of range -2.0 to 2.0)
        with self.assertRaises(ValidationError):
            CompletionRequest(
                model="test-model", prompt="Hello", frequency_penalty=-2.1
            )

        with self.assertRaises(ValidationError):
            CompletionRequest(model="test-model", prompt="Hello", frequency_penalty=2.1)

    def test_completion_presence_penalty_validation(self):
        """Test presence_penalty parameter validation for CompletionRequest"""
        # Valid presence_penalty
        request = CompletionRequest(
            model="test-model", prompt="Hello", presence_penalty=-2.0
        )
        self.assertEqual(request.presence_penalty, -2.0)

        request = CompletionRequest(
            model="test-model", prompt="Hello", presence_penalty=2.0
        )
        self.assertEqual(request.presence_penalty, 2.0)

        # Invalid presence_penalty (out of range -2.0 to 2.0)
        with self.assertRaises(ValidationError):
            CompletionRequest(model="test-model", prompt="Hello", presence_penalty=-2.1)

        with self.assertRaises(ValidationError):
            CompletionRequest(model="test-model", prompt="Hello", presence_penalty=2.1)

    def test_chat_completion_n_validation(self):
        """Test n parameter validation for ChatCompletionRequest"""
        messages = [{"role": "user", "content": "Hello"}]

        # Valid n
        request = ChatCompletionRequest(model="test-model", messages=messages, n=1)
        self.assertEqual(request.n, 1)

        # Invalid n
        with self.assertRaises(ValidationError):
            ChatCompletionRequest(model="test-model", messages=messages, n=0)

    def test_chat_completion_top_logprobs_validation(self):
        """Test top_logprobs parameter validation for ChatCompletionRequest"""
        messages = [{"role": "user", "content": "Hello"}]

        # Valid top_logprobs
        request = ChatCompletionRequest(
            model="test-model", messages=messages, top_logprobs=0
        )
        self.assertEqual(request.top_logprobs, 0)

        request = ChatCompletionRequest(
            model="test-model", messages=messages, top_logprobs=20
        )
        self.assertEqual(request.top_logprobs, 20)

        # Invalid top_logprobs (out of range 0-20)
        with self.assertRaises(ValidationError):
            ChatCompletionRequest(
                model="test-model", messages=messages, top_logprobs=-1
            )

        with self.assertRaises(ValidationError):
            ChatCompletionRequest(
                model="test-model", messages=messages, top_logprobs=21
            )

    def test_chat_completion_max_completion_tokens_validation(self):
        """Test max_completion_tokens parameter validation for ChatCompletionRequest"""
        messages = [{"role": "user", "content": "Hello"}]

        # Valid max_completion_tokens
        request = ChatCompletionRequest(
            model="test-model", messages=messages, max_completion_tokens=100
        )
        self.assertEqual(request.max_completion_tokens, 100)

        # Invalid max_completion_tokens (not positive)
        with self.assertRaises(ValidationError):
            ChatCompletionRequest(
                model="test-model", messages=messages, max_completion_tokens=0
            )

        with self.assertRaises(ValidationError):
            ChatCompletionRequest(
                model="test-model", messages=messages, max_completion_tokens=-1
            )

    def test_chat_completion_temperature_validation(self):
        """Test temperature parameter validation for ChatCompletionRequest"""
        messages = [{"role": "user", "content": "Hello"}]

        # Valid temperature
        request = ChatCompletionRequest(
            model="test-model", messages=messages, temperature=0
        )
        self.assertEqual(request.temperature, 0)

        request = ChatCompletionRequest(
            model="test-model", messages=messages, temperature=2.0
        )
        self.assertEqual(request.temperature, 2.0)

        # Invalid temperature (negative)
        with self.assertRaises(ValidationError):
            ChatCompletionRequest(
                model="test-model", messages=messages, temperature=-0.1
            )

    def test_chat_completion_top_p_validation(self):
        """Test top_p parameter validation for ChatCompletionRequest"""
        messages = [{"role": "user", "content": "Hello"}]

        # Valid top_p
        request = ChatCompletionRequest(
            model="test-model", messages=messages, top_p=0.5
        )
        self.assertEqual(request.top_p, 0.5)

        # Invalid top_p
        with self.assertRaises(ValidationError):
            ChatCompletionRequest(model="test-model", messages=messages, top_p=-0.1)

        with self.assertRaises(ValidationError):
            ChatCompletionRequest(model="test-model", messages=messages, top_p=1.1)

    def test_chat_completion_frequency_penalty_validation(self):
        """Test frequency_penalty parameter validation for ChatCompletionRequest"""
        messages = [{"role": "user", "content": "Hello"}]

        # Valid frequency_penalty
        request = ChatCompletionRequest(
            model="test-model", messages=messages, frequency_penalty=0.5
        )
        self.assertEqual(request.frequency_penalty, 0.5)

        # Invalid frequency_penalty
        with self.assertRaises(ValidationError):
            ChatCompletionRequest(
                model="test-model", messages=messages, frequency_penalty=-2.1
            )

        with self.assertRaises(ValidationError):
            ChatCompletionRequest(
                model="test-model", messages=messages, frequency_penalty=2.1
            )

    def test_chat_completion_presence_penalty_validation(self):
        """Test presence_penalty parameter validation for ChatCompletionRequest"""
        messages = [{"role": "user", "content": "Hello"}]

        # Valid presence_penalty
        request = ChatCompletionRequest(
            model="test-model", messages=messages, presence_penalty=0.5
        )
        self.assertEqual(request.presence_penalty, 0.5)

        # Invalid presence_penalty
        with self.assertRaises(ValidationError):
            ChatCompletionRequest(
                model="test-model", messages=messages, presence_penalty=-2.1
            )

        with self.assertRaises(ValidationError):
            ChatCompletionRequest(
                model="test-model", messages=messages, presence_penalty=2.1
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
