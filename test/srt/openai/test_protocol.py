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

import json
import time
import unittest
from typing import Dict, List, Optional

from pydantic import ValidationError

from sglang.srt.entrypoints.openai.protocol import (
    BatchRequest,
    BatchResponse,
    ChatCompletionMessageContentImagePart,
    ChatCompletionMessageContentTextPart,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatCompletionTokenLogprob,
    ChatMessage,
    ChoiceLogprobs,
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    DeltaMessage,
    EmbeddingObject,
    EmbeddingRequest,
    EmbeddingResponse,
    ErrorResponse,
    FileDeleteResponse,
    FileRequest,
    FileResponse,
    Function,
    FunctionResponse,
    JsonSchemaResponseFormat,
    LogProbs,
    ModelCard,
    ModelList,
    MultimodalEmbeddingInput,
    ResponseFormat,
    ScoringRequest,
    ScoringResponse,
    StreamOptions,
    StructuralTagResponseFormat,
    Tool,
    ToolCall,
    ToolChoice,
    TopLogprob,
    UsageInfo,
)


class TestModelCard(unittest.TestCase):
    """Test ModelCard protocol model"""

    def test_basic_model_card_creation(self):
        """Test basic model card creation with required fields"""
        card = ModelCard(id="test-model")
        self.assertEqual(card.id, "test-model")
        self.assertEqual(card.object, "model")
        self.assertEqual(card.owned_by, "sglang")
        self.assertIsInstance(card.created, int)
        self.assertIsNone(card.root)
        self.assertIsNone(card.max_model_len)

    def test_model_card_with_optional_fields(self):
        """Test model card with optional fields"""
        card = ModelCard(
            id="test-model",
            root="/path/to/model",
            max_model_len=2048,
            created=1234567890,
        )
        self.assertEqual(card.id, "test-model")
        self.assertEqual(card.root, "/path/to/model")
        self.assertEqual(card.max_model_len, 2048)
        self.assertEqual(card.created, 1234567890)

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


class TestErrorResponse(unittest.TestCase):
    """Test ErrorResponse protocol model"""

    def test_basic_error_response(self):
        """Test basic error response creation"""
        error = ErrorResponse(
            message="Invalid request", type="BadRequestError", code=400
        )
        self.assertEqual(error.object, "error")
        self.assertEqual(error.message, "Invalid request")
        self.assertEqual(error.type, "BadRequestError")
        self.assertEqual(error.code, 400)
        self.assertIsNone(error.param)

    def test_error_response_with_param(self):
        """Test error response with parameter"""
        error = ErrorResponse(
            message="Invalid temperature",
            type="ValidationError",
            code=422,
            param="temperature",
        )
        self.assertEqual(error.param, "temperature")


class TestUsageInfo(unittest.TestCase):
    """Test UsageInfo protocol model"""

    def test_basic_usage_info(self):
        """Test basic usage info creation"""
        usage = UsageInfo(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        self.assertEqual(usage.prompt_tokens, 10)
        self.assertEqual(usage.completion_tokens, 20)
        self.assertEqual(usage.total_tokens, 30)
        self.assertIsNone(usage.prompt_tokens_details)

    def test_usage_info_with_cache_details(self):
        """Test usage info with cache details"""
        usage = UsageInfo(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            prompt_tokens_details={"cached_tokens": 5},
        )
        self.assertEqual(usage.prompt_tokens_details, {"cached_tokens": 5})


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

    def test_completion_request_with_options(self):
        """Test completion request with various options"""
        request = CompletionRequest(
            model="test-model",
            prompt=["Hello", "world"],
            max_tokens=100,
            temperature=0.7,
            top_p=0.9,
            n=2,
            stream=True,
            echo=True,
            stop=[".", "!"],
            logprobs=5,
        )
        self.assertEqual(request.prompt, ["Hello", "world"])
        self.assertEqual(request.max_tokens, 100)
        self.assertEqual(request.temperature, 0.7)
        self.assertEqual(request.top_p, 0.9)
        self.assertEqual(request.n, 2)
        self.assertTrue(request.stream)
        self.assertTrue(request.echo)
        self.assertEqual(request.stop, [".", "!"])
        self.assertEqual(request.logprobs, 5)

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


class TestCompletionResponse(unittest.TestCase):
    """Test CompletionResponse protocol model"""

    def test_basic_completion_response(self):
        """Test basic completion response"""
        choice = CompletionResponseChoice(
            index=0, text="Hello world!", finish_reason="stop"
        )
        usage = UsageInfo(prompt_tokens=2, completion_tokens=3, total_tokens=5)
        response = CompletionResponse(
            id="test-id", model="test-model", choices=[choice], usage=usage
        )
        self.assertEqual(response.id, "test-id")
        self.assertEqual(response.object, "text_completion")
        self.assertEqual(response.model, "test-model")
        self.assertEqual(len(response.choices), 1)
        self.assertEqual(response.choices[0].text, "Hello world!")
        self.assertEqual(response.usage.total_tokens, 5)


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
        self.assertEqual(request.temperature, 0.7)  # default
        self.assertFalse(request.stream)  # default
        self.assertEqual(request.tool_choice, "none")  # default when no tools

    def test_chat_completion_with_multimodal_content(self):
        """Test chat completion with multimodal content"""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQ..."},
                    },
                ],
            }
        ]
        request = ChatCompletionRequest(model="test-model", messages=messages)
        self.assertEqual(len(request.messages[0].content), 2)
        self.assertEqual(request.messages[0].content[0].type, "text")
        self.assertEqual(request.messages[0].content[1].type, "image_url")

    def test_chat_completion_with_tools(self):
        """Test chat completion with tools"""
        messages = [{"role": "user", "content": "What's the weather?"}]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather information",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                    },
                },
            }
        ]
        request = ChatCompletionRequest(
            model="test-model", messages=messages, tools=tools
        )
        self.assertEqual(len(request.tools), 1)
        self.assertEqual(request.tools[0].function.name, "get_weather")
        self.assertEqual(request.tool_choice, "auto")  # default when tools present

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


class TestChatCompletionResponse(unittest.TestCase):
    """Test ChatCompletionResponse protocol model"""

    def test_basic_chat_completion_response(self):
        """Test basic chat completion response"""
        message = ChatMessage(role="assistant", content="Hello there!")
        choice = ChatCompletionResponseChoice(
            index=0, message=message, finish_reason="stop"
        )
        usage = UsageInfo(prompt_tokens=2, completion_tokens=3, total_tokens=5)
        response = ChatCompletionResponse(
            id="test-id", model="test-model", choices=[choice], usage=usage
        )
        self.assertEqual(response.id, "test-id")
        self.assertEqual(response.object, "chat.completion")
        self.assertEqual(response.model, "test-model")
        self.assertEqual(len(response.choices), 1)
        self.assertEqual(response.choices[0].message.content, "Hello there!")

    def test_chat_completion_response_with_tool_calls(self):
        """Test chat completion response with tool calls"""
        tool_call = ToolCall(
            id="call_123",
            function=FunctionResponse(
                name="get_weather", arguments='{"location": "San Francisco"}'
            ),
        )
        message = ChatMessage(role="assistant", content=None, tool_calls=[tool_call])
        choice = ChatCompletionResponseChoice(
            index=0, message=message, finish_reason="tool_calls"
        )
        usage = UsageInfo(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        response = ChatCompletionResponse(
            id="test-id", model="test-model", choices=[choice], usage=usage
        )
        self.assertEqual(
            response.choices[0].message.tool_calls[0].function.name, "get_weather"
        )
        self.assertEqual(response.choices[0].finish_reason, "tool_calls")


class TestEmbeddingRequest(unittest.TestCase):
    """Test EmbeddingRequest protocol model"""

    def test_basic_embedding_request(self):
        """Test basic embedding request"""
        request = EmbeddingRequest(model="test-model", input="Hello world")
        self.assertEqual(request.model, "test-model")
        self.assertEqual(request.input, "Hello world")
        self.assertEqual(request.encoding_format, "float")  # default
        self.assertIsNone(request.dimensions)  # default

    def test_embedding_request_with_list_input(self):
        """Test embedding request with list input"""
        request = EmbeddingRequest(
            model="test-model", input=["Hello", "world"], dimensions=512
        )
        self.assertEqual(request.input, ["Hello", "world"])
        self.assertEqual(request.dimensions, 512)

    def test_multimodal_embedding_request(self):
        """Test multimodal embedding request"""
        multimodal_input = [
            MultimodalEmbeddingInput(text="Hello", image="base64_image_data"),
            MultimodalEmbeddingInput(text="World", image=None),
        ]
        request = EmbeddingRequest(model="test-model", input=multimodal_input)
        self.assertEqual(len(request.input), 2)
        self.assertEqual(request.input[0].text, "Hello")
        self.assertEqual(request.input[0].image, "base64_image_data")
        self.assertEqual(request.input[1].text, "World")
        self.assertIsNone(request.input[1].image)


class TestEmbeddingResponse(unittest.TestCase):
    """Test EmbeddingResponse protocol model"""

    def test_basic_embedding_response(self):
        """Test basic embedding response"""
        embedding_obj = EmbeddingObject(embedding=[0.1, 0.2, 0.3], index=0)
        usage = UsageInfo(prompt_tokens=3, total_tokens=3)
        response = EmbeddingResponse(
            data=[embedding_obj], model="test-model", usage=usage
        )
        self.assertEqual(response.object, "list")
        self.assertEqual(len(response.data), 1)
        self.assertEqual(response.data[0].embedding, [0.1, 0.2, 0.3])
        self.assertEqual(response.data[0].index, 0)
        self.assertEqual(response.usage.prompt_tokens, 3)


class TestScoringRequest(unittest.TestCase):
    """Test ScoringRequest protocol model"""

    def test_basic_scoring_request(self):
        """Test basic scoring request"""
        request = ScoringRequest(
            model="test-model", query="Hello", items=["World", "Earth"]
        )
        self.assertEqual(request.model, "test-model")
        self.assertEqual(request.query, "Hello")
        self.assertEqual(request.items, ["World", "Earth"])
        self.assertFalse(request.apply_softmax)  # default
        self.assertFalse(request.item_first)  # default

    def test_scoring_request_with_token_ids(self):
        """Test scoring request with token IDs"""
        request = ScoringRequest(
            model="test-model",
            query=[1, 2, 3],
            items=[[4, 5], [6, 7]],
            label_token_ids=[8, 9],
            apply_softmax=True,
            item_first=True,
        )
        self.assertEqual(request.query, [1, 2, 3])
        self.assertEqual(request.items, [[4, 5], [6, 7]])
        self.assertEqual(request.label_token_ids, [8, 9])
        self.assertTrue(request.apply_softmax)
        self.assertTrue(request.item_first)


class TestScoringResponse(unittest.TestCase):
    """Test ScoringResponse protocol model"""

    def test_basic_scoring_response(self):
        """Test basic scoring response"""
        response = ScoringResponse(scores=[[0.1, 0.9], [0.3, 0.7]], model="test-model")
        self.assertEqual(response.object, "scoring")
        self.assertEqual(response.scores, [[0.1, 0.9], [0.3, 0.7]])
        self.assertEqual(response.model, "test-model")
        self.assertIsNone(response.usage)  # default


class TestFileOperations(unittest.TestCase):
    """Test file operation protocol models"""

    def test_file_request(self):
        """Test file request model"""
        file_data = b"test file content"
        request = FileRequest(file=file_data, purpose="batch")
        self.assertEqual(request.file, file_data)
        self.assertEqual(request.purpose, "batch")

    def test_file_response(self):
        """Test file response model"""
        response = FileResponse(
            id="file-123",
            bytes=1024,
            created_at=1234567890,
            filename="test.jsonl",
            purpose="batch",
        )
        self.assertEqual(response.id, "file-123")
        self.assertEqual(response.object, "file")
        self.assertEqual(response.bytes, 1024)
        self.assertEqual(response.filename, "test.jsonl")

    def test_file_delete_response(self):
        """Test file delete response model"""
        response = FileDeleteResponse(id="file-123", deleted=True)
        self.assertEqual(response.id, "file-123")
        self.assertEqual(response.object, "file")
        self.assertTrue(response.deleted)


class TestBatchOperations(unittest.TestCase):
    """Test batch operation protocol models"""

    def test_batch_request(self):
        """Test batch request model"""
        request = BatchRequest(
            input_file_id="file-123",
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"custom": "value"},
        )
        self.assertEqual(request.input_file_id, "file-123")
        self.assertEqual(request.endpoint, "/v1/chat/completions")
        self.assertEqual(request.completion_window, "24h")
        self.assertEqual(request.metadata, {"custom": "value"})

    def test_batch_response(self):
        """Test batch response model"""
        response = BatchResponse(
            id="batch-123",
            endpoint="/v1/chat/completions",
            input_file_id="file-123",
            completion_window="24h",
            created_at=1234567890,
        )
        self.assertEqual(response.id, "batch-123")
        self.assertEqual(response.object, "batch")
        self.assertEqual(response.status, "validating")  # default
        self.assertEqual(response.endpoint, "/v1/chat/completions")


class TestResponseFormats(unittest.TestCase):
    """Test response format protocol models"""

    def test_basic_response_format(self):
        """Test basic response format"""
        format_obj = ResponseFormat(type="json_object")
        self.assertEqual(format_obj.type, "json_object")
        self.assertIsNone(format_obj.json_schema)

    def test_json_schema_response_format(self):
        """Test JSON schema response format"""
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        json_schema = JsonSchemaResponseFormat(
            name="person_schema", description="Person schema", schema=schema
        )
        format_obj = ResponseFormat(type="json_schema", json_schema=json_schema)
        self.assertEqual(format_obj.type, "json_schema")
        self.assertEqual(format_obj.json_schema.name, "person_schema")
        self.assertEqual(format_obj.json_schema.schema_, schema)

    def test_structural_tag_response_format(self):
        """Test structural tag response format"""
        structures = [
            {
                "begin": "<thinking>",
                "schema_": {"type": "string"},
                "end": "</thinking>",
            }
        ]
        format_obj = StructuralTagResponseFormat(
            type="structural_tag", structures=structures, triggers=["think"]
        )
        self.assertEqual(format_obj.type, "structural_tag")
        self.assertEqual(len(format_obj.structures), 1)
        self.assertEqual(format_obj.triggers, ["think"])


class TestLogProbs(unittest.TestCase):
    """Test LogProbs protocol models"""

    def test_basic_logprobs(self):
        """Test basic LogProbs model"""
        logprobs = LogProbs(
            text_offset=[0, 5, 11],
            token_logprobs=[-0.1, -0.2, -0.3],
            tokens=["Hello", " ", "world"],
            top_logprobs=[{"Hello": -0.1}, {" ": -0.2}, {"world": -0.3}],
        )
        self.assertEqual(len(logprobs.tokens), 3)
        self.assertEqual(logprobs.tokens, ["Hello", " ", "world"])
        self.assertEqual(logprobs.token_logprobs, [-0.1, -0.2, -0.3])

    def test_choice_logprobs(self):
        """Test ChoiceLogprobs model"""
        token_logprob = ChatCompletionTokenLogprob(
            token="Hello",
            bytes=[72, 101, 108, 108, 111],
            logprob=-0.1,
            top_logprobs=[
                TopLogprob(token="Hello", bytes=[72, 101, 108, 108, 111], logprob=-0.1)
            ],
        )
        choice_logprobs = ChoiceLogprobs(content=[token_logprob])
        self.assertEqual(len(choice_logprobs.content), 1)
        self.assertEqual(choice_logprobs.content[0].token, "Hello")


class TestStreamingModels(unittest.TestCase):
    """Test streaming response models"""

    def test_stream_options(self):
        """Test StreamOptions model"""
        options = StreamOptions(include_usage=True)
        self.assertTrue(options.include_usage)

    def test_chat_completion_stream_response(self):
        """Test ChatCompletionStreamResponse model"""
        delta = DeltaMessage(role="assistant", content="Hello")
        choice = ChatCompletionResponseStreamChoice(index=0, delta=delta)
        response = ChatCompletionStreamResponse(
            id="test-id", model="test-model", choices=[choice]
        )
        self.assertEqual(response.object, "chat.completion.chunk")
        self.assertEqual(response.choices[0].delta.content, "Hello")


class TestValidationEdgeCases(unittest.TestCase):
    """Test edge cases and validation scenarios"""

    def test_empty_messages_validation(self):
        """Test validation with empty messages"""
        with self.assertRaises(ValidationError):
            ChatCompletionRequest(model="test-model", messages=[])

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

    def test_invalid_temperature_range(self):
        """Test invalid temperature values"""
        # Note: The current protocol doesn't enforce temperature range,
        # but this test documents expected behavior
        request = CompletionRequest(model="test-model", prompt="Hello", temperature=5.0)
        self.assertEqual(request.temperature, 5.0)  # Currently allowed

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
