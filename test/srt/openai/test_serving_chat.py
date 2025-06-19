"""
Unit-tests for OpenAIServingChat — rewritten to use only the std-lib 'unittest'.
Run with either:
    python tests/test_serving_chat_unit.py -v
or
    python -m unittest discover -s tests -p "test_*unit.py" -v
"""

import unittest
import uuid
from typing import Optional
from unittest.mock import Mock, patch

from fastapi import Request

from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.srt.managers.io_struct import GenerateReqInput


class _MockTokenizerManager:
    """Minimal mock that satisfies OpenAIServingChat."""

    def __init__(self):
        self.model_config = Mock(is_multimodal=False)
        self.server_args = Mock(
            enable_cache_report=False,
            tool_call_parser="hermes",
            reasoning_parser=None,
        )
        self.chat_template_name: Optional[str] = "llama-3"

        # tokenizer stub
        self.tokenizer = Mock()
        self.tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        self.tokenizer.decode.return_value = "Test response"
        self.tokenizer.chat_template = None
        self.tokenizer.bos_token_id = 1

        # async generator stub for generate_request
        async def _mock_generate():
            yield {
                "text": "Test response",
                "meta_info": {
                    "id": f"chatcmpl-{uuid.uuid4()}",
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "cached_tokens": 0,
                    "finish_reason": {"type": "stop", "matched": None},
                    "output_token_logprobs": [(0.1, 1, "Test"), (0.2, 2, "response")],
                    "output_top_logprobs": None,
                },
                "index": 0,
            }

        self.generate_request = Mock(return_value=_mock_generate())
        self.create_abort_task = Mock()


class ServingChatTestCase(unittest.TestCase):
    # ------------- common fixtures -------------
    def setUp(self):
        self.tm = _MockTokenizerManager()
        self.chat = OpenAIServingChat(self.tm)

        # frequently reused requests
        self.basic_req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi?"}],
            temperature=0.7,
            max_tokens=100,
            stream=False,
        )
        self.stream_req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi?"}],
            temperature=0.7,
            max_tokens=100,
            stream=True,
        )

        self.fastapi_request = Mock(spec=Request)
        self.fastapi_request.headers = {}

    # ------------- conversion tests -------------
    def test_convert_to_internal_request_single(self):
        with patch(
            "sglang.srt.entrypoints.openai.serving_chat.generate_chat_conv"
        ) as conv_mock, patch.object(self.chat, "_process_messages") as proc_mock:
            conv_ins = Mock()
            conv_ins.get_prompt.return_value = "Test prompt"
            conv_ins.image_data = conv_ins.audio_data = None
            conv_ins.modalities = []
            conv_ins.stop_str = ["</s>"]
            conv_mock.return_value = conv_ins

            proc_mock.return_value = (
                "Test prompt",
                [1, 2, 3],
                None,
                None,
                [],
                ["</s>"],
                None,
            )

            adapted, processed = self.chat._convert_to_internal_request(
                [self.basic_req], ["rid"]
            )
            self.assertIsInstance(adapted, GenerateReqInput)
            self.assertFalse(adapted.stream)
            self.assertEqual(processed, self.basic_req)

    # ------------- tool-call branch -------------
    def test_tool_call_request_conversion(self):
        req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Weather?"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            tool_choice="auto",
        )

        with patch.object(
            self.chat,
            "_process_messages",
            return_value=("Prompt", [1, 2, 3], None, None, [], ["</s>"], None),
        ):
            adapted, _ = self.chat._convert_to_internal_request([req], ["rid"])
            self.assertEqual(adapted.rid, "rid")

    def test_tool_choice_none(self):
        req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi"}],
            tools=[{"type": "function", "function": {"name": "noop"}}],
            tool_choice="none",
        )
        with patch.object(
            self.chat,
            "_process_messages",
            return_value=("Prompt", [1, 2, 3], None, None, [], ["</s>"], None),
        ):
            adapted, _ = self.chat._convert_to_internal_request([req], ["rid"])
            self.assertEqual(adapted.rid, "rid")

    # ------------- multimodal branch -------------
    def test_multimodal_request_with_images(self):
        self.tm.model_config.is_multimodal = True

        req = ChatCompletionRequest(
            model="x",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's in the image?"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/jpeg;base64,"},
                        },
                    ],
                }
            ],
        )

        with patch.object(
            self.chat,
            "_apply_jinja_template",
            return_value=("prompt", [1, 2], ["img"], None, [], []),
        ), patch.object(
            self.chat,
            "_apply_conversation_template",
            return_value=("prompt", ["img"], None, [], []),
        ):
            out = self.chat._process_messages(req, True)
            _, _, image_data, *_ = out
            self.assertEqual(image_data, ["img"])

    # ------------- template handling -------------
    def test_jinja_template_processing(self):
        req = ChatCompletionRequest(
            model="x", messages=[{"role": "user", "content": "Hello"}]
        )
        self.tm.chat_template_name = None
        self.tm.tokenizer.chat_template = "<jinja>"

        with patch.object(
            self.chat,
            "_apply_jinja_template",
            return_value=("processed", [1], None, None, [], ["</s>"]),
        ), patch("builtins.hasattr", return_value=True):
            prompt, prompt_ids, *_ = self.chat._process_messages(req, False)
            self.assertEqual(prompt, "processed")
            self.assertEqual(prompt_ids, [1])

    # ------------- sampling-params -------------
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
        with patch.object(
            self.chat,
            "_process_messages",
            return_value=("Prompt", [1], None, None, [], ["</s>"], None),
        ):
            params = self.chat._build_sampling_params(req, ["</s>"], None)
            self.assertEqual(params["temperature"], 0.8)
            self.assertEqual(params["max_new_tokens"], 150)
            self.assertEqual(params["min_new_tokens"], 5)
            self.assertEqual(params["stop"], ["</s>"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
