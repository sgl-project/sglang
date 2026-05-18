"""Unit tests for OpenAIServingResponses (/v1/responses).

Regression coverage for https://github.com/sgl-project/sglang/issues/25593:
on multimodal models, ``_make_request`` returns the rendered chat-template
string (not token ids), so the caller must construct ``GenerateReqInput``
with ``text=`` and forward the extracted multimodal data — otherwise the
string is fed to ``input_ids=`` and ``GenerateReqInput.normalize_batch_and_arguments``
raises ``"input_ids should be a list of lists for batch processing."``.
"""

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede any import that pulls in sgl_kernel

import unittest
import uuid
from typing import Optional
from unittest.mock import Mock, patch

from fastapi import Request

from sglang.srt.entrypoints.openai.protocol import (
    MessageProcessingResult,
    ResponsesRequest,
)
from sglang.srt.entrypoints.openai.serving_responses import OpenAIServingResponses
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class _MockTokenizerManager:
    """Minimal mock that satisfies OpenAIServingResponses."""

    def __init__(self, is_multimodal: bool):
        # hf_config.model_type drives harmony detection in __init__.
        hf_config = Mock()
        hf_config.model_type = "qwen3"
        hf_config.architectures = ["Qwen3ForCausalLM"]

        self.model_config = Mock(
            is_multimodal=is_multimodal,
            context_len=4096,
            hf_config=hf_config,
        )
        self.server_args = Mock(
            enable_cache_report=False,
            tool_call_parser=None,
            reasoning_parser=None,
            stream_response_default_include_usage=False,
        )

        self.tokenizer = Mock()
        self.tokenizer.encode.return_value = [10, 11, 12, 13]
        self.tokenizer.decode.return_value = "<rendered prompt>"
        self.tokenizer.chat_template = None
        self.tokenizer.bos_token_id = 1

        self.num_reserved_tokens = 0
        self.model_path = "test-model"

        captured_requests: list[GenerateReqInput] = []
        self._captured_requests = captured_requests

        async def _mock_generate(adapted_request, raw_request):
            captured_requests.append(adapted_request)
            yield {
                "text": "ok",
                "meta_info": {
                    "id": f"resp-{uuid.uuid4()}",
                    "prompt_tokens": 4,
                    "completion_tokens": 1,
                    "cached_tokens": 0,
                    "finish_reason": {"type": "stop", "matched": None},
                },
                "index": 0,
            }

        self.generate_request = Mock(side_effect=_mock_generate)
        self.create_abort_task = Mock()


class _MockTemplateManager:
    def __init__(self):
        self.chat_template_name: Optional[str] = None
        self.jinja_template_content_format: Optional[str] = None
        self.completion_template_name: Optional[str] = None
        self.reasoning_config = None
        self.force_reasoning = False


class ServingResponsesTestCase(unittest.TestCase):
    def _build(self, is_multimodal: bool) -> OpenAIServingResponses:
        tm = _MockTokenizerManager(is_multimodal=is_multimodal)
        srv = OpenAIServingResponses(tm, _MockTemplateManager())
        return srv

    def test_make_request_multimodal_returns_processed_messages(self):
        """_make_request must return processed_messages so the caller has
        access to image/video/audio data for multimodal models."""
        srv = self._build(is_multimodal=True)
        request = ResponsesRequest(model="m", input="hello")

        processed = MessageProcessingResult(
            prompt="<rendered prompt>",
            prompt_ids=[10, 11, 12, 13],
            image_data=["img://0"],
            audio_data=None,
            video_data=None,
            modalities=["image"],
            stop=[],
        )
        with patch.object(srv, "_process_messages", return_value=processed):
            import asyncio

            messages, request_prompts, engine_prompts, returned = asyncio.run(
                srv._make_request(request, None, srv.tokenizer_manager.tokenizer)
            )

        # For multimodal, engine_prompts carries the rendered template string.
        self.assertEqual(engine_prompts, ["<rendered prompt>"])
        self.assertEqual(request_prompts, ["<rendered prompt>"])
        self.assertIs(returned, processed)
        self.assertEqual(returned.image_data, ["img://0"])

    def test_make_request_text_only_returns_processed_messages(self):
        """Text-only path returns prompt_ids and still surfaces processed_messages."""
        srv = self._build(is_multimodal=False)
        request = ResponsesRequest(model="m", input="hello")

        processed = MessageProcessingResult(
            prompt="<rendered prompt>",
            prompt_ids=[10, 11, 12, 13],
            image_data=None,
            audio_data=None,
            video_data=None,
            modalities=[],
            stop=[],
        )
        with patch.object(srv, "_process_messages", return_value=processed):
            import asyncio

            _, request_prompts, engine_prompts, returned = asyncio.run(
                srv._make_request(request, None, srv.tokenizer_manager.tokenizer)
            )

        self.assertEqual(engine_prompts, [[10, 11, 12, 13]])
        self.assertEqual(request_prompts, [[10, 11, 12, 13]])
        self.assertIs(returned, processed)

    def test_create_responses_multimodal_builds_text_kwargs(self):
        """Regression for #25593: multimodal must use text= + multimodal data,
        not input_ids=, otherwise GenerateReqInput.normalize_batch_and_arguments
        raises 'input_ids should be a list of lists for batch processing.'."""
        srv = self._build(is_multimodal=True)
        request = ResponsesRequest(model="m", input="hello")
        raw_request = Mock(spec=Request)
        raw_request.state = Mock()

        processed = MessageProcessingResult(
            prompt="<rendered prompt>",
            prompt_ids=[10, 11, 12, 13],
            image_data=["img://0"],
            audio_data=None,
            video_data=None,
            modalities=["image"],
            stop=[],
        )
        with patch.object(srv, "_process_messages", return_value=processed):
            import asyncio

            asyncio.run(srv.create_responses(request, raw_request))

        captured = srv.tokenizer_manager._captured_requests
        self.assertEqual(len(captured), 1)
        adapted: GenerateReqInput = captured[0]
        self.assertEqual(adapted.text, "<rendered prompt>")
        self.assertIsNone(adapted.input_ids)
        self.assertEqual(adapted.image_data, ["img://0"])
        self.assertEqual(adapted.modalities, ["image"])

        # The original crash in #25593 surfaced inside
        # normalize_batch_and_arguments — exercise it here so the test
        # would catch a regression that passes the multimodal prompt
        # through input_ids= again.
        adapted.normalize_batch_and_arguments()

    def test_create_responses_text_only_uses_input_ids(self):
        """Text-only path must keep using input_ids=<list[int]>."""
        srv = self._build(is_multimodal=False)
        request = ResponsesRequest(model="m", input="hello")
        raw_request = Mock(spec=Request)
        raw_request.state = Mock()

        processed = MessageProcessingResult(
            prompt="<rendered prompt>",
            prompt_ids=[10, 11, 12, 13],
            image_data=None,
            audio_data=None,
            video_data=None,
            modalities=[],
            stop=[],
        )
        with patch.object(srv, "_process_messages", return_value=processed):
            import asyncio

            asyncio.run(srv.create_responses(request, raw_request))

        captured = srv.tokenizer_manager._captured_requests
        self.assertEqual(len(captured), 1)
        adapted: GenerateReqInput = captured[0]
        self.assertEqual(adapted.input_ids, [10, 11, 12, 13])
        self.assertIsNone(adapted.text)


if __name__ == "__main__":
    unittest.main()
