"""
Unit tests for multimodal content validation in OpenAIServingChat.

Verifies that requests containing multimodal content (image_url, video_url,
audio_url) are rejected with a clear error when the server is not started
with --enable-multimodal.

Refs: https://github.com/sgl-project/sglang/issues/21695
"""

import importlib
import importlib.abc
import importlib.machinery
import sys
import types
import unittest
from unittest.mock import MagicMock, Mock


# Stub out sgl_kernel before any sglang import so the test runs on CPU-only runners.
class _SglKernelMockLoader(importlib.abc.Loader):
    def create_module(self, spec):
        mod = types.ModuleType(spec.name)
        mod.__path__ = []
        mod.__package__ = spec.name
        mod.__loader__ = self
        mod.__getattr__ = lambda name: MagicMock()
        return mod

    def exec_module(self, module):
        pass


class _SglKernelMockFinder(importlib.abc.MetaPathFinder):
    _PREFIX = "sgl_kernel"
    _loader = _SglKernelMockLoader()

    def find_spec(self, fullname, path, target=None):
        if fullname == self._PREFIX or fullname.startswith(self._PREFIX + "."):
            return importlib.machinery.ModuleSpec(
                fullname, self._loader, is_package=True
            )
        return None


if "sgl_kernel" not in sys.modules:
    sys.meta_path.insert(0, _SglKernelMockFinder())

from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionMessageContentAudioPart,
    ChatCompletionMessageContentAudioURL,
    ChatCompletionMessageContentImagePart,
    ChatCompletionMessageContentImageURL,
    ChatCompletionMessageContentTextPart,
    ChatCompletionMessageContentVideoPart,
    ChatCompletionMessageContentVideoURL,
    ChatCompletionMessageUserParam,
    ChatCompletionRequest,
)
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


def _make_text_request() -> ChatCompletionRequest:
    """Create a text-only chat completion request."""
    return ChatCompletionRequest(
        model="test-model",
        messages=[
            ChatCompletionMessageUserParam(
                role="user",
                content="Hello, how are you?",
            )
        ],
    )


def _make_image_request() -> ChatCompletionRequest:
    """Create a chat request with an image."""
    return ChatCompletionRequest(
        model="test-model",
        messages=[
            ChatCompletionMessageUserParam(
                role="user",
                content=[
                    ChatCompletionMessageContentTextPart(
                        type="text", text="Describe this image."
                    ),
                    ChatCompletionMessageContentImagePart(
                        type="image_url",
                        image_url=ChatCompletionMessageContentImageURL(
                            url="data:image/png;base64,iVBORw0KGgo="
                        ),
                    ),
                ],
            )
        ],
    )


def _make_video_request() -> ChatCompletionRequest:
    """Create a chat request with a video."""
    return ChatCompletionRequest(
        model="test-model",
        messages=[
            ChatCompletionMessageUserParam(
                role="user",
                content=[
                    ChatCompletionMessageContentTextPart(
                        type="text", text="Describe this video."
                    ),
                    ChatCompletionMessageContentVideoPart(
                        type="video_url",
                        video_url=ChatCompletionMessageContentVideoURL(
                            url="https://example.com/video.mp4"
                        ),
                    ),
                ],
            )
        ],
    )


def _make_audio_request() -> ChatCompletionRequest:
    """Create a chat request with audio."""
    return ChatCompletionRequest(
        model="test-model",
        messages=[
            ChatCompletionMessageUserParam(
                role="user",
                content=[
                    ChatCompletionMessageContentTextPart(
                        type="text", text="Transcribe this audio."
                    ),
                    ChatCompletionMessageContentAudioPart(
                        type="audio_url",
                        audio_url=ChatCompletionMessageContentAudioURL(
                            url="https://example.com/audio.wav"
                        ),
                    ),
                ],
            )
        ],
    )


def _make_multipart_text_only_request() -> ChatCompletionRequest:
    """Create a request with list content that only has text parts."""
    return ChatCompletionRequest(
        model="test-model",
        messages=[
            ChatCompletionMessageUserParam(
                role="user",
                content=[
                    ChatCompletionMessageContentTextPart(
                        type="text", text="First part."
                    ),
                    ChatCompletionMessageContentTextPart(
                        type="text", text="Second part."
                    ),
                ],
            )
        ],
    )


def _build_serving_chat(is_multimodal: bool) -> OpenAIServingChat:
    """Build an OpenAIServingChat instance with mocked dependencies."""
    tokenizer_manager = Mock()
    tokenizer_manager.model_config = Mock()
    tokenizer_manager.model_config.is_multimodal = is_multimodal
    tokenizer_manager.model_config.get_default_sampling_params.return_value = None
    tokenizer_manager.model_config.hf_config = Mock()
    tokenizer_manager.model_config.hf_config.architectures = ["LlamaForCausalLM"]
    tokenizer_manager.model_config.hf_config.model_type = "llama"
    tokenizer_manager.server_args = Mock()
    tokenizer_manager.server_args.tool_call_parser = None
    tokenizer_manager.server_args.reasoning_parser = None
    tokenizer_manager.server_args.context_length = 4096
    tokenizer_manager.server_args.allow_auto_truncate = False
    tokenizer_manager.tokenizer = Mock()
    tokenizer_manager.tokenizer.chat_template = None

    template_manager = Mock()
    template_manager.chat_template_name = None
    template_manager.jinja_template_content_format = None

    return OpenAIServingChat(tokenizer_manager, template_manager)


class TestMultimodalValidationDisabled(CustomTestCase):
    """Tests that multimodal content is rejected when is_multimodal=False."""

    def setUp(self):
        self.serving = _build_serving_chat(is_multimodal=False)

    def test_text_only_request_accepted(self):
        """Text-only request should pass validation."""
        result = self.serving._validate_request(_make_text_request())
        self.assertIsNone(result)

    def test_multipart_text_only_accepted(self):
        """List content with only text parts should pass validation."""
        result = self.serving._validate_request(_make_multipart_text_only_request())
        self.assertIsNone(result)

    def test_image_request_rejected(self):
        """Request with image_url should be rejected."""
        result = self.serving._validate_request(_make_image_request())
        self.assertIsNotNone(result)
        self.assertIn("image_url", result)
        self.assertIn("--enable-multimodal", result)

    def test_video_request_rejected(self):
        """Request with video_url should be rejected."""
        result = self.serving._validate_request(_make_video_request())
        self.assertIsNotNone(result)
        self.assertIn("video_url", result)
        self.assertIn("--enable-multimodal", result)

    def test_audio_request_rejected(self):
        """Request with audio_url should be rejected."""
        result = self.serving._validate_request(_make_audio_request())
        self.assertIsNotNone(result)
        self.assertIn("audio_url", result)
        self.assertIn("--enable-multimodal", result)


class TestMultimodalValidationEnabled(CustomTestCase):
    """Tests that multimodal content is accepted when is_multimodal=True."""

    def setUp(self):
        self.serving = _build_serving_chat(is_multimodal=True)

    def test_text_only_request_accepted(self):
        """Text-only request should pass validation."""
        result = self.serving._validate_request(_make_text_request())
        self.assertIsNone(result)

    def test_image_request_accepted(self):
        """Image request should pass validation when multimodal enabled."""
        result = self.serving._validate_request(_make_image_request())
        self.assertIsNone(result)

    def test_video_request_accepted(self):
        """Video request should pass validation when multimodal enabled."""
        result = self.serving._validate_request(_make_video_request())
        self.assertIsNone(result)

    def test_audio_request_accepted(self):
        """Audio request should pass validation when multimodal enabled."""
        result = self.serving._validate_request(_make_audio_request())
        self.assertIsNone(result)


class TestDetectMultimodalContent(CustomTestCase):
    """Tests for the _detect_multimodal_content static method."""

    def test_text_only_returns_none(self):
        result = OpenAIServingChat._detect_multimodal_content(_make_text_request())
        self.assertIsNone(result)

    def test_multipart_text_returns_none(self):
        result = OpenAIServingChat._detect_multimodal_content(
            _make_multipart_text_only_request()
        )
        self.assertIsNone(result)

    def test_image_detected(self):
        result = OpenAIServingChat._detect_multimodal_content(_make_image_request())
        self.assertEqual(result, "image_url")

    def test_video_detected(self):
        result = OpenAIServingChat._detect_multimodal_content(_make_video_request())
        self.assertEqual(result, "video_url")

    def test_audio_detected(self):
        result = OpenAIServingChat._detect_multimodal_content(_make_audio_request())
        self.assertEqual(result, "audio_url")

    def test_empty_messages(self):
        """Request with empty messages should return None."""
        req = ChatCompletionRequest(model="test-model", messages=[])
        result = OpenAIServingChat._detect_multimodal_content(req)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
