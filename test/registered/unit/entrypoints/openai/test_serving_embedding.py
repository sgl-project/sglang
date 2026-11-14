"""
Unit tests for the OpenAIServingEmbedding class from serving_embedding.py.
"""

import base64
import importlib
import importlib.abc
import importlib.machinery
import struct
import sys
import types
import unittest
import uuid
from unittest.mock import MagicMock, Mock

import jinja2


# Stub out sgl_kernel (and all submodules) before any sglang import so
# the test runs on CPU-only runners without the real CUDA library.
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
    """Import hook that intercepts all sgl_kernel.* imports and returns mocks."""

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

from fastapi import Request
from pydantic import ValidationError

from sglang.srt.entrypoints.openai.protocol import (
    EmbeddingRequest,
    MultimodalEmbeddingInput,
)
from sglang.srt.entrypoints.openai.serving_embedding import OpenAIServingEmbedding
from sglang.srt.managers.io_struct import EmbeddingReqInput
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")
register_cpu_ci(est_time=6, suite="stage-b-test-cpu-intel")


# Mock TokenizerManager for embedding tests
class _MockTokenizerManager:
    def __init__(self):
        self.model_config = Mock()
        self.model_config.is_multimodal = False
        self.server_args = Mock()
        self.server_args.enable_cache_report = False
        self.model_path = "test-model"

        # Mock tokenizer
        self.tokenizer = Mock()
        self.tokenizer.encode = Mock(return_value=[1, 2, 3, 4, 5])
        self.tokenizer.decode = Mock(return_value="Test embedding input")
        self.tokenizer.chat_template = None
        self.tokenizer.bos_token_id = 1

        # Mock generate_request method for embeddings
        async def mock_generate_embedding():
            yield {
                "embedding": [0.1, 0.2, 0.3, 0.4, 0.5] * 20,  # 100-dim embedding
                "meta_info": {
                    "id": f"embd-{uuid.uuid4()}",
                    "prompt_tokens": 5,
                },
            }

        self.generate_request = Mock(return_value=mock_generate_embedding())


# Writes each content part where it appears, so a test can see where media landed.
_ORDER_TEMPLATE = (
    "{% for m in messages %}<{{ m.role }}>"
    "{% if m.content is string %}{{ m.content }}"
    "{% else %}{% for p in m.content %}"
    "{% if p.type == 'text' %}{{ p.text }}{% else %}[{{ p.type }}]{% endif %}"
    "{% endfor %}{% endif %}{% endfor %}"
    "{% if instruction %}<instruction:{{ instruction }}>{% endif %}"
    "{% if add_generation_prompt %}<assistant>{% endif %}"
)


def _render_order_template(messages, tokenize, add_generation_prompt, **kwargs):
    return jinja2.Template(_ORDER_TEMPLATE).render(
        messages=messages, add_generation_prompt=add_generation_prompt, **kwargs
    )


def _text(text):
    return {"type": "text", "text": text}


def _image(url):
    return {"type": "image_url", "image_url": {"url": url}}


def _video(url):
    return {"type": "video_url", "video_url": {"url": url}}


def _urls(media):
    return [item.url for item in media]


# Mock TemplateManager for embedding tests
class _MockTemplateManager:
    def __init__(self):
        self.chat_template_name = None  # None for embeddings usually
        self.jinja_template_content_format = "openai"
        self.completion_template_name = None
        self.jinja_template_may_reorder_tool_results = False


class ServingEmbeddingTestCase(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.tokenizer_manager = _MockTokenizerManager()
        self.template_manager = _MockTemplateManager()
        self.serving_embedding = OpenAIServingEmbedding(
            self.tokenizer_manager, self.template_manager
        )

        self.request = Mock(spec=Request)
        self.request.headers = {}

        self.basic_req = EmbeddingRequest(
            model="test-model",
            input="Hello, how are you?",
            encoding_format="float",
        )
        self.list_req = EmbeddingRequest(
            model="test-model",
            input=["Hello, how are you?", "I am fine, thank you!"],
            encoding_format="float",
        )
        self.multimodal_req = EmbeddingRequest(
            model="test-model",
            input=[
                MultimodalEmbeddingInput(text="Hello", image="base64_image_data"),
                MultimodalEmbeddingInput(text="World", image=None),
            ],
            encoding_format="float",
        )
        self.image_only_multimodal_req = EmbeddingRequest(
            model="test-model",
            input=[
                MultimodalEmbeddingInput(text=None, image="base64_image_data"),
            ],
            encoding_format="float",
        )
        self.video_multimodal_req = EmbeddingRequest(
            model="test-model",
            input=[
                MultimodalEmbeddingInput(
                    text="Describe", image=None, video="base64_video_data"
                ),
            ],
            encoding_format="float",
        )
        self.token_ids_req = EmbeddingRequest(
            model="test-model",
            input=[1, 2, 3, 4, 5],
            encoding_format="float",
        )

    def test_convert_single_string_request(self):
        """Test converting single string request to internal format."""
        adapted_request, processed_request = (
            self.serving_embedding._convert_to_internal_request(self.basic_req)
        )

        self.assertIsInstance(adapted_request, EmbeddingReqInput)
        self.assertEqual(adapted_request.text, "Hello, how are you?")
        # self.assertEqual(adapted_request.rid, "test-id")
        self.assertEqual(processed_request, self.basic_req)

    def test_convert_list_string_request(self):
        """Test converting list of strings request to internal format."""
        adapted_request, processed_request = (
            self.serving_embedding._convert_to_internal_request(self.list_req)
        )

        self.assertIsInstance(adapted_request, EmbeddingReqInput)
        self.assertEqual(
            adapted_request.text, ["Hello, how are you?", "I am fine, thank you!"]
        )
        # self.assertEqual(adapted_request.rid, "test-id")
        self.assertEqual(processed_request, self.list_req)

    def test_convert_token_ids_request(self):
        """Test converting token IDs request to internal format."""
        adapted_request, processed_request = (
            self.serving_embedding._convert_to_internal_request(self.token_ids_req)
        )

        self.assertIsInstance(adapted_request, EmbeddingReqInput)
        self.assertEqual(adapted_request.input_ids, [1, 2, 3, 4, 5])
        # self.assertEqual(adapted_request.rid, "test-id")
        self.assertEqual(processed_request, self.token_ids_req)

    def test_convert_multimodal_request(self):
        """Test converting multimodal request to internal format."""
        adapted_request, processed_request = (
            self.serving_embedding._convert_to_internal_request(self.multimodal_req)
        )

        self.assertIsInstance(adapted_request, EmbeddingReqInput)
        # Should extract text and images separately
        self.assertEqual(len(adapted_request.text), 2)
        self.assertIn("Hello", adapted_request.text)
        self.assertIn("World", adapted_request.text)
        self.assertEqual(adapted_request.image_data[0], "base64_image_data")
        self.assertIsNone(adapted_request.image_data[1])
        # self.assertEqual(adapted_request.rid, "test-id")

    def test_convert_multimodal_request_with_jinja_chat_template(self):
        """Multimodal embeddings should apply explicit/HF Jinja chat templates."""
        self.tokenizer_manager.tokenizer.chat_template = "mock-template"
        self.tokenizer_manager.tokenizer.apply_chat_template = Mock(
            side_effect=[
                "<prompt>Hello<image></prompt>",
                "<prompt>World</prompt>",
            ]
        )

        adapted_request, _ = self.serving_embedding._convert_to_internal_request(
            self.multimodal_req
        )

        self.assertEqual(
            adapted_request.text,
            ["<prompt>Hello<image></prompt>", "<prompt>World</prompt>"],
        )
        self.assertEqual(adapted_request.image_data[0], "base64_image_data")
        self.assertIsNone(adapted_request.image_data[1])
        self.assertEqual(
            self.tokenizer_manager.tokenizer.apply_chat_template.call_count, 2
        )
        first_call = (
            self.tokenizer_manager.tokenizer.apply_chat_template.call_args_list[0]
        )
        first_messages = first_call.args[0]
        self.assertEqual(first_messages[0]["role"], "user")
        self.assertEqual(first_messages[0]["content"][0]["type"], "image")
        self.assertEqual(first_messages[0]["content"][1]["type"], "text")
        self.assertEqual(first_messages[0]["content"][1]["text"], "Hello")
        self.assertEqual(first_call.kwargs["tokenize"], False)
        self.assertEqual(first_call.kwargs["add_generation_prompt"], True)

        second_call = (
            self.tokenizer_manager.tokenizer.apply_chat_template.call_args_list[1]
        )
        second_messages = second_call.args[0]
        self.assertEqual(len(second_messages[0]["content"]), 1)
        self.assertEqual(second_messages[0]["content"][0]["type"], "text")
        self.assertEqual(second_messages[0]["content"][0]["text"], "World")

    def test_convert_image_only_multimodal_request_with_jinja_chat_template(self):
        """Image-only requests should not inject literal padding into Jinja prompts."""
        self.tokenizer_manager.tokenizer.chat_template = "mock-template"
        self.tokenizer_manager.tokenizer.apply_chat_template = Mock(
            return_value="<prompt><image></prompt>"
        )

        adapted_request, _ = self.serving_embedding._convert_to_internal_request(
            self.image_only_multimodal_req
        )

        self.assertEqual(adapted_request.text, "<prompt><image></prompt>")
        first_call = self.tokenizer_manager.tokenizer.apply_chat_template.call_args
        first_messages = first_call.args[0]
        self.assertEqual(first_messages[0]["role"], "user")
        self.assertEqual(len(first_messages[0]["content"]), 1)
        self.assertEqual(first_messages[0]["content"][0]["type"], "image")

    def test_convert_video_multimodal_request_with_jinja_chat_template(self):
        """Video inputs should land in video_data and flow through the Jinja branch."""
        self.tokenizer_manager.tokenizer.chat_template = "mock-template"
        self.tokenizer_manager.tokenizer.apply_chat_template = Mock(
            return_value="<prompt>Describe<video></prompt>"
        )

        adapted_request, _ = self.serving_embedding._convert_to_internal_request(
            self.video_multimodal_req
        )

        self.assertEqual(adapted_request.text, "<prompt>Describe<video></prompt>")
        self.assertEqual(adapted_request.video_data, "base64_video_data")
        self.assertIsNone(adapted_request.image_data)
        first_messages = (
            self.tokenizer_manager.tokenizer.apply_chat_template.call_args.args[0]
        )
        content = first_messages[0]["content"]
        self.assertEqual([c["type"] for c in content], ["video", "text"])

    def test_multimodal_request_falls_back_when_no_chat_template(self):
        """Without any chat template the raw-text fallback must run without raising."""
        self.tokenizer_manager.tokenizer.chat_template = None

        adapted_request, _ = self.serving_embedding._convert_to_internal_request(
            self.image_only_multimodal_req
        )

        # text=None on an image-only input falls back to the "padding" literal.
        self.assertEqual(adapted_request.text, "padding")
        self.assertEqual(adapted_request.image_data, "base64_image_data")

    def test_multimodal_request_with_no_tokenizer_uses_fallback(self):
        """Missing tokenizer should not crash the Jinja branch check."""
        self.tokenizer_manager.tokenizer = None

        adapted_request, _ = self.serving_embedding._convert_to_internal_request(
            self.multimodal_req
        )

        self.assertEqual(adapted_request.text, ["Hello", "World"])

    def test_jinja_template_errors_are_raised_as_value_error(self):
        """Template failures should be converted to ValueError for a 400 response."""
        self.tokenizer_manager.tokenizer.chat_template = "mock-template"
        self.tokenizer_manager.tokenizer.apply_chat_template = Mock(
            side_effect=jinja2.TemplateError("bad template")
        )

        with self.assertRaisesRegex(ValueError, "bad template"):
            self.serving_embedding._convert_to_internal_request(
                self.image_only_multimodal_req
            )

    def test_jinja_template_syntax_error_includes_location(self):
        """TemplateSyntaxError should surface template name and line number."""
        err = jinja2.TemplateSyntaxError("unexpected end", lineno=7, name="mock.jinja")
        self.tokenizer_manager.tokenizer.chat_template = "mock-template"
        self.tokenizer_manager.tokenizer.apply_chat_template = Mock(side_effect=err)

        with self.assertRaises(ValueError) as ctx:
            self.serving_embedding._convert_to_internal_request(
                self.image_only_multimodal_req
            )
        message = str(ctx.exception)
        self.assertIn("mock.jinja", message)
        self.assertIn("line=7", message)

    def test_non_jinja_template_errors_are_raised_as_value_error(self):
        """TypeError / KeyError from apply_chat_template should map to 400, not 500."""
        self.tokenizer_manager.tokenizer.chat_template = "mock-template"
        self.tokenizer_manager.tokenizer.apply_chat_template = Mock(
            side_effect=KeyError("missing_field")
        )

        with self.assertRaisesRegex(ValueError, "missing_field"):
            self.serving_embedding._convert_to_internal_request(
                self.image_only_multimodal_req
            )

    def test_base64_embedding_response_uses_little_endian_float32(self):
        response = self.serving_embedding._build_embedding_response(
            [{"embedding": [0.25, -1.5], "meta_info": {"prompt_tokens": 2}}],
            encoding_format="base64",
        )

        encoded_embedding = response.data[0].embedding
        self.assertIsInstance(encoded_embedding, str)
        self.assertEqual(
            struct.unpack("<2f", base64.b64decode(encoded_embedding)), (0.25, -1.5)
        )
        self.assertEqual(response.usage.prompt_tokens, 2)

    def test_rejects_unknown_embedding_encoding_format(self):
        invalid_request = EmbeddingRequest(
            model="test-model", input="hello", encoding_format="binary"
        )

        self.assertIn(
            "encoding_format must be either",
            self.serving_embedding._validate_request(invalid_request),
        )

    def _use_order_template(self, *, is_multimodal=True):
        self.tokenizer_manager.model_config.is_multimodal = is_multimodal
        self.tokenizer_manager.tokenizer.chat_template = _ORDER_TEMPLATE
        self.tokenizer_manager.tokenizer.apply_chat_template = Mock(
            side_effect=_render_order_template
        )

    def _convert(self, **request_fields):
        request = EmbeddingRequest(model="test-model", **request_fields)
        self.assertIsNone(self.serving_embedding._validate_request(request))
        adapted_request, _ = self.serving_embedding._convert_to_internal_request(
            request
        )
        return adapted_request

    def test_chat_conversation_in_input_is_rendered_not_dropped(self):
        """A vLLM-style conversation sent in `input` must not lose its content.

        It used to validate as an empty MultimodalEmbeddingInput and be embedded
        as the literal "padding", dropping the text and the image.
        """
        self._use_order_template()

        adapted_request = self._convert(
            input=[{"role": "user", "content": [_text("a red car"), _image("car")]}]
        )

        self.assertEqual(adapted_request.text, "<user>a red car[image]")
        self.assertEqual(_urls(adapted_request.image_data), ["car"])

    def test_chat_messages_keep_interleaved_media_order(self):
        """Media stays at its position in the prompt and in the media lists,
        and no generation prompt is added by default (as in vLLM)."""
        self._use_order_template()

        adapted_request = self._convert(
            messages=[
                {
                    "role": "user",
                    "content": [
                        _text("before"),
                        _image("first"),
                        _text("between"),
                        _video("clip"),
                        _image("second"),
                        _text("after"),
                    ],
                }
            ]
        )

        self.assertEqual(
            adapted_request.text,
            "<user>before[image]between[video][image]after",
        )
        self.assertEqual(_urls(adapted_request.image_data), ["first", "second"])
        self.assertEqual(adapted_request.video_data, ["clip"])

    def test_batched_chat_messages_give_one_media_entry_per_conversation(self):
        """Media lists must line up with the batch, with None for text-only items."""
        self._use_order_template()

        adapted_request = self._convert(
            messages=[
                [{"role": "query", "content": [_text("cat"), _image("cat.png")]}],
                [{"role": "document", "content": "a text-only document"}],
            ]
        )

        self.assertEqual(
            adapted_request.text,
            ["<query>cat[image]", "<document>a text-only document"],
        )
        self.assertEqual(_urls(adapted_request.image_data[0]), ["cat.png"])
        self.assertIsNone(adapted_request.image_data[1])
        self.assertIsNone(adapted_request.video_data)
        adapted_request.normalize_batch_and_arguments()
        self.assertEqual(adapted_request.batch_size, 2)

    def test_text_only_model_does_not_add_special_tokens_to_chat_prompt(self):
        """The chat template already writes BOS; encoding must not add another."""
        self._use_order_template(is_multimodal=False)
        bos = 1
        self.tokenizer_manager.tokenizer.encode = lambda text, add_special_tokens=True: (
            ([bos] if add_special_tokens else []) + [len(text)]
        )

        adapted_request = self._convert(messages=[{"role": "user", "content": "hello"}])

        self.assertEqual(adapted_request.input_ids, [len("<user>hello")])
        self.assertIsNone(adapted_request.text)

    def test_request_options_override_chat_template_kwargs(self):
        """chat_template_kwargs reach the template, but add_generation_prompt from
        the request wins over a kwarg of the same name, as in vLLM."""
        self._use_order_template()

        adapted_request = self._convert(
            messages=[{"role": "user", "content": "hello"}],
            chat_template_kwargs={
                "instruction": "query",
                "add_generation_prompt": True,
            },
        )

        self.assertEqual(adapted_request.text, "<user>hello<instruction:query>")

    def test_media_is_rejected_for_text_only_model(self):
        """A text-only model would render the prompt and drop the image silently."""
        self._use_order_template(is_multimodal=False)
        request = EmbeddingRequest(
            model="test-model",
            messages=[{"role": "user", "content": [_text("x"), _image("y")]}],
        )

        self.assertIn(
            "only supports text input",
            self.serving_embedding._validate_request(request),
        )

    def test_media_is_rejected_when_model_lacks_that_modality(self):
        """A vision model given audio (or an audio model given an image) would
        render the placeholder and drop the media silently."""
        cases = [
            (
                "is_audio_understandable_model",
                {"type": "audio_url", "audio_url": {"url": "a.wav"}},
                "audio",
            ),
            ("is_image_understandable_model", _image("y"), "image"),
            ("is_image_understandable_model", _video("v"), "video"),
        ]
        for flag, part, modality in cases:
            with self.subTest(modality=modality):
                self._use_order_template()
                model_config = self.tokenizer_manager.model_config
                model_config.is_audio_understandable_model = True
                model_config.is_image_understandable_model = True
                setattr(model_config, flag, False)
                request = EmbeddingRequest(
                    model="test-model",
                    messages=[{"role": "user", "content": [_text("x"), part]}],
                )

                self.assertIn(
                    f"does not support {modality} input",
                    self.serving_embedding._validate_request(request),
                )

    def test_media_is_rejected_for_string_format_template(self):
        """A "string" content-format template keeps only text parts."""
        self._use_order_template()
        self.template_manager.jinja_template_content_format = "string"
        request = EmbeddingRequest(
            model="test-model",
            messages=[{"role": "user", "content": [_text("x"), _image("y")]}],
        )

        with self.assertRaisesRegex(ValueError, "renders text only"):
            self.serving_embedding._convert_to_internal_request(request)

    def test_input_and_messages_are_mutually_exclusive(self):
        with self.assertRaisesRegex(ValidationError, "exactly one of"):
            EmbeddingRequest(
                model="test-model",
                input="hello",
                messages=[{"role": "user", "content": "hello"}],
            )

    def test_conversation_template_renders_messages_in_order(self):
        """Registered conversation templates (gme-qwen2-vl) render the message
        through generate_chat_conv, keeping the image where it was sent."""
        self.tokenizer_manager.model_config.is_multimodal = True
        self.template_manager.chat_template_name = "gme-qwen2-vl"

        adapted_request = self._convert(
            messages=[
                {
                    "role": "user",
                    "content": [_text("a photo of"), _image("dog"), _text("outside")],
                }
            ]
        )

        self.assertIn(
            "a photo of<|vision_start|><|image_pad|><|vision_end|>outside",
            adapted_request.text,
        )
        self.assertEqual(_urls(adapted_request.image_data), ["dog"])

    def test_conversation_template_prompt_matches_multimodal_input(self):
        """For gme-qwen2-vl, `messages` and the existing `input` form must give the
        same prompt, so both produce the same embedding for the same content."""
        self.tokenizer_manager.model_config.is_multimodal = True
        self.template_manager.chat_template_name = "gme-qwen2-vl"

        from_messages = self._convert(
            messages=[
                {"role": "user", "content": [_image("dog"), _text("a dog outside")]}
            ]
        )
        from_input = self._convert(
            input=[MultimodalEmbeddingInput(text="a dog outside", image="dog")]
        )

        self.assertEqual(from_messages.text, from_input.text)

    def test_conversation_template_rejects_roles_it_cannot_render(self):
        self.tokenizer_manager.model_config.is_multimodal = True
        self.template_manager.chat_template_name = "gme-qwen2-vl"
        request = EmbeddingRequest(
            model="test-model", messages=[{"role": "query", "content": "cat"}]
        )

        with self.assertRaisesRegex(ValueError, "role 'query' is not supported"):
            self.serving_embedding._convert_to_internal_request(request)


if __name__ == "__main__":
    unittest.main(verbosity=2)
