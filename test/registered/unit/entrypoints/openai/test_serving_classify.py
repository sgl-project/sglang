"""Unit tests for OpenAIServingClassify -- no server, no model loading."""

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede imports that may transitively load sgl_kernel

import json
import math
import unittest
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.entrypoints.openai.protocol import ClassifyRequest, ClassifyResponse
from sglang.srt.entrypoints.openai.serving_classify import OpenAIServingClassify
from sglang.srt.managers.io_struct import EmbeddingReqInput
from sglang.srt.utils import get_or_create_event_loop
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


async def _yield_once(value):
    yield value


async def _raise_value_error(message):
    raise ValueError(message)
    yield None


class _FakeTokenizerManager:
    def __init__(
        self,
        *,
        id2label=None,
        num_labels=None,
        served_model_name=None,
        model_path="fallback-model",
        generate_result=None,
        generate_error=None,
    ):
        hf_config = SimpleNamespace(id2label=id2label, num_labels=num_labels)
        self.model_config = SimpleNamespace(hf_config=hf_config)
        self.server_args = SimpleNamespace(model_path=model_path)
        self.served_model_name = served_model_name
        self.request_logger = SimpleNamespace(log_requests=False, log_requests_level=0)

        if generate_error is not None:
            self.generate_request = Mock(
                side_effect=lambda *_, **__: _raise_value_error(generate_error)
            )
        else:
            result = (
                generate_result
                if generate_result is not None
                else {
                    "embedding": [0.25, 1.25],
                    "meta_info": {"prompt_tokens": 4},
                }
            )
            self.generate_request = Mock(
                side_effect=lambda *_, **__: _yield_once(result)
            )


class TestOpenAIServingClassify(CustomTestCase):
    def _serving(self, **manager_kwargs):
        return OpenAIServingClassify(
            _FakeTokenizerManager(**manager_kwargs),
            template_manager=object(),
        )

    def _run(self, coro):
        return get_or_create_event_loop().run_until_complete(coro)

    def test_init_uses_served_model_name_and_configured_labels(self):
        serving = self._serving(
            id2label={0: "negative", 1: "positive"},
            served_model_name="served-classifier",
            model_path="base-path",
        )

        self.assertEqual(serving.id2label, {0: "negative", 1: "positive"})
        self.assertEqual(serving.model_name, "served-classifier")
        self.assertEqual(serving._request_id_prefix(), "classify-")

    def test_init_builds_default_label_mapping_from_num_labels(self):
        serving = self._serving(id2label={}, num_labels=3)

        self.assertEqual(
            serving.id2label,
            {
                0: "LABEL_0",
                1: "LABEL_1",
                2: "LABEL_2",
            },
        )
        self.assertEqual(serving.model_name, "fallback-model")

    def test_init_rejects_missing_label_mapping(self):
        with self.assertRaisesRegex(ValueError, "id2label mapping is missing"):
            self._serving(id2label={}, num_labels=0)

    def test_convert_string_input_to_embedding_text_request(self):
        serving = self._serving(id2label={0: "negative", 1: "positive"})
        request = ClassifyRequest(
            model="test-model",
            input="classify this",
            rid="rid-1",
            priority=7,
        )

        adapted_request, processed_request = serving._convert_to_internal_request(
            request
        )

        self.assertIsInstance(adapted_request, EmbeddingReqInput)
        self.assertEqual(adapted_request.text, "classify this")
        self.assertIsNone(adapted_request.input_ids)
        self.assertEqual(adapted_request.rid, "rid-1")
        self.assertEqual(adapted_request.priority, 7)
        self.assertIs(processed_request, request)

    def test_convert_string_list_input_to_embedding_text_batch(self):
        serving = self._serving(id2label={0: "negative", 1: "positive"})
        request = ClassifyRequest(model="test-model", input=["first", "second"])

        adapted_request, processed_request = serving._convert_to_internal_request(
            request
        )

        self.assertEqual(adapted_request.text, ["first", "second"])
        self.assertIsNone(adapted_request.input_ids)
        self.assertIs(processed_request, request)

    def test_convert_token_id_input_to_embedding_token_ids(self):
        serving = self._serving(id2label={0: "negative", 1: "positive"})
        request = ClassifyRequest(model="test-model", input=[11, 22, 33])

        adapted_request, _ = serving._convert_to_internal_request(request)

        self.assertIsNone(adapted_request.text)
        self.assertEqual(adapted_request.input_ids, [11, 22, 33])

    def test_validate_rejects_empty_and_whitespace_inputs(self):
        serving = self._serving(id2label={0: "negative", 1: "positive"})

        self.assertEqual(
            serving._validate_request(
                ClassifyRequest(model="test-model", input="   \t")
            ),
            "Input cannot be empty or whitespace only",
        )
        self.assertEqual(
            serving._validate_request(ClassifyRequest(model="test-model", input=[])),
            "Input cannot be empty",
        )
        self.assertEqual(
            serving._validate_request(
                ClassifyRequest(model="test-model", input=["ok", ""])
            ),
            "Input at index 1 cannot be empty or whitespace only",
        )

    def test_validate_rejects_mixed_or_negative_token_ids(self):
        serving = self._serving(id2label={0: "negative", 1: "positive"})

        self.assertEqual(
            serving._validate_request(
                ClassifyRequest(model="test-model", input=[1, -1])
            ),
            "Token ID at index 1 must be non-negative",
        )
        self.assertEqual(
            serving._validate_request(
                ClassifyRequest.model_construct(model="test-model", input=[1, "bad"])
            ),
            "All items in input list must be integers",
        )

    def test_build_classify_response_applies_softmax_and_usage_totals(self):
        serving = self._serving(
            id2label={0: "negative", 1: "positive"},
            served_model_name="served-classifier",
        )

        with patch(
            "sglang.srt.entrypoints.openai.serving_classify.time.time",
            return_value=1234567890,
        ), patch(
            "sglang.srt.entrypoints.openai.serving_classify.uuid.uuid4",
            return_value=SimpleNamespace(hex="abc123"),
        ):
            response = serving._build_classify_response(
                [
                    {
                        "embedding": [1.0, 3.0],
                        "meta_info": {"prompt_tokens": 5, "e2e_latency": 0.25},
                    },
                    {
                        "embedding": [4.0, 0.0],
                        "meta_info": {"prompt_tokens": 7, "e2e_latency": 0.5},
                    },
                ]
            )

        self.assertIsInstance(response, ClassifyResponse)
        self.assertEqual(response.id, "classify-abc123")
        self.assertEqual(response.created, 1234567890)
        self.assertEqual(response.model, "served-classifier")
        self.assertEqual(response.usage.prompt_tokens, 12)
        self.assertEqual(response.usage.total_tokens, 12)
        self.assertEqual(response.usage.completion_tokens, 0)
        self.assertEqual(
            [item.label for item in response.data], ["positive", "negative"]
        )
        self.assertEqual([item.index for item in response.data], [0, 1])
        self.assertTrue(math.isclose(sum(response.data[0].probs), 1.0, rel_tol=1e-6))
        self.assertEqual(response.data[0].num_classes, 2)
        self.assertGreater(response.data[0].probs[1], response.data[0].probs[0])

    def test_build_classify_response_uses_default_for_empty_embedding(self):
        serving = self._serving(id2label={0: "negative", 1: "positive"})

        response = serving._build_classify_response(
            [{"embedding": [], "meta_info": {"prompt_tokens": 3}}]
        )

        self.assertEqual(response.data[0].label, "Default")
        self.assertEqual(response.data[0].probs, [1.0])
        self.assertEqual(response.data[0].num_classes, 1)
        self.assertEqual(response.usage.prompt_tokens, 3)

    def test_handle_non_streaming_request_wraps_single_generation_result(self):
        manager = _FakeTokenizerManager(
            id2label={0: "negative", 1: "positive"},
            generate_result={
                "embedding": [0.0, 2.0],
                "meta_info": {"prompt_tokens": 9},
            },
        )
        serving = OpenAIServingClassify(manager, template_manager=object())
        request = ClassifyRequest(model="test-model", input="hello")
        adapted_request, processed_request = serving._convert_to_internal_request(
            request
        )
        raw_request = Mock()

        response = self._run(
            serving._handle_non_streaming_request(
                adapted_request, processed_request, raw_request
            )
        )

        self.assertIsInstance(response, ClassifyResponse)
        self.assertEqual(response.data[0].label, "positive")
        self.assertEqual(response.usage.prompt_tokens, 9)
        manager.generate_request.assert_called_once_with(adapted_request, raw_request)

    def test_handle_non_streaming_request_converts_generation_value_error(self):
        serving = self._serving(
            id2label={0: "negative", 1: "positive"},
            generate_error="bad classify input",
        )
        request = ClassifyRequest(model="test-model", input="hello")
        adapted_request, processed_request = serving._convert_to_internal_request(
            request
        )

        response = self._run(
            serving._handle_non_streaming_request(
                adapted_request, processed_request, Mock()
            )
        )
        payload = json.loads(response.body)

        self.assertEqual(response.status_code, HTTPStatus.BAD_REQUEST)
        self.assertEqual(payload["type"], "BadRequestError")
        self.assertEqual(payload["message"], "bad classify input")


if __name__ == "__main__":
    unittest.main(verbosity=2)
