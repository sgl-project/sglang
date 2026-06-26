"""Unit tests for OpenAI rerank serving helpers."""

import math
import unittest
from unittest.mock import Mock

from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

# Stub CUDA-only deps before importing serving modules on CPU-only runners.
maybe_stub_sgl_kernel()

from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionMessageContentImagePart,
    ChatCompletionMessageContentImageURL,
    ChatCompletionMessageContentTextPart,
    ChatCompletionMessageContentVideoPart,
    ChatCompletionMessageContentVideoURL,
    V1RerankReqInput,
)
from sglang.srt.entrypoints.openai.serving_rerank import (
    OpenAIServingRerank,
    _detect_rerank_backend,
    _extract_text_from_content,
    _get_yes_no_token_ids,
    _is_qwen3_reranker_template,
    _is_qwen3_vl_model,
    _is_qwen3_vl_reranker_template,
    _qwen3_rerank_score,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


QWEN3_RERANK_TEMPLATE = 'The answer can only be "yes" or "no".'
QWEN3_VL_RERANK_TEMPLATE = QWEN3_RERANK_TEMPLATE + " <|vision_start|>"


class _MockTokenizer:
    def __init__(self, chat_template=""):
        self.chat_template = chat_template
        self.encode = Mock(return_value=[1])
        self.convert_tokens_to_ids = Mock(return_value=None)


class _MockTokenizerManager:
    def __init__(
        self, *, chat_template="", model_path="test-model", is_generation=True
    ):
        self.server_args = object()
        self.model_config = Mock()
        self.model_config.model_path = model_path
        self.model_config.is_generation = is_generation
        self.tokenizer = _MockTokenizer(chat_template=chat_template)


class ServingRerankHelperTestCase(CustomTestCase):
    def setUp(self):
        self.handler = OpenAIServingRerank(_MockTokenizerManager())

    def test_get_yes_no_token_ids_from_single_token_encode(self):
        tokenizer = Mock()
        tokenizer.encode.side_effect = [[101], [202]]

        self.assertEqual(_get_yes_no_token_ids(tokenizer), (101, 202))
        tokenizer.convert_tokens_to_ids.assert_not_called()

    def test_get_yes_no_token_ids_falls_back_to_convert_tokens_to_ids(self):
        tokenizer = Mock()
        tokenizer.encode.side_effect = [[1, 2], [3, 4]]
        tokenizer.convert_tokens_to_ids.side_effect = [303, 404]

        self.assertEqual(_get_yes_no_token_ids(tokenizer), (303, 404))

    def test_get_yes_no_token_ids_uses_known_fallback_on_error(self):
        tokenizer = Mock()
        tokenizer.encode.side_effect = RuntimeError("tokenizer unavailable")

        self.assertEqual(_get_yes_no_token_ids(tokenizer), (9693, 2152))

    def test_qwen3_template_detection(self):
        self.assertTrue(_is_qwen3_reranker_template(QWEN3_RERANK_TEMPLATE))
        self.assertTrue(
            _is_qwen3_reranker_template('ANSWER CAN ONLY BE "YES" OR "NO"')
        )
        self.assertFalse(_is_qwen3_reranker_template("plain chat template"))
        self.assertFalse(_is_qwen3_reranker_template(""))

    def test_qwen3_vl_template_detection_requires_vision_token(self):
        self.assertTrue(_is_qwen3_vl_reranker_template(QWEN3_VL_RERANK_TEMPLATE))
        self.assertTrue(
            _is_qwen3_vl_reranker_template(
                QWEN3_RERANK_TEMPLATE + " <|image_pad|>"
            )
        )
        self.assertFalse(_is_qwen3_vl_reranker_template(QWEN3_RERANK_TEMPLATE))
        self.assertFalse(_is_qwen3_vl_reranker_template("<|vision_start|>"))

    def test_qwen3_vl_model_detection(self):
        self.assertTrue(_is_qwen3_vl_model("Qwen/Qwen3-VL-Reranker"))
        self.assertTrue(_is_qwen3_vl_model("local/qwen3vl-reranker"))
        self.assertFalse(_is_qwen3_vl_model("Qwen/Qwen3-Reranker"))
        self.assertFalse(_is_qwen3_vl_model(""))

    def test_detect_rerank_backend(self):
        text_request = V1RerankReqInput(query="q", documents=["d"])
        multimodal_request = V1RerankReqInput(
            query=[ChatCompletionMessageContentTextPart(type="text", text="q")],
            documents=["d"],
        )

        self.assertEqual(
            _detect_rerank_backend(
                request=text_request,
                chat_template=QWEN3_VL_RERANK_TEMPLATE,
                model_path="test-model",
            ),
            "vl_decoder",
        )
        self.assertEqual(
            _detect_rerank_backend(
                request=text_request,
                chat_template="plain template",
                model_path="Qwen/Qwen3-VL-Reranker",
            ),
            "vl_decoder",
        )
        self.assertEqual(
            _detect_rerank_backend(
                request=multimodal_request,
                chat_template=QWEN3_RERANK_TEMPLATE,
                model_path="test-model",
            ),
            "vl_decoder",
        )
        self.assertEqual(
            _detect_rerank_backend(
                request=text_request,
                chat_template=QWEN3_RERANK_TEMPLATE,
                model_path="test-model",
            ),
            "text_decoder",
        )
        self.assertEqual(
            _detect_rerank_backend(
                request=text_request,
                chat_template=None,
                model_path="test-model",
            ),
            "cross_encoder",
        )

    def test_qwen3_rerank_score(self):
        self.assertAlmostEqual(_qwen3_rerank_score(0.8, 0.2), 0.8)
        self.assertAlmostEqual(_qwen3_rerank_score(0.0, 0.5), 0.0)
        self.assertAlmostEqual(_qwen3_rerank_score(0.0, 0.0), 0.0)

    def test_extract_text_from_multimodal_content(self):
        content = [
            ChatCompletionMessageContentTextPart(type="text", text="hello"),
            {"type": "text", "text": "world"},
            {"type": "image_url", "image_url": {"url": "image-data"}},
        ]

        self.assertEqual(_extract_text_from_content("plain text"), "plain text")
        self.assertEqual(_extract_text_from_content(content), "hello world")

    def test_content_to_template_list_collects_media_data(self):
        image_data = []
        video_data = []
        content = [
            ChatCompletionMessageContentTextPart(type="text", text="caption"),
            ChatCompletionMessageContentImagePart(
                type="image_url",
                image_url=ChatCompletionMessageContentImageURL(url="image-a"),
            ),
            ChatCompletionMessageContentVideoPart(
                type="video_url",
                video_url=ChatCompletionMessageContentVideoURL(url="video-a"),
            ),
            {"type": "image_url", "image_url": {"url": "image-b"}},
            {"type": "video_url", "video_url": "video-b"},
        ]

        result = self.handler._content_to_template_list(content, image_data, video_data)

        self.assertEqual(
            result,
            [
                {"type": "text", "text": "caption"},
                {"type": "image"},
                {"type": "video"},
                {"type": "image"},
                {"type": "video"},
            ],
        )
        self.assertEqual(image_data, ["image-a", "image-b"])
        self.assertEqual(video_data, ["video-a", "video-b"])

    def test_extract_score_from_logprobs(self):
        self.handler._yes_token_id = 11
        self.handler._no_token_id = 22
        ret = {
            "meta_info": {
                "output_top_logprobs": [
                    [(math.log(0.75), 11, "yes"), (math.log(0.25), 22, "no")]
                ]
            }
        }

        self.assertAlmostEqual(self.handler._extract_score_from_logprobs(ret), 0.75)
        self.assertEqual(self.handler._extract_score_from_logprobs({}), 0.0)


class ServingRerankRequestTestCase(CustomTestCase):
    def setUp(self):
        self.handler = OpenAIServingRerank(_MockTokenizerManager())

    def test_validate_request_rejects_empty_query_and_documents(self):
        self.assertEqual(
            self.handler._validate_request(V1RerankReqInput(query="", documents=["d"])),
            "Query cannot be empty",
        )
        self.assertEqual(
            self.handler._validate_request(
                V1RerankReqInput(query="   ", documents=["d"])
            ),
            "Query cannot be empty or whitespace only",
        )
        self.assertEqual(
            self.handler._validate_request(V1RerankReqInput(query="q", documents=[])),
            "Documents cannot be empty",
        )
        self.assertEqual(
            self.handler._validate_request(
                V1RerankReqInput(query="q", documents=["valid", " "])
            ),
            "Each document cannot be empty or whitespace only",
        )

    def test_convert_to_internal_request_cross_encoder_text_pairs(self):
        req = V1RerankReqInput(query="q", documents=["doc-a", "doc-b"])

        adapted, processed = self.handler._convert_to_internal_request(req)

        self.assertTrue(adapted.is_cross_encoder_request)
        self.assertEqual(adapted.text, [["q", "doc-a"], ["q", "doc-b"]])
        self.assertIs(processed, req)

    def test_convert_to_internal_request_cross_encoder_extracts_multimodal_text(self):
        req = V1RerankReqInput(
            query=[
                ChatCompletionMessageContentTextPart(type="text", text="query text"),
                {"type": "image_url", "image_url": {"url": "query-image"}},
            ],
            documents=[
                [
                    ChatCompletionMessageContentTextPart(type="text", text="doc text"),
                    {"type": "image_url", "image_url": {"url": "doc-image"}},
                ]
            ],
        )

        adapted, _ = self.handler._convert_to_internal_request(req)

        self.assertTrue(adapted.is_cross_encoder_request)
        self.assertEqual(adapted.text, [["query text", "doc text"]])

    def test_convert_to_internal_request_decoder_reranker_keeps_original_request(self):
        handler = OpenAIServingRerank(
            _MockTokenizerManager(chat_template=QWEN3_RERANK_TEMPLATE)
        )
        req = V1RerankReqInput(query="q", documents=["d"])

        adapted, processed = handler._convert_to_internal_request(req)

        self.assertIs(adapted, req)
        self.assertIs(processed, req)

    def test_build_rerank_response_sorts_scores_and_preserves_indices(self):
        req = V1RerankReqInput(
            query="q", documents=["doc-a", "doc-b", "doc-c"], return_documents=True
        )

        res = self.handler._build_rerank_response([0.2, 0.9, 0.1], req)

        self.assertEqual([r.document for r in res], ["doc-b", "doc-a", "doc-c"])
        self.assertEqual([r.index for r in res], [1, 0, 2])
        self.assertEqual([r.score for r in res], [0.9, 0.2, 0.1])

    def test_build_rerank_response_uses_embedding_score_and_meta_info(self):
        req = V1RerankReqInput(query="q", documents=["a", "b"], return_documents=True)
        ret = [
            {"embedding": [0.1, 0.2], "meta_info": {"id": "a"}},
            {"embedding": [0.8], "meta_info": {"id": "b"}},
        ]

        res = self.handler._build_rerank_response(ret, req)

        self.assertEqual([r.document for r in res], ["b", "a"])
        self.assertEqual([r.index for r in res], [1, 0])
        self.assertEqual([r.meta_info for r in res], [{"id": "b"}, {"id": "a"}])

    def test_build_rerank_response_respects_return_documents_false_and_top_n(self):
        req = V1RerankReqInput(
            query="q",
            documents=["a", "b", "c"],
            return_documents=False,
            top_n=2,
        )

        res = self.handler._build_rerank_response([0.2, 0.9, 0.1], req)

        self.assertEqual(len(res), 2)
        self.assertEqual([r.document for r in res], [None, None])
        self.assertEqual([r.index for r in res], [1, 0])
        self.assertEqual([r.score for r in res], [0.9, 0.2])

    def test_build_rerank_response_rejects_invalid_embedding_score(self):
        req = V1RerankReqInput(query="q", documents=["a"], return_documents=True)

        with self.assertRaisesRegex(ValueError, "Invalid embedding score"):
            self.handler._build_rerank_response([{"embedding": []}], req)


if __name__ == "__main__":
    unittest.main()
