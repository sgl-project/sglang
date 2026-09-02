"""CPU-only unit tests for the OpenAI rerank serving helpers."""

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import math
import unittest

from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionMessageContentImagePart,
    ChatCompletionMessageContentTextPart,
    ChatCompletionMessageContentVideoPart,
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
    _render_jinja_chat_template,
    _render_vl_jinja_template,
)
from sglang.srt.managers.io_struct import EmbeddingReqInput
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


_RERANK_TEMPLATE = 'The answer can only be "yes" or "no".'
_VL_RERANK_TEMPLATE = _RERANK_TEMPLATE + " <|vision_start|><|image_pad|>"


class _Tokenizer:
    def __init__(self, chat_template=""):
        self.chat_template = chat_template

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return {"yes": [101], "no": [202]}[text]


class _ModelConfig:
    def __init__(self, model_path="test/cross-encoder"):
        self.model_path = model_path
        self.is_generation = True


class _TokenizerManager:
    def __init__(self, chat_template="", model_path="test/cross-encoder"):
        self.server_args = object()
        self.tokenizer = _Tokenizer(chat_template)
        self.model_config = _ModelConfig(model_path)


def _text_part(text):
    return ChatCompletionMessageContentTextPart(type="text", text=text)


def _image_part(url="data:image/png;base64,image"):
    return ChatCompletionMessageContentImagePart(
        type="image_url", image_url={"url": url}
    )


def _video_part(url="data:video/mp4;base64,video"):
    return ChatCompletionMessageContentVideoPart(
        type="video_url", video_url={"url": url}
    )


class TestRerankBackendDetection(CustomTestCase):
    def test_backend_routing_matrix_uses_request_template_and_model(self):
        text_request = V1RerankReqInput(query="query", documents=["document"])
        multimodal_request = V1RerankReqInput(
            query=[_text_part("query"), _image_part()], documents=["document"]
        )
        cases = [
            (text_request, None, "test/cross-encoder", "cross_encoder"),
            (text_request, _RERANK_TEMPLATE, "Qwen/Qwen3-Reranker", "text_decoder"),
            (text_request, _VL_RERANK_TEMPLATE, "test/model", "vl_decoder"),
            (text_request, None, "QWEN/QWEN3VL-8B", "vl_decoder"),
            (
                multimodal_request,
                _RERANK_TEMPLATE,
                "test/model",
                "vl_decoder",
            ),
        ]

        for request, template, model_path, expected in cases:
            with self.subTest(expected=expected, model_path=model_path):
                self.assertEqual(
                    _detect_rerank_backend(
                        request=request,
                        chat_template=template,
                        model_path=model_path,
                    ),
                    expected,
                )

    def test_template_detectors_require_their_distinguishing_markers(self):
        text_cases = [
            (_RERANK_TEMPLATE.upper(), True),
            ('answer can only be something: "yes" or "no"', True),
            ('answer can only be "yes"', False),
            ("ordinary chat template", False),
            ("", False),
        ]
        for template, expected in text_cases:
            with self.subTest(detector="text", template=template):
                self.assertEqual(_is_qwen3_reranker_template(template), expected)

        vl_cases = [
            (_RERANK_TEMPLATE + " <|vision_start|>", True),
            (_RERANK_TEMPLATE.upper() + " <|IMAGE_PAD|>", True),
            (_RERANK_TEMPLATE, False),
            ("<|vision_start|> describe the image", False),
        ]
        for template, expected in vl_cases:
            with self.subTest(detector="vl", template=template):
                self.assertEqual(_is_qwen3_vl_reranker_template(template), expected)

    def test_vl_model_detector_accepts_supported_spellings_only(self):
        cases = [
            ("Qwen/Qwen3-VL-8B", True),
            ("local/QWEN3VL-reranker", True),
            ("Qwen/Qwen3-Reranker", False),
            ("Qwen/Qwen2-VL-7B", False),
            ("", False),
        ]
        for model_path, expected in cases:
            with self.subTest(model_path=model_path):
                self.assertEqual(_is_qwen3_vl_model(model_path), expected)


class TestRerankHelpers(CustomTestCase):
    def test_qwen3_score_normalizes_yes_no_mass_and_guards_denominator(self):
        cases = [
            (0.8, 0.2, 0.8),
            (0.0, 0.0, 0.0),
            (-0.1, 0.05, 0.0),
        ]
        for p_yes, p_no, expected in cases:
            with self.subTest(p_yes=p_yes, p_no=p_no):
                self.assertAlmostEqual(_qwen3_rerank_score(p_yes, p_no), expected)

    def test_extract_text_preserves_order_and_ignores_media(self):
        content = [
            _text_part("first"),
            _image_part(),
            {"type": "text", "text": "second"},
            _video_part(),
            {"type": "text", "text": "third"},
        ]

        self.assertEqual(_extract_text_from_content("direct"), "direct")
        self.assertEqual(_extract_text_from_content(content), "first second third")

    def test_yes_no_token_ids_use_single_token_encoding(self):
        self.assertEqual(_get_yes_no_token_ids(_Tokenizer()), (101, 202))

    def test_yes_no_token_ids_fall_back_to_token_conversion(self):
        class MultiTokenTokenizer:
            def encode(self, _text, add_special_tokens=False):
                del add_special_tokens
                return [1, 2]

            def convert_tokens_to_ids(self, text):
                return {"yes": 303, "no": 404}[text]

        self.assertEqual(_get_yes_no_token_ids(MultiTokenTokenizer()), (303, 404))

    def test_yes_no_token_ids_have_a_final_qwen_fallback(self):
        class BrokenTokenizer:
            def encode(self, _text, add_special_tokens=False):
                del add_special_tokens
                raise RuntimeError("tokenizer unavailable")

        self.assertEqual(_get_yes_no_token_ids(BrokenTokenizer()), (9693, 2152))

    def test_text_template_rendering_extracts_text_and_defaults_instruction(self):
        template = (
            "{{ instruct | default('DEFAULT') }}|"
            "{{ messages[0].content }}|{{ messages[1].content }}"
        )
        query = [_text_part("query text"), _image_part()]
        document = [_video_part(), _text_part("document text")]

        self.assertEqual(
            _render_jinja_chat_template(
                template,
                query=query,
                document=document,
                instruct=None,
            ),
            "DEFAULT|query text|document text",
        )
        self.assertEqual(
            _render_jinja_chat_template(
                template,
                query="query",
                document="document",
                instruct="custom",
            ),
            "custom|query|document",
        )

    def test_vl_template_rendering_receives_content_lists_and_instruction(self):
        template = (
            "{{ instruct | default('DEFAULT') }}|"
            "{{ query[0].text }}|{{ document[0].type }}"
        )
        query = [{"type": "text", "text": "query"}]
        document = [{"type": "image"}]

        self.assertEqual(
            _render_vl_jinja_template(
                template,
                query=query,
                document=document,
                instruct="custom",
            ),
            "custom|query|image",
        )


class TestOpenAIServingRerank(CustomTestCase):
    def setUp(self):
        self.serving = OpenAIServingRerank(_TokenizerManager())

    def test_request_validation_rejects_empty_text_boundaries(self):
        cases = [
            (V1RerankReqInput(query="", documents=["document"]), "Query cannot"),
            (V1RerankReqInput(query="  ", documents=["document"]), "Query cannot"),
            (V1RerankReqInput(query="query", documents=[]), "Documents cannot"),
            (
                V1RerankReqInput(query="query", documents=[""]),
                "Each document must",
            ),
            (
                V1RerankReqInput(query="query", documents=[" \t"]),
                "Each document cannot",
            ),
        ]
        for request, message_fragment in cases:
            with self.subTest(request=request):
                self.assertIn(message_fragment, self.serving._validate_request(request))

        valid = V1RerankReqInput(query="query", documents=["document"])
        self.assertIsNone(self.serving._validate_request(valid))

    def test_cross_encoder_conversion_creates_query_document_pairs(self):
        request = V1RerankReqInput(
            query="query", documents=["document one", "document two"]
        )

        adapted, original = self.serving._convert_to_internal_request(request)

        self.assertIsInstance(adapted, EmbeddingReqInput)
        self.assertTrue(adapted.is_cross_encoder_request)
        self.assertEqual(
            adapted.text,
            [["query", "document one"], ["query", "document two"]],
        )
        self.assertIs(original, request)

    def test_cross_encoder_conversion_extracts_multimodal_text(self):
        request = V1RerankReqInput(
            query=[_text_part("query"), _image_part()],
            documents=[
                [_video_part(), _text_part("first")],
                [_text_part("second"), _image_part()],
            ],
        )

        adapted, _ = self.serving._convert_to_internal_request(request)

        self.assertEqual(adapted.text, [["query", "first"], ["query", "second"]])

    def test_decoder_conversions_preserve_the_original_request(self):
        request = V1RerankReqInput(query="query", documents=["document"])
        cases = [
            (_RERANK_TEMPLATE, "test/model"),
            ("", "Qwen/Qwen3-VL-8B"),
        ]
        for chat_template, model_path in cases:
            with self.subTest(chat_template=chat_template, model_path=model_path):
                serving = OpenAIServingRerank(
                    _TokenizerManager(chat_template, model_path)
                )
                adapted, original = serving._convert_to_internal_request(request)
                self.assertIs(adapted, request)
                self.assertIs(original, request)

    def test_logprob_score_uses_token_ids_independent_of_entry_order(self):
        ret = {
            "meta_info": {
                "output_top_logprobs": [
                    [
                        (math.log(0.1), 999, "other"),
                        (math.log(0.2), 202, "no"),
                        (math.log(0.8), 101, "yes"),
                    ]
                ]
            }
        }

        self.assertAlmostEqual(self.serving._extract_score_from_logprobs(ret), 0.8)

    def test_logprob_score_handles_missing_yes_or_no_entries(self):
        yes_only = {
            "meta_info": {"output_top_logprobs": [[(math.log(0.6), 101, "yes")]]}
        }
        no_only = {"meta_info": {"output_top_logprobs": [[(math.log(0.6), 202, "no")]]}}

        self.assertEqual(self.serving._extract_score_from_logprobs(yes_only), 1.0)
        self.assertEqual(self.serving._extract_score_from_logprobs(no_only), 0.0)
        self.assertEqual(self.serving._extract_score_from_logprobs({}), 0.0)

    def test_response_accepts_scalar_embedding_scores(self):
        request = V1RerankReqInput(
            query="query", documents=["low", "high"], return_documents=True
        )
        results = [
            {"embedding": 0.2, "meta_info": {"id": "low"}},
            {"embedding": 0.9, "meta_info": {"id": "high"}},
        ]

        responses = self.serving._build_rerank_response(results, request)

        self.assertEqual(
            [(response.document, response.index) for response in responses],
            [("high", 1), ("low", 0)],
        )
        self.assertEqual(responses[0].meta_info, {"id": "high"})

    def test_response_rejects_malformed_embedding_lists(self):
        request = V1RerankReqInput(query="query", documents=["document"])

        for embedding in ([], ["not-a-number"]):
            with self.subTest(embedding=embedding):
                with self.assertRaisesRegex(ValueError, "Invalid embedding score"):
                    self.serving._build_rerank_response(
                        [{"embedding": embedding}], request
                    )


if __name__ == "__main__":
    unittest.main()
