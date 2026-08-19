"""
Unit tests for serving_rerank.py -- no server, no model loading.

Tests the pure helper functions used by OpenAIServingRerank:
  - _is_qwen3_reranker_template
  - _is_qwen3_vl_reranker_template
  - _is_qwen3_vl_model
  - _qwen3_rerank_score
  - _extract_text_from_content
  - _detect_rerank_backend
"""

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import unittest
from unittest.mock import Mock

from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionMessageContentTextPart,
    V1RerankReqInput,
)
from sglang.srt.entrypoints.openai.serving_rerank import (
    _detect_rerank_backend,
    _extract_text_from_content,
    _is_qwen3_reranker_template,
    _is_qwen3_vl_model,
    _is_qwen3_vl_reranker_template,
    _qwen3_rerank_score,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestIsQwen3RerankerTemplate(unittest.TestCase):
    """Tests for _is_qwen3_reranker_template."""

    def test_exact_phrase(self):
        tmpl = 'You are a reranker. The answer can only be "yes" or "no".'
        self.assertTrue(_is_qwen3_reranker_template(tmpl))

    def test_case_insensitive(self):
        tmpl = 'The Answer Can Only Be "Yes" or "No"'
        self.assertTrue(_is_qwen3_reranker_template(tmpl))

    def test_variant_phrase(self):
        tmpl = "Answer can only be yes or no"
        self.assertTrue(_is_qwen3_reranker_template(tmpl))

    def test_empty_string(self):
        self.assertFalse(_is_qwen3_reranker_template(""))

    def test_none(self):
        self.assertFalse(_is_qwen3_reranker_template(None))

    def test_unrelated_template(self):
        tmpl = "You are a helpful assistant. Answer the user's question."
        self.assertFalse(_is_qwen3_reranker_template(tmpl))

    def test_partial_match_no_both_quotes(self):
        tmpl = 'Answer can only be "yes"'
        self.assertFalse(_is_qwen3_reranker_template(tmpl))


class TestIsQwen3VlRerankerTemplate(unittest.TestCase):
    "Tests for _is_qwen3_vl_reranker_template."

    def test_reranker_with_image_token(self):
        tmpl = '<|vision_start|>...<|vision_end|>Answer can only be "yes" or "no".'
        self.assertTrue(_is_qwen3_vl_reranker_template(tmpl))

    def test_reranker_with_image_placeholder(self):
        tmpl = '<|vision_start|>...<|vision_end|>Answer can only be "yes" or "no".'
        self.assertTrue(_is_qwen3_vl_reranker_template(tmpl))

    def test_text_only_not_vl(self):
        tmpl = 'Answer can only be "yes" or "no".'
        self.assertFalse(_is_qwen3_vl_reranker_template(tmpl))

    def test_vision_token_without_reranker(self):
        tmpl = "<|vision_start|>...<|vision_end|>Describe this image."
        self.assertFalse(_is_qwen3_vl_reranker_template(tmpl))

    def test_empty_string(self):
        self.assertFalse(_is_qwen3_vl_reranker_template(""))

    def test_none(self):
        self.assertFalse(_is_qwen3_vl_reranker_template(None))


class TestIsQwen3VlModel(unittest.TestCase):
    "Tests for _is_qwen3_vl_model."

    def test_qwen3_vl_hyphenated(self):
        self.assertTrue(_is_qwen3_vl_model("Qwen3-VL-7B"))

    def test_qwen3_vl_no_hyphen(self):
        self.assertTrue(_is_qwen3_vl_model("qwen3vl"))

    def test_case_insensitive(self):
        self.assertTrue(_is_qwen3_vl_model("QWEN3-VL"))

    def test_non_vl_model(self):
        self.assertFalse(_is_qwen3_vl_model("Qwen3-7B"))

    def test_empty_string(self):
        self.assertFalse(_is_qwen3_vl_model(""))

    def test_none(self):
        self.assertFalse(_is_qwen3_vl_model(None))

    def test_partial_match(self):
        self.assertFalse(_is_qwen3_vl_model("my-qwen3-vl-tool"))


class TestQwen3RerankScore(unittest.TestCase):
    "Tests for _qwen3_rerank_score."

    def test_equal_probabilities(self):
        self.assertAlmostEqual(_qwen3_rerank_score(0.5, 0.5), 0.5)

    def test_all_yes(self):
        self.assertAlmostEqual(_qwen3_rerank_score(1.0, 0.0), 1.0)

    def test_all_no(self):
        self.assertAlmostEqual(_qwen3_rerank_score(0.0, 1.0), 0.0)

    def test_higher_yes(self):
        score = _qwen3_rerank_score(0.8, 0.2)
        self.assertAlmostEqual(score, 0.8)

    def test_both_zero_returns_zero(self):
        self.assertEqual(_qwen3_rerank_score(0.0, 0.0), 0.0)

    def test_negative_denominator_returns_zero(self):
        self.assertEqual(_qwen3_rerank_score(-1.0, -1.0), 0.0)


class TestExtractTextFromContent(unittest.TestCase):
    "Tests for _extract_text_from_content."

    def test_string_input(self):
        self.assertEqual(_extract_text_from_content("hello"), "hello")

    def test_text_parts(self):
        parts = [
            ChatCompletionMessageContentTextPart(type="text", text="hello "),
            ChatCompletionMessageContentTextPart(type="text", text="world"),
        ]
        self.assertEqual(_extract_text_from_content(parts), "hello world")

    def test_dict_text_parts(self):
        parts = [{"type": "text", "text": "foo"}, {"type": "text", "text": "bar"}]
        self.assertEqual(_extract_text_from_content(parts), "foo bar")

    def test_mixed_types_skips_non_text(self):
        parts = [
            {"type": "text", "text": "keep"},
            {"type": "image", "image_url": {"url": "http://img.png"}},
        ]
        self.assertEqual(_extract_text_from_content(parts), "keep")

    def test_empty_list(self):
        self.assertEqual(_extract_text_from_content([]), "")

    def test_single_text_part(self):
        parts = [ChatCompletionMessageContentTextPart(type="text", text="solo")]
        self.assertEqual(_extract_text_from_content(parts), "solo")


class TestDetectRerankBackend(unittest.TestCase):
    "Tests for _detect_rerank_backend."

    def _make_request(self, is_multimodal: bool = False) -> V1RerankReqInput:
        req = Mock(spec=V1RerankReqInput)
        req.is_multimodal = Mock(return_value=is_multimodal)
        return req

    def test_text_template_returns_text_decoder(self):
        req = self._make_request()
        tmpl = 'Answer can only be "yes" or "no".'
        result = _detect_rerank_backend(
            request=req, chat_template=tmpl, model_path="some-model"
        )
        self.assertEqual(result, "text_decoder")

    def test_vl_template_returns_vl_decoder(self):
        req = self._make_request()
        tmpl = '<think>...</think>Answer can only be "yes" or "no".'
        result = _detect_rerank_backend(
            request=req, chat_template=tmpl, model_path="some-model"
        )
        self.assertEqual(result, "vl_decoder")

    def test_vl_model_returns_vl_decoder(self):
        req = self._make_request()
        result = _detect_rerank_backend(
            request=req, chat_template=None, model_path="Qwen3-VL-7B"
        )
        self.assertEqual(result, "vl_decoder")

    def test_multimodal_with_text_template_returns_vl_decoder(self):
        req = self._make_request(is_multimodal=True)
        tmpl = 'Answer can only be "yes" or "no".'
        result = _detect_rerank_backend(
            request=req, chat_template=tmpl, model_path="some-model"
        )
        self.assertEqual(result, "vl_decoder")

    def test_no_template_no_vl_returns_cross_encoder(self):
        req = self._make_request()
        result = _detect_rerank_backend(
            request=req, chat_template=None, model_path="some-model"
        )
        self.assertEqual(result, "cross_encoder")

    def test_unrelated_template_returns_cross_encoder(self):
        req = self._make_request()
        tmpl = "You are a helpful assistant."
        result = _detect_rerank_backend(
            request=req, chat_template=tmpl, model_path="some-model"
        )
        self.assertEqual(result, "cross_encoder")


if __name__ == "__main__":
    unittest.main()
