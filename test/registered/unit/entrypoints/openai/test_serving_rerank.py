"""Unit tests for entrypoints/openai/serving_rerank.py — no server, no model loading."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_cpu_ci(est_time=6, suite="base-b-test-cpu")

import unittest
from unittest.mock import MagicMock

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
from sglang.test.test_utils import CustomTestCase


class TestQwen3RerankScore(CustomTestCase):
    def test_normal_score(self):
        self.assertAlmostEqual(_qwen3_rerank_score(0.8, 0.2), 0.8)

    def test_score_symmetry(self):
        score = _qwen3_rerank_score(0.3, 0.1)
        self.assertAlmostEqual(score, 0.75)

    def test_zero_denominator_returns_zero(self):
        self.assertEqual(_qwen3_rerank_score(0.0, 0.0), 0.0)

    def test_negative_denominator_does_not_crash(self):
        result = _qwen3_rerank_score(0.5, -0.6)
        self.assertLessEqual(result, 0.0)


class TestTemplateDetection(CustomTestCase):
    def test_reranker_template_with_yes_no_phrase(self):
        tmpl = 'Answer can only be "yes" or "no".'
        self.assertTrue(_is_qwen3_reranker_template(tmpl))

    def test_reranker_template_with_alt_phrase(self):
        tmpl = 'Answer can only be "yes" / "no". But the content...'
        self.assertTrue(_is_qwen3_reranker_template(tmpl))

    def test_non_reranker_template_returns_false(self):
        self.assertFalse(_is_qwen3_reranker_template(""))
        self.assertFalse(_is_qwen3_reranker_template("Just a normal chat template."))

    def test_vl_template_with_vision_tokens_and_reranker_phrase(self):
        tmpl = 'Answer can only be "yes" or "no". <|vision_start|>image<|vision_end|>'
        self.assertTrue(_is_qwen3_vl_reranker_template(tmpl))

    def test_vl_template_with_image_pad(self):
        tmpl = 'answer can only be "yes" or "no". <|image_pad|>'
        self.assertTrue(_is_qwen3_vl_reranker_template(tmpl))

    def test_vl_template_without_reranker_phrase_returns_false(self):
        self.assertFalse(
            _is_qwen3_vl_reranker_template("<|vision_start|>image<|vision_end|>")
        )

    def test_qwen3_vl_model_path_detected(self):
        self.assertTrue(_is_qwen3_vl_model("Qwen/Qwen3-VL-7B"))
        self.assertTrue(_is_qwen3_vl_model("some/qwen3vl-2b"))
        self.assertTrue(_is_qwen3_vl_model("local/qwen3-VL-instruct"))

    def test_non_vl_model_path_returns_false(self):
        self.assertFalse(_is_qwen3_vl_model(""))
        self.assertFalse(_is_qwen3_vl_model("Qwen/Qwen2.5-7B"))


class TestDetectRerankBackend(CustomTestCase):
    def _req(self, multimodal=False):
        req = MagicMock(spec=V1RerankReqInput)
        req.is_multimodal = MagicMock(return_value=multimodal)
        return req

    def test_text_reranker_template_returns_text_decoder(self):
        result = _detect_rerank_backend(
            request=self._req(False),
            chat_template='Answer can only be "yes" or "no".',
            model_path="Qwen/Qwen3-0.5B",
        )
        self.assertEqual(result, "text_decoder")

    def test_vl_reranker_template_returns_vl_decoder(self):
        result = _detect_rerank_backend(
            request=self._req(False),
            chat_template='Answer can only be "yes" or "no". <|vision_start|>',
            model_path="Qwen/Qwen3-0.5B",
        )
        self.assertEqual(result, "vl_decoder")

    def test_vl_model_path_returns_vl_decoder(self):
        result = _detect_rerank_backend(
            request=self._req(False),
            chat_template="",
            model_path="Qwen/Qwen3-VL-7B",
        )
        self.assertEqual(result, "vl_decoder")

    def test_multimodal_with_text_template_returns_vl_decoder(self):
        result = _detect_rerank_backend(
            request=self._req(True),
            chat_template='Answer can only be "yes" or "no".',
            model_path="Qwen/Qwen3-0.5B",
        )
        self.assertEqual(result, "vl_decoder")

    def test_no_template_no_vl_model_returns_cross_encoder(self):
        result = _detect_rerank_backend(
            request=self._req(False),
            chat_template="Some other template",
            model_path="some-model",
        )
        self.assertEqual(result, "cross_encoder")


class TestExtractTextFromContent(CustomTestCase):
    def test_plain_string_is_returned_unchanged(self):
        self.assertEqual(_extract_text_from_content("hello"), "hello")

    def test_list_of_text_parts_is_concatenated(self):
        content = [
            ChatCompletionMessageContentTextPart(type="text", text="Hello"),
            ChatCompletionMessageContentTextPart(type="text", text="world"),
        ]
        self.assertEqual(_extract_text_from_content(content), "Hello world")

    def test_dict_with_text_type_is_extracted(self):
        content = [{"type": "text", "text": "from dict"}]
        self.assertEqual(_extract_text_from_content(content), "from dict")

    def test_mixed_text_parts_and_dicts(self):
        content = [
            ChatCompletionMessageContentTextPart(type="text", text="A"),
            {"type": "text", "text": "B"},
            ChatCompletionMessageContentTextPart(type="text", text="C"),
        ]
        self.assertEqual(_extract_text_from_content(content), "A B C")


if __name__ == "__main__":
    unittest.main()
