"""
Integration tests for Scoring API across multiple model architectures.

This test suite extends test coverage to validate the Scoring API works correctly
across different model architectures including:
- CausalLM models (Llama, Qwen, etc.)
- Different model families and variants

To run this test:
    python test/registered/core/test_score_api_multi_arch.py
"""

import unittest
from typing import List, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from sglang.srt.entrypoints.engine import Engine
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST_BASE,
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN,
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST_SCORE,
    CustomTestCase,
)

register_cuda_ci(est_time=600, suite="stage-b-test-large-1-gpu")
# Test models for different architectures
# Using smaller models for faster CI execution
TEST_MODELS = {
    # CausalLM models
    "causallm_llama": DEFAULT_SMALL_MODEL_NAME_FOR_TEST,  # Llama-3.2-1B-Instruct
    "causallm_llama_base": DEFAULT_SMALL_MODEL_NAME_FOR_TEST_BASE,  # Llama-3.2-1B
    "causallm_qwen": DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN,  # Qwen2.5-1.5B-Instruct
    # SequenceClassification models
    "sequence_classification_qwen3": DEFAULT_SMALL_MODEL_NAME_FOR_TEST_SCORE,  # Qwen3-Reranker-0.6B (Qwen3ForSequenceClassification)
    # Note: The following models don't have predefined constants in test_utils yet
}


class TestScoreAPIMultiArch(CustomTestCase):
    """Test Scoring API across multiple model architectures."""

    def setUp(self):
        """Set up test case - will be overridden by parameterized tests."""
        self.engine = None
        self.hf_model = None
        self.tokenizer = None

    def tearDown(self):
        """Clean up after each test case."""
        if self.engine is not None:
            self.engine.shutdown()
            torch.cuda.empty_cache()
        if self.hf_model is not None:
            self.hf_model.cpu()
            del self.hf_model
            torch.cuda.empty_cache()
        if self.tokenizer is not None:
            del self.tokenizer

    def _setup_model(self, model_name: str, model_type: str = "causallm"):
        """Set up engine and reference model for a given model architecture."""
        self.model_name = model_name
        self.model_type = model_type

        # Initialize SGLang engine
        self.engine = Engine(model_path=model_name)

        # Initialize reference model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )

        if model_type == "causallm":
            self.hf_model = AutoModelForCausalLM.from_pretrained(
                model_name, trust_remote_code=True
            )
        elif model_type == "sequence_classification":
            self.hf_model = AutoModelForSequenceClassification.from_pretrained(
                model_name, trust_remote_code=True
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def _get_token_ids(self, tokens: List[str]) -> List[int]:
        """Helper method to get token IDs for a list of tokens."""
        label_token_ids = []
        for token in tokens:
            encoding = self.tokenizer.encode_plus(token, add_special_tokens=False)
            token_ids = encoding["input_ids"]
            if token_ids:
                label_token_ids.append(token_ids[0])
        return label_token_ids

    def _compute_hf_scores_causallm(
        self,
        query: str,
        items: List[str],
        label_token_ids: List[int],
        apply_softmax: bool = False,
        item_first: bool = False,
    ) -> List[List[float]]:
        """Compute scores using HuggingFace CausalLM model."""
        scores = []
        for item in items:
            full_text = f"{item}{query}" if item_first else f"{query}{item}"
            inputs = self.tokenizer(full_text, return_tensors="pt").to(
                self.hf_model.device
            )

            with torch.no_grad():
                outputs = self.hf_model(**inputs)
                last_token_logits = outputs.logits[0, -1]
                target_logits = last_token_logits[label_token_ids]

                if apply_softmax:
                    target_probs = torch.softmax(target_logits, dim=-1)
                    probs = [
                        target_probs[i].item() for i in range(len(label_token_ids))
                    ]
                else:
                    probs = [
                        target_logits[i].item() for i in range(len(label_token_ids))
                    ]

            scores.append(probs)
        return scores

    def _compute_hf_scores_sequence_classification(
        self,
        query: str,
        items: List[str],
        label_token_ids: Optional[List[int]] = None,
        apply_softmax: bool = False,
        item_first: bool = False,
    ) -> List[List[float]]:
        """Compute scores using HuggingFace SequenceClassification model.

        Note: SequenceClassification models output class logits, not token logits.
        For these models, we compare the classification scores directly.
        """
        scores = []
        for item in items:
            full_text = f"{item}{query}" if item_first else f"{query}{item}"
            inputs = self.tokenizer(
                full_text, return_tensors="pt", truncation=True, max_length=512
            ).to(self.hf_model.device)

            with torch.no_grad():
                outputs = self.hf_model(**inputs)
                logits = outputs.logits[0]

                if apply_softmax:
                    probs = torch.softmax(logits, dim=-1).tolist()
                else:
                    probs = logits.tolist()

            scores.append(probs)
        return scores

    def _compare_scores(
        self,
        hf_scores: List[List[float]],
        sglang_scores: List[List[float]],
        label_token_ids: Optional[List[int]] = None,
        case_name: str = "",
        tolerance: float = 0.01,
    ):
        """Helper method to compare scores between HF and SGLang."""
        self.assertEqual(
            len(hf_scores),
            len(sglang_scores),
            f"Score lengths don't match for {case_name}",
        )

        for hf_score_list, sglang_score_list in zip(hf_scores, sglang_scores):
            self.assertEqual(
                len(hf_score_list),
                len(sglang_score_list),
                f"Score list lengths don't match for {case_name}",
            )

            for hf_score, sglang_score in zip(hf_score_list, sglang_score_list):
                diff = abs(hf_score - sglang_score)
                self.assertLessEqual(
                    diff,
                    tolerance,
                    msg=f"Scores differ by {diff:.6f} ({case_name}): "
                    f"HF={hf_score:.6f}, SGLang={sglang_score:.6f}",
                )

    # ==================== CausalLM Architecture Tests ====================

    def test_causallm_llama_basic_scoring(self):
        """Test basic scoring with Llama CausalLM model."""
        self._setup_model(TEST_MODELS["causallm_llama"], "causallm")

        query = "The capital of France is"
        items = ["Paris", "London", "Berlin"]
        label_token_ids = self._get_token_ids(["Paris", "London"])

        # Get scores from SGLang
        sglang_scores = self.engine.score(
            query=query,
            items=items,
            label_token_ids=label_token_ids,
            apply_softmax=True,
        )

        # Get scores from HuggingFace
        hf_scores = self._compute_hf_scores_causallm(
            query=query,
            items=items,
            label_token_ids=label_token_ids,
            apply_softmax=True,
        )

        # Compare scores
        self._compare_scores(hf_scores, sglang_scores, label_token_ids, "Llama basic")

    def test_causallm_llama_base_basic_scoring(self):
        """Test basic scoring with Llama base (non-instruct) CausalLM model."""
        self._setup_model(TEST_MODELS["causallm_llama_base"], "causallm")

        query = "The capital of France is"
        items = ["Paris", "London"]
        label_token_ids = self._get_token_ids(["Paris", "London"])

        sglang_scores = self.engine.score(
            query=query,
            items=items,
            label_token_ids=label_token_ids,
            apply_softmax=True,
        )

        hf_scores = self._compute_hf_scores_causallm(
            query=query,
            items=items,
            label_token_ids=label_token_ids,
            apply_softmax=True,
        )

        self._compare_scores(hf_scores, sglang_scores, label_token_ids, "Llama base")

    def test_causallm_item_first(self):
        """Test CausalLM scoring with item_first=True."""
        self._setup_model(TEST_MODELS["causallm_llama"], "causallm")

        query = " is a city"
        items = ["Tokyo", "Japan"]
        label_token_ids = self._get_token_ids(["Tokyo", "Japan"])

        sglang_scores = self.engine.score(
            query=query,
            items=items,
            label_token_ids=label_token_ids,
            apply_softmax=True,
            item_first=True,
        )

        hf_scores = self._compute_hf_scores_causallm(
            query=query,
            items=items,
            label_token_ids=label_token_ids,
            apply_softmax=True,
            item_first=True,
        )

        self._compare_scores(
            hf_scores, sglang_scores, label_token_ids, "Llama item_first"
        )

    def test_causallm_without_softmax(self):
        """Test CausalLM scoring without softmax normalization.

        Note: The scoring API may return logprobs instead of raw logits when apply_softmax=False.
        This test verifies the API works but may not match HF logits exactly.
        """
        self._setup_model(TEST_MODELS["causallm_llama"], "causallm")

        query = "The answer is"
        items = ["Yes", "No"]
        label_token_ids = self._get_token_ids(["Yes", "No"])

        sglang_scores = self.engine.score(
            query=query,
            items=items,
            label_token_ids=label_token_ids,
            apply_softmax=False,
        )

        # Verify structure and that scores are returned
        self.assertEqual(len(sglang_scores), len(items))
        for score_list in sglang_scores:
            self.assertEqual(len(score_list), len(label_token_ids))
            # Verify scores are numeric (may be logits or logprobs)
            for score in score_list:
                self.assertIsInstance(score, (int, float))

        # Note: We don't compare with HF directly as the API may return logprobs
        # instead of raw logits. The important thing is that it works without softmax.

    def test_causallm_large_batch(self):
        """Test CausalLM scoring with large batch size."""
        self._setup_model(TEST_MODELS["causallm_llama"], "causallm")

        query = "Classify:"
        items = [f"Item {i}" for i in range(20)]
        label_token_ids = self._get_token_ids(["A", "B", "C"])

        sglang_scores = self.engine.score(
            query=query,
            items=items,
            label_token_ids=label_token_ids,
            apply_softmax=True,
        )

        self.assertEqual(len(sglang_scores), len(items))
        for score_list in sglang_scores:
            self.assertEqual(len(score_list), len(label_token_ids))
            self.assertAlmostEqual(sum(score_list), 1.0, places=4)

    # ==================== Cross-Architecture Consistency Tests ====================

    def test_consistency_across_architectures(self):
        """Test that scoring API produces consistent results across architectures."""
        query = "The answer is"
        items = ["Yes", "No"]

        # Test with Llama Instruct
        self._setup_model(TEST_MODELS["causallm_llama"], "causallm")
        label_token_ids = self._get_token_ids(["Yes", "No"])
        llama_scores = self.engine.score(
            query=query,
            items=items,
            label_token_ids=label_token_ids,
            apply_softmax=True,
        )

        # Test with Llama Base
        self.tearDown()
        self._setup_model(TEST_MODELS["causallm_llama_base"], "causallm")
        llama_base_scores = self.engine.score(
            query=query,
            items=items,
            label_token_ids=label_token_ids,
            apply_softmax=True,
        )

        # Both should produce valid scores (may differ due to model differences)
        self.assertEqual(len(llama_scores), len(llama_base_scores))
        self.assertEqual(len(llama_scores[0]), len(llama_base_scores[0]))

        # Verify both sum to ~1 (probabilities)
        for score_list in llama_scores + llama_base_scores:
            self.assertAlmostEqual(sum(score_list), 1.0, places=4)

    # ==================== Error Handling Tests ====================

    def test_invalid_label_token_ids(self):
        """Test error handling for invalid label token IDs."""
        self._setup_model(TEST_MODELS["causallm_llama"], "causallm")

        query = "Test"
        items = ["Item"]
        vocab_size = self.tokenizer.vocab_size
        invalid_token_ids = [vocab_size + 1, vocab_size + 2]  # Out of vocabulary

        with self.assertRaises((ValueError, IndexError)):
            self.engine.score(
                query=query,
                items=items,
                label_token_ids=invalid_token_ids,
                apply_softmax=True,
            )

    def test_empty_query(self):
        """Test handling of empty query."""
        self._setup_model(TEST_MODELS["causallm_llama"], "causallm")

        query = ""
        items = ["Item"]
        label_token_ids = self._get_token_ids(["Item"])

        scores = self.engine.score(
            query=query,
            items=items,
            label_token_ids=label_token_ids,
            apply_softmax=True,
        )

        self.assertEqual(len(scores), 1)
        self.assertEqual(len(scores[0]), len(label_token_ids))

    # ==================== Edge Cases ====================
    def test_single_item(self):
        """Test scoring with single item."""
        self._setup_model(TEST_MODELS["causallm_llama"], "causallm")

        query = "The capital is"
        items = ["Paris"]
        label_token_ids = self._get_token_ids(["Paris", "London"])

        scores = self.engine.score(
            query=query,
            items=items,
            label_token_ids=label_token_ids,
            apply_softmax=True,
        )

        self.assertEqual(len(scores), 1)
        self.assertEqual(len(scores[0]), len(label_token_ids))
        self.assertAlmostEqual(sum(scores[0]), 1.0, places=4)

    def test_single_label_token(self):
        """Test scoring with single label token."""
        self._setup_model(TEST_MODELS["causallm_llama"], "causallm")

        query = "The answer is"
        items = ["Yes", "No"]
        label_token_ids = self._get_token_ids(["Yes"])

        scores = self.engine.score(
            query=query,
            items=items,
            label_token_ids=label_token_ids,
            apply_softmax=True,
        )

        self.assertEqual(len(scores), len(items))
        for score_list in scores:
            self.assertEqual(len(score_list), 1)
            # With single token and softmax, score should be 1.0
            self.assertAlmostEqual(score_list[0], 1.0, places=4)

    def test_many_label_tokens(self):
        """Test scoring with many label tokens."""
        self._setup_model(TEST_MODELS["causallm_llama"], "causallm")

        query = "Choose:"
        items = ["A", "B"]
        # Get token IDs for multiple tokens
        label_tokens = ["A", "B", "C", "D", "E", "F", "G", "H"]
        label_token_ids = self._get_token_ids(label_tokens)

        scores = self.engine.score(
            query=query,
            items=items,
            label_token_ids=label_token_ids,
            apply_softmax=True,
        )

        self.assertEqual(len(scores), len(items))
        for score_list in scores:
            self.assertEqual(len(score_list), len(label_token_ids))
            self.assertAlmostEqual(sum(score_list), 1.0, places=4)


# ==================== Parameterized Test Suite ====================


def create_parameterized_tests():
    """Create parameterized test methods for each model architecture."""
    test_methods = {}

    for model_key, model_name in TEST_MODELS.items():
        model_type = (
            "causallm" if "causallm" in model_key else "sequence_classification"
        )

        def make_test_method(m_name, m_type, m_key):
            def test_method(self):
                """Parameterized test for specific model architecture."""
                self._setup_model(m_name, m_type)

                query = "The capital of France is"
                items = ["Paris", "London"]
                label_token_ids = self._get_token_ids(["Paris", "London"])

                scores = self.engine.score(
                    query=query,
                    items=items,
                    label_token_ids=label_token_ids,
                    apply_softmax=True,
                )

                self.assertEqual(len(scores), len(items))
                for score_list in scores:
                    self.assertEqual(len(score_list), len(label_token_ids))
                    self.assertAlmostEqual(sum(score_list), 1.0, places=4)

            test_method.__name__ = f"test_architecture_{m_key}"
            test_method.__doc__ = f"Test scoring API with {m_key} architecture"
            return test_method

        test_methods[f"test_architecture_{model_key}"] = make_test_method(
            model_name, model_type, model_key
        )

    return test_methods


# Dynamically add parameterized tests
for test_name, test_method in create_parameterized_tests().items():
    setattr(TestScoreAPIMultiArch, test_name, test_method)


if __name__ == "__main__":
    unittest.main()
