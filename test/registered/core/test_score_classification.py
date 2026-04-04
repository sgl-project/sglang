"""Tests for Scoring API with SequenceClassification models.

Covers both single-item and multi-item scoring (MIS) for classification
models.  Uses json_model_override_args to override a regular Qwen3 model's
architecture to Qwen3ForSequenceClassification so we can validate the
scoring pipeline without needing a dedicated classification checkpoint
(the score head gets randomly initialised, which is fine for shape /
pipeline correctness).
"""

import json
import unittest

import torch

from sglang.srt.entrypoints.engine import Engine
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=120, suite="stage-b-test-1-gpu-small")

# A lightweight Qwen3 checkpoint whose backbone weights load cleanly into
# Qwen3ForSequenceClassification (the classification head is random).
TEST_BASE_MODEL = "Qwen/Qwen3-0.6B"
# <|endoftext|> for Qwen3 tokenizer — used as MIS delimiter
QWEN3_ENDOFTEXT_TOKEN_ID = 151643


class TestScoreClassification(CustomTestCase):
    """Single-item scoring with a SequenceClassification model."""

    NUM_LABELS = 2

    @classmethod
    def setUpClass(cls):
        override_args = json.dumps(
            {
                "architectures": ["Qwen3ForSequenceClassification"],
                "num_labels": cls.NUM_LABELS,
            }
        )
        cls.engine = Engine(
            model_path=TEST_BASE_MODEL,
            disable_radix_cache=True,
            json_model_override_args=override_args,
            mem_fraction_static=0.15,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "engine") and cls.engine is not None:
            cls.engine.shutdown()
        torch.cuda.empty_cache()

    def test_basic_single_item(self):
        """Each item gets a score vector of length num_labels."""
        scores = self.engine.score(
            query="Rate each option:",
            items=["Option A", "Option B"],
            apply_softmax=True,
        ).scores

        self.assertEqual(len(scores), 2)
        for i, score_list in enumerate(scores):
            self.assertEqual(len(score_list), self.NUM_LABELS)
            self.assertAlmostEqual(
                sum(score_list),
                1.0,
                places=5,
                msg=f"Softmax scores for item {i} should sum to 1",
            )
            for val in score_list:
                self.assertGreaterEqual(val, 0.0)
                self.assertLessEqual(val, 1.0)

    def test_single_item_edge_case(self):
        """Single item in the list."""
        scores = self.engine.score(
            query="Evaluate:",
            items=["Only item"],
            apply_softmax=True,
        ).scores

        self.assertEqual(len(scores), 1)
        self.assertEqual(len(scores[0]), self.NUM_LABELS)
        self.assertAlmostEqual(sum(scores[0]), 1.0, places=5)

    def test_raw_logits_without_softmax(self):
        """Without softmax, returns raw logits (no probability constraints)."""
        scores = self.engine.score(
            query="Evaluate:",
            items=["Alpha", "Beta"],
            apply_softmax=False,
        ).scores

        self.assertEqual(len(scores), 2)
        for score_list in scores:
            self.assertEqual(len(score_list), self.NUM_LABELS)
            for val in score_list:
                self.assertTrue(
                    isinstance(val, (int, float)),
                    f"Expected numeric score, got {type(val)}",
                )

    def test_deterministic(self):
        """Identical inputs yield near-identical scores (fp16 non-determinism allowed)."""
        kwargs = dict(query="Evaluate:", items=["alpha", "beta", "gamma"])

        scores1 = self.engine.score(**kwargs).scores
        scores2 = self.engine.score(**kwargs).scores

        self.assertEqual(len(scores1), len(scores2))
        for s1, s2 in zip(scores1, scores2):
            for v1, v2 in zip(s1, s2):
                self.assertAlmostEqual(v1, v2, places=1)

    def test_tokenized_inputs(self):
        """Pre-tokenized query and items work the same as text inputs."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(TEST_BASE_MODEL)
        query_text = "Rate this:"
        items_text = ["Good", "Bad"]

        text_scores = self.engine.score(
            query=query_text,
            items=items_text,
            apply_softmax=True,
        ).scores

        query_ids = tokenizer.encode(query_text)
        items_ids = [tokenizer.encode(item) for item in items_text]
        token_scores = self.engine.score(
            query=query_ids,
            items=items_ids,
            apply_softmax=True,
        ).scores

        self.assertEqual(len(text_scores), len(token_scores))
        for txt_s, tok_s in zip(text_scores, token_scores):
            for t, k in zip(txt_s, tok_s):
                self.assertAlmostEqual(t, k, places=4)

    def test_label_token_ids_ignored(self):
        """SequenceClassification models ignore label_token_ids (no crash)."""
        scores = self.engine.score(
            query="Evaluate:",
            items=["Test item"],
            label_token_ids=[1, 2, 3],
            apply_softmax=True,
        ).scores

        self.assertEqual(len(scores), 1)
        self.assertEqual(len(scores[0]), self.NUM_LABELS)


class TestScoreClassificationMIS(CustomTestCase):
    """Multi-item scoring (MIS) with a SequenceClassification model.

    MIS packs all items into one sequence separated by a delimiter token.
    The score_and_pool function extracts per-item scores at delimiter
    positions.
    """

    NUM_LABELS = 2

    @classmethod
    def setUpClass(cls):
        override_args = json.dumps(
            {
                "architectures": ["Qwen3ForSequenceClassification"],
                "num_labels": cls.NUM_LABELS,
            }
        )
        cls.engine = Engine(
            model_path=TEST_BASE_MODEL,
            disable_radix_cache=True,
            chunked_prefill_size=-1,
            multi_item_scoring_delimiter=QWEN3_ENDOFTEXT_TOKEN_ID,
            json_model_override_args=override_args,
            mem_fraction_static=0.15,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "engine") and cls.engine is not None:
            cls.engine.shutdown()
        torch.cuda.empty_cache()

    def test_mis_basic(self):
        """MIS produces one score vector per item."""
        items = ["Option A", "Option B", "Option C"]
        scores = self.engine.score(
            query="Rate each option:",
            items=items,
            apply_softmax=True,
        ).scores

        self.assertEqual(len(scores), len(items))
        for i, score_list in enumerate(scores):
            self.assertEqual(
                len(score_list),
                self.NUM_LABELS,
                f"Item {i} should have {self.NUM_LABELS} scores",
            )
            self.assertAlmostEqual(
                sum(score_list),
                1.0,
                places=5,
                msg=f"Scores for item {i} should sum to 1",
            )
            for val in score_list:
                self.assertGreaterEqual(val, 0.0)
                self.assertLessEqual(val, 1.0)

    def test_mis_many_items(self):
        """Stress test: 10 items."""
        items = [f"Item {i}" for i in range(10)]
        scores = self.engine.score(
            query="Classify each:",
            items=items,
            apply_softmax=True,
        ).scores

        self.assertEqual(len(scores), len(items))
        for score_list in scores:
            self.assertEqual(len(score_list), self.NUM_LABELS)
            self.assertAlmostEqual(sum(score_list), 1.0, places=5)

    def test_mis_single_item(self):
        """Edge case: single item through MIS path."""
        scores = self.engine.score(
            query="Evaluate:",
            items=["Single item"],
            apply_softmax=True,
        ).scores

        self.assertEqual(len(scores), 1)
        self.assertEqual(len(scores[0]), self.NUM_LABELS)
        self.assertAlmostEqual(sum(scores[0]), 1.0, places=5)

    def test_items_produce_distinct_scores(self):
        """Different items must produce different score vectors.

        Even with a randomly initialised classification head, different
        item texts produce different hidden states, so scores should
        differ.  This catches bugs where all delimiter tokens share the
        same pooled representation.
        """
        items = [
            "Option A is about cats",
            "Option B is about dogs",
            "Option C is about fish",
        ]
        scores = self.engine.score(query="Rate each option:", items=items).scores

        self.assertEqual(len(scores), len(items))
        all_identical = all(scores[0] == s for s in scores[1:])
        self.assertFalse(
            all_identical,
            f"All {len(items)} items returned identical scores — "
            f"MIS delimiter indexing is likely broken. Scores: {scores[0]}",
        )

    def test_deterministic(self):
        """Identical MIS requests return identical scores."""
        kwargs = dict(
            query="Evaluate:",
            items=["alpha", "beta", "gamma"],
        )
        scores1 = self.engine.score(**kwargs).scores
        scores2 = self.engine.score(**kwargs).scores

        self.assertEqual(scores1, scores2)

    def test_softmax_valid(self):
        """With softmax, each item's scores form a valid probability distribution."""
        items = ["Option A", "Option B", "Option C"]
        scores = self.engine.score(
            query="Rate each option:",
            items=items,
            apply_softmax=True,
        ).scores

        for i, score_list in enumerate(scores):
            self.assertEqual(len(score_list), self.NUM_LABELS)
            for val in score_list:
                self.assertGreaterEqual(val, 0.0)
                self.assertLessEqual(val, 1.0)
            self.assertAlmostEqual(
                sum(score_list),
                1.0,
                places=6,
                msg=f"Softmax scores for item {i} don't sum to 1: {sum(score_list)}",
            )


class TestScoreClassificationMISAdvanced(CustomTestCase):
    """Advanced MIS tests with more labels to stress tensor shape handling."""

    NUM_LABELS = 12

    @classmethod
    def setUpClass(cls):
        override_args = json.dumps(
            {
                "architectures": ["Qwen3ForSequenceClassification"],
                "num_labels": cls.NUM_LABELS,
            }
        )
        cls.engine = Engine(
            model_path=TEST_BASE_MODEL,
            disable_radix_cache=True,
            chunked_prefill_size=-1,
            multi_item_scoring_delimiter=QWEN3_ENDOFTEXT_TOKEN_ID,
            json_model_override_args=override_args,
            mem_fraction_static=0.15,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "engine") and cls.engine is not None:
            cls.engine.shutdown()
        torch.cuda.empty_cache()

    def test_many_labels_shape(self):
        """Verify correct shape with many labels (catches 2D tensor bugs)."""
        items = [f"Item {i}" for i in range(5)]
        scores = self.engine.score(
            query="Classify:",
            items=items,
            apply_softmax=True,
        ).scores

        self.assertEqual(len(scores), len(items))
        for score_list in scores:
            self.assertEqual(len(score_list), self.NUM_LABELS)
            self.assertAlmostEqual(sum(score_list), 1.0, places=5)

    def test_many_items_distinct(self):
        """15 items should not all produce identical scores."""
        items = [f"City {i}" for i in range(15)]
        scores = self.engine.score(query="Classify each city:", items=items).scores

        self.assertEqual(len(scores), len(items))
        unique_count = len({tuple(s) for s in scores})
        self.assertGreater(unique_count, 1, "All 15 items returned identical scores")


if __name__ == "__main__":
    unittest.main()
