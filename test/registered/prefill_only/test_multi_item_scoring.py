"""Tests for the Multi-Item Scoring (MIS) optimization.

MIS is a server-side optimization enabled via --enable-mis that batches
multiple items into a single forward pass using delimiter tokens (token ID 9999).
This is different from batch scoring (multiple items in one API call) which
processes items as separate requests.

The key difference:
- Batch scoring: N items -> N separate forward passes
- MIS optimization: N items -> 1 forward pass with delimiter-separated items

These tests ensure the MIS optimization produces correct results and catches
bugs in tensor shape handling (e.g., 2D tensors [num_delimiters, num_label_tokens]).
"""

import asyncio
import os
import unittest

import torch
from transformers import AutoConfig, AutoTokenizer

from sglang.srt.entrypoints.engine import Engine
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    CustomTestCase,
)

register_cuda_ci(est_time=221, suite="stage-b-test-1-gpu-small")

TEST_MODEL_NAME = os.environ.get("TEST_MODEL_NAME", DEFAULT_SMALL_MODEL_NAME_FOR_TEST)
TEST_CLASSIFICATION_BASE_MODEL = os.environ.get(
    "TEST_CLASSIFICATION_BASE_MODEL",
    "tomaarsen/Qwen3-Reranker-0.6B-seq-cls",
)
_CLS_NUM_LABELS = AutoConfig.from_pretrained(TEST_CLASSIFICATION_BASE_MODEL).num_labels


class TestMISServerArgsValidation(unittest.TestCase):
    """Test ServerArgs defaults for MIS mode."""

    def test_enable_mis_default(self):
        """Test that enable_mis defaults to False."""
        from sglang.srt.server_args import ServerArgs

        self.assertEqual(ServerArgs.enable_mis, False)


class TestMultiItemScoringOptimization(CustomTestCase):
    """Test the Multi-Item Scoring (MIS) optimization with generation models."""

    @classmethod
    def setUpClass(cls):
        cls.engine = Engine(
            model_path=TEST_MODEL_NAME,
            disable_radix_cache=True,
            chunked_prefill_size=-1,
            enable_mis=True,
            attention_backend="flashinfer",
            mem_fraction_static=0.15,
        )
        cls.non_mis_engine = Engine(
            model_path=TEST_MODEL_NAME,
            disable_radix_cache=True,
            chunked_prefill_size=-1,
            mem_fraction_static=0.15,
        )

    @classmethod
    def tearDownClass(cls):
        if cls.engine is not None:
            cls.engine.shutdown()
        if cls.non_mis_engine is not None:
            cls.non_mis_engine.shutdown()
        torch.cuda.empty_cache()

    def test_mis_basic(self):
        """Test basic MIS: correct shapes, valid probabilities."""
        query = "Rate each option:"
        items = ["Option A", "Option B", "Option C"]
        label_token_ids = [9454, 2753]  # "Yes" and "No" tokens

        scores = self.engine.score(
            query=query,
            items=items,
            label_token_ids=label_token_ids,
            apply_softmax=True,
        ).scores

        self.assertEqual(len(scores), len(items))
        for i, score_list in enumerate(scores):
            self.assertEqual(len(score_list), len(label_token_ids))
            self.assertAlmostEqual(sum(score_list), 1.0, places=5)
            for score in score_list:
                self.assertGreaterEqual(score, 0)
                self.assertLessEqual(score, 1)

    def test_mis_consistency_with_single_item(self):
        """MIS with one item should match non-MIS scoring closely."""
        query = "Is this a fact?\n"
        items = [" The sun rises in the east"]
        label_token_ids = [9454, 2753]

        mis_scores = self.engine.score(
            query=query,
            items=items,
            label_token_ids=label_token_ids,
            apply_softmax=True,
        ).scores

        non_mis_scores = self.non_mis_engine.score(
            query=query,
            items=items,
            label_token_ids=label_token_ids,
            apply_softmax=True,
        ).scores

        self.assertEqual(len(mis_scores), 1)
        self.assertEqual(len(non_mis_scores), 1)
        for j, (m, n) in enumerate(zip(mis_scores[0], non_mis_scores[0])):
            relative_diff = abs(m - n) / max(abs(n), 1e-6)
            self.assertLess(
                relative_diff,
                0.08,
                msg=f"label {j}: MIS={m} vs non-MIS={n} (diff: {relative_diff:.3f})",
            )

    def test_mis_empty_query(self):
        """MIS with empty query — delimiter indices start at position 0."""
        items = ["alpha", "beta"]
        label_token_ids = [9454, 2753]

        scores = self.engine.score(
            query="",
            items=items,
            label_token_ids=label_token_ids,
            apply_softmax=True,
        ).scores

        self.assertEqual(len(scores), len(items))
        for score_list in scores:
            self.assertEqual(len(score_list), len(label_token_ids))
            self.assertAlmostEqual(sum(score_list), 1.0, places=5)


class TestMultiItemScoringClassification(CustomTestCase):
    """Test MIS with classification models.

    Uses a pre-trained Qwen3ForSequenceClassification model so that the
    classification head weights are deterministic across Engine instances.
    """

    NUM_LABELS = _CLS_NUM_LABELS

    def setUp(self):
        self.engine = Engine(
            model_path=TEST_CLASSIFICATION_BASE_MODEL,
            disable_radix_cache=True,
            chunked_prefill_size=-1,
            enable_mis=True,
            attention_backend="flashinfer",
            mem_fraction_static=0.15,
        )

    def tearDown(self):
        if self.engine is not None:
            self.engine.shutdown()
            torch.cuda.empty_cache()

    def test_classification_mis_basic(self):
        """Classification MIS: correct shapes, valid softmax probabilities."""
        query = "Rate each option:"
        items = ["Option A", "Option B", "Option C"]

        scores = self.engine.score(query=query, items=items, apply_softmax=True).scores

        self.assertEqual(len(scores), len(items))
        for i, score_list in enumerate(scores):
            self.assertEqual(len(score_list), self.NUM_LABELS)
            self.assertAlmostEqual(sum(score_list), 1.0, places=5)
            for score in score_list:
                self.assertGreaterEqual(score, 0)
                self.assertLessEqual(score, 1)

    def test_classification_mis_tokenized_input(self):
        """Classification MIS with pre-tokenized query and items."""
        tokenizer = AutoTokenizer.from_pretrained(TEST_CLASSIFICATION_BASE_MODEL)
        query_ids = tokenizer.encode("Rate each option:", add_special_tokens=False)
        items_ids = [
            tokenizer.encode(item, add_special_tokens=False)
            for item in ["Option A", "Option B"]
        ]

        scores = self.engine.score(
            query=query_ids, items=items_ids, apply_softmax=True
        ).scores

        self.assertEqual(len(scores), len(items_ids))
        for score_list in scores:
            self.assertEqual(len(score_list), self.NUM_LABELS)
            self.assertAlmostEqual(sum(score_list), 1.0, places=5)

    def test_classification_non_mis_fallback(self):
        """Classification model works correctly without --enable-mis."""
        non_mis_engine = Engine(
            model_path=TEST_CLASSIFICATION_BASE_MODEL,
            disable_radix_cache=True,
            mem_fraction_static=0.15,
        )
        try:
            scores = non_mis_engine.score(
                query="Test:", items=["A", "B"], apply_softmax=True
            ).scores

            self.assertEqual(len(scores), 2)
            for score_list in scores:
                self.assertEqual(len(score_list), self.NUM_LABELS)
                self.assertAlmostEqual(sum(score_list), 1.0, places=5)
        finally:
            non_mis_engine.shutdown()
            torch.cuda.empty_cache()


class TestMultiItemScoringParity(CustomTestCase):
    """Test that MIS produces the same results as single-item scoring."""

    @classmethod
    def setUpClass(cls):
        cls.engine_single = Engine(
            model_path=TEST_MODEL_NAME,
            disable_radix_cache=True,
            log_level="error",
            mem_fraction_static=0.15,
        )
        cls.engine_mis = Engine(
            model_path=TEST_MODEL_NAME,
            disable_radix_cache=True,
            chunked_prefill_size=-1,
            log_level="error",
            enable_mis=True,
            attention_backend="flashinfer",
            mem_fraction_static=0.15,
        )

    @classmethod
    def tearDownClass(cls):
        if cls.engine_single is not None:
            cls.engine_single.shutdown()
        if cls.engine_mis is not None:
            cls.engine_mis.shutdown()
        torch.cuda.empty_cache()

    def _compare_scores(
        self, query, items, label_token_ids=None, apply_softmax=True, test_name=""
    ):
        """Compare MIS vs single-item scoring results."""
        single_scores = self.engine_single.score(
            query=query,
            items=items,
            label_token_ids=label_token_ids,
            apply_softmax=apply_softmax,
        ).scores

        mis_scores = self.engine_mis.score(
            query=query,
            items=items,
            label_token_ids=label_token_ids,
            apply_softmax=apply_softmax,
        ).scores

        self.assertEqual(
            len(mis_scores), len(single_scores), f"{test_name}: count mismatch"
        )
        for i, (ms, ss) in enumerate(zip(mis_scores, single_scores)):
            self.assertEqual(len(ms), len(ss), f"{test_name}: item {i} length mismatch")
            for j, (m, s) in enumerate(zip(ms, ss)):
                self.assertAlmostEqual(
                    m,
                    s,
                    places=1,
                    msg=f"{test_name}: item {i} label {j}: MIS={m} vs single={s}",
                )

    def test_parity_basic(self):
        tokenizer = AutoTokenizer.from_pretrained(TEST_MODEL_NAME)
        query = "Rate this option:"
        items = [" Option A", " Option B", " Option C"]
        labels = [" good", " bad"]
        label_ids = [tokenizer.encode(lb, add_special_tokens=False)[0] for lb in labels]
        self._compare_scores(query, items, label_ids, test_name="basic")

    def test_parity_tokenized_inputs(self):
        tokenizer = AutoTokenizer.from_pretrained(TEST_MODEL_NAME)
        query = "Rate this option:"
        items = [" Option X", " Option Y"]
        labels = [" good", " bad"]
        query_ids = tokenizer.encode(query, add_special_tokens=False)
        items_ids = [tokenizer.encode(i, add_special_tokens=False) for i in items]
        label_ids = [tokenizer.encode(lb, add_special_tokens=False)[0] for lb in labels]
        self._compare_scores(query_ids, items_ids, label_ids, test_name="tokenized")

    def test_parity_without_softmax(self):
        tokenizer = AutoTokenizer.from_pretrained(TEST_MODEL_NAME)
        query = "The weather today is"
        items = [" sunny", " cloudy", " rainy"]
        labels = [" nice", " bad"]
        label_ids = [tokenizer.encode(lb, add_special_tokens=False)[0] for lb in labels]
        self._compare_scores(
            query, items, label_ids, apply_softmax=False, test_name="no_softmax"
        )

    def test_parity_many_items(self):
        tokenizer = AutoTokenizer.from_pretrained(TEST_MODEL_NAME)
        query = "Rate this option from 1 to 5:"
        items = [f" Option {i}" for i in range(10)]
        labels = [" 1", " 2", " 3", " 4", " 5"]
        label_ids = [tokenizer.encode(lb, add_special_tokens=False)[0] for lb in labels]
        self._compare_scores(query, items, label_ids, test_name="many_items")


class TestMultiItemScoringClassificationParity(CustomTestCase):
    """Test that MIS multi-item batching matches single-item MIS scoring.

    Both paths use the MIS engine (with delimiter tokens in the attention
    context).  The reference scores each item individually so each gets its
    own forward pass; the batched path packs all items into one pass.
    This isolates the MIS batching logic from the delimiter-presence effect.
    """

    NUM_LABELS = _CLS_NUM_LABELS

    @classmethod
    def setUpClass(cls):
        cls.engine = Engine(
            model_path=TEST_CLASSIFICATION_BASE_MODEL,
            disable_radix_cache=True,
            chunked_prefill_size=-1,
            enable_mis=True,
            attention_backend="flashinfer",
            mem_fraction_static=0.15,
        )

    @classmethod
    def tearDownClass(cls):
        if cls.engine is not None:
            cls.engine.shutdown()
        torch.cuda.empty_cache()

    def _compare_scores(self, query, items, apply_softmax=True, test_name=""):
        """Compare MIS batched vs MIS single-item scoring results."""
        single_scores = []
        for item in items:
            result = self.engine.score(
                query=query,
                items=[item],
                apply_softmax=apply_softmax,
            ).scores
            single_scores.append(result[0])

        batched_scores = self.engine.score(
            query=query,
            items=items,
            apply_softmax=apply_softmax,
        ).scores

        self.assertEqual(
            len(batched_scores), len(single_scores), f"{test_name}: count mismatch"
        )
        for i, (bs, ss) in enumerate(zip(batched_scores, single_scores)):
            self.assertEqual(len(bs), len(ss), f"{test_name}: item {i} length mismatch")
            for j, (b, s) in enumerate(zip(bs, ss)):
                self.assertAlmostEqual(
                    b,
                    s,
                    places=1,
                    msg=f"{test_name}: item {i} label {j}: batched={b} vs single={s}",
                )

    def test_parity_basic(self):
        query = "Rate this option:"
        items = [" Option A", " Option B", " Option C"]
        self._compare_scores(query, items, test_name="cls_basic")

    def test_parity_tokenized_inputs(self):
        tokenizer = AutoTokenizer.from_pretrained(TEST_CLASSIFICATION_BASE_MODEL)
        query_ids = tokenizer.encode("Rate this option:", add_special_tokens=False)
        items_ids = [
            tokenizer.encode(item, add_special_tokens=False)
            for item in [" Option X", " Option Y"]
        ]
        self._compare_scores(query_ids, items_ids, test_name="cls_tokenized")

    def test_parity_without_softmax(self):
        query = "The weather today is"
        items = [" sunny", " cloudy", " rainy"]
        self._compare_scores(
            query, items, apply_softmax=False, test_name="cls_no_softmax"
        )

    def test_parity_many_items(self):
        query = "Classify this option:"
        items = [f" Option {i}" for i in range(10)]
        self._compare_scores(query, items, test_name="cls_many_items")


class TestMultiItemScoringClassificationMISvsNonMIS(CustomTestCase):
    """Test that MIS single-item approximates non-MIS single-item.

    The MIS path inserts delimiter tokens into the attention context,
    which slightly perturbs hidden states.  After softmax the scores
    should still be close.  Uses places=1 (±0.05) tolerance.

    Runs as a separate class so each engine is created and destroyed
    independently to avoid GPU OOM.
    """

    def test_mis_single_vs_non_mis(self):
        non_mis_engine = Engine(
            model_path=TEST_CLASSIFICATION_BASE_MODEL,
            disable_radix_cache=True,
            mem_fraction_static=0.15,
        )
        try:
            query = "Rate this option:"
            items = [" Option A", " Option B", " Option C"]
            non_mis_scores = non_mis_engine.score(
                query=query,
                items=items,
                apply_softmax=True,
            ).scores
        finally:
            non_mis_engine.shutdown()
            torch.cuda.empty_cache()

        mis_engine = Engine(
            model_path=TEST_CLASSIFICATION_BASE_MODEL,
            disable_radix_cache=True,
            chunked_prefill_size=-1,
            enable_mis=True,
            attention_backend="flashinfer",
            mem_fraction_static=0.15,
        )
        try:
            mis_scores = mis_engine.score(
                query=query,
                items=items,
                apply_softmax=True,
            ).scores
        finally:
            mis_engine.shutdown()
            torch.cuda.empty_cache()

        self.assertEqual(len(mis_scores), len(non_mis_scores))
        for i, (ms, ns) in enumerate(zip(mis_scores, non_mis_scores)):
            self.assertEqual(len(ms), len(ns))
            for j, (m, n) in enumerate(zip(ms, ns)):
                self.assertAlmostEqual(
                    m,
                    n,
                    places=1,
                    msg=f"item {i} label {j}: MIS={m} vs non-MIS={n}",
                )


class TestMultiItemScoringClassificationAdvanced(CustomTestCase):
    """Advanced MIS tests for classification models: score distinctness,
    determinism, and concurrent request handling."""

    NUM_LABELS = _CLS_NUM_LABELS

    @classmethod
    def setUpClass(cls):
        cls.engine = Engine(
            model_path=TEST_CLASSIFICATION_BASE_MODEL,
            disable_radix_cache=True,
            chunked_prefill_size=-1,
            enable_mis=True,
            attention_backend="flashinfer",
            mem_fraction_static=0.15,
        )

    @classmethod
    def tearDownClass(cls):
        if cls.engine is not None:
            cls.engine.shutdown()
        torch.cuda.empty_cache()

    def test_items_produce_distinct_scores(self):
        """Different items must produce different score vectors.

        Core regression test: before the delimiter-index fix, all items got
        identical scores because the MIS attention mask only let delimiter
        tokens attend to the query prefix.
        """
        query = "Rate each option:"
        items = [
            "Option A is about cats",
            "Option B is about dogs",
            "Option C is about fish",
        ]

        scores = self.engine.score(query=query, items=items).scores

        self.assertEqual(len(scores), len(items))
        all_identical = all(scores[0] == s for s in scores[1:])
        self.assertFalse(
            all_identical,
            f"All {len(items)} items returned identical scores — "
            f"MIS delimiter indexing is broken. Scores: {scores[0]}",
        )

    def test_many_items_distinct(self):
        """Stress test: 15 items should not all produce identical scores."""
        query = "Classify each city:"
        items = [f"City {i}" for i in range(15)]

        scores = self.engine.score(query=query, items=items).scores

        self.assertEqual(len(scores), len(items))
        for score_list in scores:
            self.assertEqual(len(score_list), self.NUM_LABELS)

        unique_count = len({tuple(s) for s in scores})
        self.assertGreater(unique_count, 1, "All 15 items returned identical scores")

    def test_deterministic(self):
        """Identical requests should return identical scores."""
        query = "Evaluate:"
        items = ["alpha", "beta", "gamma"]

        scores1 = self.engine.score(query=query, items=items).scores
        scores2 = self.engine.score(query=query, items=items).scores

        self.assertEqual(
            scores1, scores2, "Identical inputs must produce identical scores"
        )

    def test_concurrent_requests(self):
        """Concurrent MIS requests must produce the same scores as sequential.

        Runs each request sequentially to get baseline scores, then runs all
        concurrently and asserts the results match. This catches cross-request
        contamination when multiple MIS requests share a GPU batch.
        """
        test_cases = [
            {"query": "Is this a fruit?", "items": ["apple", "car", "banana"]},
            {"query": "Is this an animal?", "items": ["dog", "table"]},
            {
                "query": "Is this a country?",
                "items": ["France", "pizza", "Japan", "chair"],
            },
            {"query": "Is this a color?", "items": ["red"]},
        ]

        # Sequential baseline
        sequential_scores = []
        for tc in test_cases:
            result = self.engine.score(query=tc["query"], items=tc["items"])
            sequential_scores.append(result.scores)

        # Concurrent execution
        async def _gather():
            return await asyncio.gather(
                *(
                    self.engine.async_score(query=tc["query"], items=tc["items"])
                    for tc in test_cases
                )
            )

        concurrent_results = self.engine.loop.run_until_complete(_gather())

        for idx, (tc, seq_scores, conc_result) in enumerate(
            zip(test_cases, sequential_scores, concurrent_results)
        ):
            conc_scores = conc_result.scores
            self.assertEqual(
                len(conc_scores),
                len(seq_scores),
                f"Case {idx}: count mismatch",
            )
            for i, (cs, ss) in enumerate(zip(conc_scores, seq_scores)):
                self.assertEqual(
                    len(cs),
                    len(ss),
                    f"Case {idx} item {i}: label count mismatch",
                )
                for j, (c, s) in enumerate(zip(cs, ss)):
                    self.assertAlmostEqual(
                        c,
                        s,
                        places=1,
                        msg=f"Case {idx} item {i} label {j}: "
                        f"concurrent={c} vs sequential={s}",
                    )


if __name__ == "__main__":
    unittest.main()
