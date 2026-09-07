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

register_cuda_ci(est_time=142, stage="base-b", runner_config="1-gpu-small")

TEST_MODEL_NAME = os.environ.get("TEST_MODEL_NAME", DEFAULT_SMALL_MODEL_NAME_FOR_TEST)
TEST_CLASSIFICATION_BASE_MODEL = os.environ.get(
    "TEST_CLASSIFICATION_BASE_MODEL",
    "tomaarsen/Qwen3-Reranker-0.6B-seq-cls",
)
_CLS_NUM_LABELS = AutoConfig.from_pretrained(TEST_CLASSIFICATION_BASE_MODEL).num_labels


def _collect_scores(engine_kwargs, calls):
    """Boot one engine, run ``calls`` through score(), shut it down.

    A process holds one live config, so the reference engine must be gone
    before the engine under test boots.
    """
    engine = Engine(**engine_kwargs)
    try:
        return [engine.score(**call).scores for call in calls]
    finally:
        engine.shutdown()
        torch.cuda.empty_cache()


class TestMISServerArgsValidation(unittest.TestCase):
    """Test ServerArgs defaults for MIS mode."""

    def test_enable_mis_default(self):
        """Test that enable_mis defaults to False."""
        from sglang.srt.server_args import ServerArgs

        self.assertEqual(ServerArgs.enable_mis, False)


class TestMultiItemScoringOptimization(CustomTestCase):
    """Test the Multi-Item Scoring (MIS) optimization with generation models."""

    CONSISTENCY_CALL = dict(
        query="Is this a fact?\n",
        items=[" The sun rises in the east"],
        label_token_ids=[9454, 2753],
        apply_softmax=True,
    )

    @classmethod
    def setUpClass(cls):
        (cls.non_mis_consistency_scores,) = _collect_scores(
            dict(
                model_path=TEST_MODEL_NAME,
                disable_radix_cache=True,
                chunked_prefill_size=-1,
                mem_fraction_static=0.15,
            ),
            [cls.CONSISTENCY_CALL],
        )
        cls.engine = Engine(
            model_path=TEST_MODEL_NAME,
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
        mis_scores = self.engine.score(**self.CONSISTENCY_CALL).scores
        non_mis_scores = self.non_mis_consistency_scores

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
    """MIS on a classification model: basics, MIS-vs-single-item parity, score
    distinctness / determinism / concurrency.

    Pre-trained Qwen3ForSequenceClassification, so the head weights are
    deterministic. One class rather than four because the CI harness demands an
    idle GPU at every setUpClass -- splitting these means re-booting the same
    engines instead of sharing them. score() is stateless and the radix cache
    is off, so sharing is safe.
    """

    NUM_LABELS = _CLS_NUM_LABELS

    FALLBACK_CALL = dict(query="Test:", items=["A", "B"], apply_softmax=True)
    SINGLE_VS_MIS_CALL = dict(
        query="Rate this option:",
        items=[" Option A", " Option B", " Option C"],
        apply_softmax=True,
    )

    @classmethod
    def setUpClass(cls):
        cls.non_mis_fallback_scores, cls.non_mis_single_scores = _collect_scores(
            dict(
                model_path=TEST_CLASSIFICATION_BASE_MODEL,
                disable_radix_cache=True,
                mem_fraction_static=0.15,
            ),
            [cls.FALLBACK_CALL, cls.SINGLE_VS_MIS_CALL],
        )
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
        scores = self.non_mis_fallback_scores

        self.assertEqual(len(scores), 2)
        for score_list in scores:
            self.assertEqual(len(score_list), self.NUM_LABELS)
            self.assertAlmostEqual(sum(score_list), 1.0, places=5)

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

    def test_mis_single_vs_non_mis(self):
        """MIS single-item must approximate non-MIS single-item.

        MIS inserts delimiter tokens into the attention context, which
        perturbs hidden states; after softmax the scores should still land
        within places=1 (+-0.05).
        """
        non_mis_scores = self.non_mis_single_scores
        mis_scores = self.engine.score(**self.SINGLE_VS_MIS_CALL).scores

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


class TestMultiItemScoringParity(CustomTestCase):
    """Test that MIS produces the same results as single-item scoring."""

    @classmethod
    def _cases(cls):
        """The scoring calls both engines run, keyed by the test that reads them."""
        tokenizer = AutoTokenizer.from_pretrained(TEST_MODEL_NAME)

        def label_ids(labels):
            return [tokenizer.encode(lb, add_special_tokens=False)[0] for lb in labels]

        return {
            "basic": dict(
                query="Rate this option:",
                items=[" Option A", " Option B", " Option C"],
                label_token_ids=label_ids([" good", " bad"]),
                apply_softmax=True,
            ),
            "tokenized": dict(
                query=tokenizer.encode("Rate this option:", add_special_tokens=False),
                items=[
                    tokenizer.encode(item, add_special_tokens=False)
                    for item in [" Option X", " Option Y"]
                ],
                label_token_ids=label_ids([" good", " bad"]),
                apply_softmax=True,
            ),
            "no_softmax": dict(
                query="The weather today is",
                items=[" sunny", " cloudy", " rainy"],
                label_token_ids=label_ids([" nice", " bad"]),
                apply_softmax=False,
            ),
            "many_items": dict(
                query="Rate this option from 1 to 5:",
                items=[f" Option {i}" for i in range(10)],
                label_token_ids=label_ids([" 1", " 2", " 3", " 4", " 5"]),
                apply_softmax=True,
            ),
        }

    @classmethod
    def setUpClass(cls):
        cases = cls._cases()
        names, calls = list(cases), list(cases.values())
        base = dict(
            model_path=TEST_MODEL_NAME,
            disable_radix_cache=True,
            log_level="error",
            mem_fraction_static=0.15,
        )
        cls.single_scores = dict(zip(names, _collect_scores(base, calls)))
        cls.mis_scores = dict(
            zip(
                names,
                _collect_scores(
                    dict(
                        base,
                        chunked_prefill_size=-1,
                        enable_mis=True,
                        attention_backend="flashinfer",
                    ),
                    calls,
                ),
            )
        )

    def _compare_scores(self, test_name):
        """Compare MIS vs single-item scoring results."""
        single_scores = self.single_scores[test_name]
        mis_scores = self.mis_scores[test_name]

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
        self._compare_scores("basic")

    def test_parity_tokenized_inputs(self):
        self._compare_scores("tokenized")

    def test_parity_without_softmax(self):
        self._compare_scores("no_softmax")

    def test_parity_many_items(self):
        self._compare_scores("many_items")


if __name__ == "__main__":
    unittest.main()
