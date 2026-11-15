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
import re
import unittest

import requests
import torch
from transformers import AutoConfig, AutoTokenizer

from sglang.srt.entrypoints.engine import Engine
from sglang.srt.server_args import _declared_default
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    CustomTestCase,
    find_available_port,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_cuda_ci(est_time=137, stage="extra-a", runner_config="1-gpu-small")

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

        self.assertEqual(_declared_default("enable_mis"), False)


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

    def test_prefill_cuda_graph_breakable_enabled(self):
        """MIS keeps breakable prefill CUDA graph active while disabling decode."""
        info = self.engine.get_server_info()
        self.assertEqual(info["cuda_graph_config"]["prefill"]["backend"], "breakable")
        self.assertEqual(info["cuda_graph_config"]["decode"]["backend"], "disabled")


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

    def test_classification_prefill_cuda_graph_breakable_enabled(self):
        """Classification MIS keeps breakable prefill CUDA graph active."""
        info = self.engine.get_server_info()
        self.assertEqual(info["cuda_graph_config"]["prefill"]["backend"], "breakable")
        self.assertEqual(info["cuda_graph_config"]["decode"]["backend"], "disabled")

    def test_classification_mis_empty_query(self):
        """Classification MIS with empty query handles clamped delimiter-1 safely."""
        for return_phs in (False, True):
            with self.subTest(return_pooled_hidden_states=return_phs):
                result = self.engine.score(
                    query="",
                    items=["alpha", "beta", "gamma"],
                    apply_softmax=True,
                    return_pooled_hidden_states=return_phs,
                )
                self.assertEqual(len(result.scores), 3)
                for row in result.scores:
                    self.assertEqual(len(row), self.NUM_LABELS)
                    self.assertAlmostEqual(sum(row), 1.0, places=5)
                if return_phs:
                    self.assertIsNotNone(result.pooled_hidden_states)
                    self.assertEqual(len(result.pooled_hidden_states), 3)
                    for phs in result.pooled_hidden_states:
                        self.assertTrue(torch.isfinite(phs).all())
                else:
                    self.assertIsNone(result.pooled_hidden_states)


class TestMultiItemScoringParity(CustomTestCase):
    """Test that MIS produces the same results as single-item scoring."""

    @classmethod
    def _cases(cls):
        """The scoring calls both engines run, keyed by the test that reads them."""
        tokenizer = AutoTokenizer.from_pretrained(TEST_MODEL_NAME)

        def label_ids(labels):
            token_ids = []
            for label in labels:
                (token_id,) = tokenizer.encode(label, add_special_tokens=False)
                token_ids.append(token_id)
            assert len(set(token_ids)) == len(token_ids), token_ids
            return token_ids

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
                label_token_ids=label_ids(["1", "2", "3", "4", "5"]),
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


class TestMultiItemScoringPrefillCudaGraphParity(CustomTestCase):
    """Compare MIS under breakable prefill CUDA graph against eager MIS."""

    @classmethod
    def _causallm_calls(cls):
        tokenizer = AutoTokenizer.from_pretrained(TEST_MODEL_NAME)
        yes_no = [
            tokenizer.encode(lbl, add_special_tokens=False)[0]
            for lbl in (" good", " bad")
        ]
        return {
            "bucket_small": dict(
                query="Rate relevance:",
                items=[f" candidate item {i}" for i in range(4)],
                label_token_ids=yes_no,
                apply_softmax=True,
            ),
            "bucket_medium": dict(
                query="Score each document for relevance to search:",
                items=[f" document summary number {i}" for i in range(12)],
                label_token_ids=yes_no,
                apply_softmax=True,
            ),
            "empty_query": dict(
                query="",
                items=[" alpha option", " beta option", " gamma option"],
                label_token_ids=yes_no,
                apply_softmax=True,
            ),
            "raw_logprobs": dict(
                query="Is this helpful?",
                items=[" very helpful overview", " irrelevant text"],
                label_token_ids=yes_no,
                apply_softmax=False,
                return_token_logprobs=True,
            ),
        }

    def _assert_scores_close(
        self,
        *,
        actual_scores,
        expected_scores,
        label: str,
        check_top1: bool = True,
    ) -> None:
        self.assertEqual(len(actual_scores), len(expected_scores), label)
        for i, (act_row, exp_row) in enumerate(zip(actual_scores, expected_scores)):
            self.assertEqual(len(act_row), len(exp_row), f"{label}[{i}]")
            for j, (act_val, exp_val) in enumerate(zip(act_row, exp_row)):
                self.assertAlmostEqual(
                    act_val,
                    exp_val,
                    delta=1e-2,
                    msg=f"{label}[{i}][{j}]: actual={act_val} vs expected={exp_val}",
                )
            if check_top1 and len(act_row) > 1:
                act_top = max(range(len(act_row)), key=act_row.__getitem__)
                exp_top = max(range(len(exp_row)), key=exp_row.__getitem__)
                self.assertEqual(act_top, exp_top, f"{label}[{i}] top-1 mismatch")

    def _assert_token_logprobs_close(
        self, *, actual_tlp, expected_tlp, label: str
    ) -> None:
        self.assertIsNotNone(actual_tlp, f"{label}: missing actual token_logprobs")
        self.assertIsNotNone(expected_tlp, f"{label}: missing expected token_logprobs")
        self.assertEqual(len(actual_tlp), len(expected_tlp), label)
        for i, (act_row, exp_row) in enumerate(zip(actual_tlp, expected_tlp)):
            self.assertEqual(len(act_row), len(exp_row), f"{label} logprob[{i}]")
            for j, (act_lp, exp_lp) in enumerate(zip(act_row, exp_row)):
                self.assertAlmostEqual(
                    act_lp,
                    exp_lp,
                    delta=5e-2,
                    msg=f"{label} logprob[{i}][{j}]: actual={act_lp} vs expected={exp_lp}",
                )
            act_top = max(range(len(act_row)), key=act_row.__getitem__)
            exp_top = max(range(len(exp_row)), key=exp_row.__getitem__)
            self.assertEqual(act_top, exp_top, f"{label} logprob[{i}] top-1 mismatch")

    def _assert_causallm_case_close(
        self,
        *,
        name: str,
        actual_scores,
        actual_tlp,
        expected_scores,
        expected_tlp,
        label: str,
    ) -> None:
        self._assert_scores_close(
            actual_scores=actual_scores,
            expected_scores=expected_scores,
            label=label,
            check_top1=(name != "raw_logprobs"),
        )
        if name == "raw_logprobs":
            self._assert_token_logprobs_close(
                actual_tlp=actual_tlp, expected_tlp=expected_tlp, label=label
            )
        else:
            self.assertIsNone(actual_tlp)
            self.assertIsNone(expected_tlp)

    def test_causallm_breakable_pcg_matches_eager_mis_and_replays_graphs(self):
        calls_map = self._causallm_calls()
        names = list(calls_map)
        calls = list(calls_map.values())

        common = dict(
            model_path=TEST_MODEL_NAME,
            enable_mis=True,
            attention_backend="flashinfer",
            disable_radix_cache=True,
            chunked_prefill_size=-1,
            mem_fraction_static=0.20,
            log_level="error",
        )
        eager_outs = dict(
            zip(
                names,
                _collect_engine_outputs(
                    engine_kwargs=dict(common, disable_cuda_graph=True),
                    calls=calls,
                ),
            )
        )
        graph_outs = dict(
            zip(
                names,
                _collect_engine_outputs(engine_kwargs=common, calls=calls),
            )
        )

        for name in names:
            g_scores, _, g_tlp = graph_outs[name]
            e_scores, _, e_tlp = eager_outs[name]
            self._assert_causallm_case_close(
                name=name,
                actual_scores=g_scores,
                actual_tlp=g_tlp,
                expected_scores=e_scores,
                expected_tlp=e_tlp,
                label=name,
            )

        port = find_available_port(31000)
        base_url = f"http://127.0.0.1:{port}"
        proc = popen_launch_server(
            TEST_MODEL_NAME,
            base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--enable-mis",
                "--attention-backend",
                "flashinfer",
                "--disable-radix-cache",
                "--chunked-prefill-size",
                "-1",
                "--mem-fraction-static",
                "0.20",
                "--enable-metrics",
            ],
        )
        try:
            info = requests.get(base_url + "/server_info", timeout=30).json()
            self.assertEqual(
                info["cuda_graph_config"]["prefill"]["backend"], "breakable"
            )
            self.assertEqual(info["cuda_graph_config"]["decode"]["backend"], "disabled")
            before = _prefill_graph_count(base_url)
            for name in names:
                resp = requests.post(
                    base_url + "/v1/score",
                    json={"model": TEST_MODEL_NAME, **calls_map[name]},
                    timeout=30,
                )
                self.assertEqual(resp.status_code, 200, resp.text)
                body = resp.json()
                e_scores, _, e_tlp = eager_outs[name]
                self._assert_causallm_case_close(
                    name=name,
                    actual_scores=body["scores"],
                    actual_tlp=body.get("token_logprobs"),
                    expected_scores=e_scores,
                    expected_tlp=e_tlp,
                    label=f"http {name}",
                )
            after = _prefill_graph_count(base_url)
            self.assertGreaterEqual(after - before, 4.0)
        finally:
            terminate_and_kill_process_tree(proc)
            torch.cuda.empty_cache()

    def test_seqcls_breakable_pcg_matches_eager_mis(self):
        calls = [
            dict(
                query="Rate this option:",
                items=[" Option A", " Option B", " Option C", " Option D"],
                apply_softmax=True,
                return_pooled_hidden_states=False,
            ),
            dict(
                query="",
                items=[" alpha item", " beta item"],
                apply_softmax=True,
                return_pooled_hidden_states=True,
            ),
        ]
        common = dict(
            model_path=TEST_CLASSIFICATION_BASE_MODEL,
            enable_mis=True,
            attention_backend="flashinfer",
            disable_radix_cache=True,
            chunked_prefill_size=-1,
            mem_fraction_static=0.20,
            log_level="error",
        )
        eager_outs = _collect_engine_outputs(
            engine_kwargs=dict(common, disable_cuda_graph=True), calls=calls
        )
        graph_outs = _collect_engine_outputs(engine_kwargs=common, calls=calls)

        for idx, ((g_scores, g_phs, g_tlp), (e_scores, e_phs, e_tlp)) in enumerate(
            zip(graph_outs, eager_outs)
        ):
            self.assertIsNone(g_tlp)
            self.assertIsNone(e_tlp)
            self._assert_scores_close(
                actual_scores=g_scores,
                expected_scores=e_scores,
                label=f"seqcls call={idx}",
                check_top1=False,
            )
            if idx == 0:
                self.assertIsNone(g_phs)
                self.assertIsNone(e_phs)
            else:
                self.assertIsNotNone(g_phs)
                self.assertIsNotNone(e_phs)
                for gp, ep in zip(g_phs, e_phs):
                    torch.testing.assert_close(gp, ep, rtol=5e-2, atol=5e-2)


def _prefill_graph_count(base_url: str) -> float:
    metrics = requests.get(base_url + "/metrics", timeout=30).text
    matches = re.findall(
        r'^sglang:cuda_graph_passes_total\{[^}]*mode="prefill_cuda_graph"[^}]*\}'
        r"\s+([0-9.eE+-]+)$",
        metrics,
        re.MULTILINE,
    )
    return sum(map(float, matches), 0.0)


def _collect_engine_outputs(*, engine_kwargs, calls):
    engine = Engine(**engine_kwargs)
    try:
        outs = [engine.score(**call) for call in calls]
        return [
            (
                o.scores,
                [t.clone() for t in o.pooled_hidden_states]
                if o.pooled_hidden_states is not None
                else None,
                o.token_logprobs,
            )
            for o in outs
        ]
    finally:
        engine.shutdown()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()
