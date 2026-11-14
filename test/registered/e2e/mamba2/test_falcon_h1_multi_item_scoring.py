"""End-to-end MIS coverage for the hybrid Falcon-H1 Mamba-2 architecture."""

import asyncio
import gc
import os
import unittest

import torch
from transformers import AutoTokenizer

from sglang.srt.entrypoints.engine import Engine
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=90, stage="extra-a", runner_config="1-gpu-small")

_PARITY_TEXT_CASES = [
    (
        "Decide whether each statement is true:",
        ["The sky is blue.", "Two plus two is five.", "Water freezes."],
    ),
    ("", ["empty query short", "empty query with a much longer item " * 8]),
    ("Classify:", ["one"]),
    (
        "Judge each passage:",
        [
            "tiny",
            "medium length passage " * 5,
            "long passage with details " * 60,
        ],
    ),
]

_CONCURRENT_TEXT_CASES = [
    ("Is it an animal?", ["cat", "table", "blue whale"]),
    ("", ["alpha", "beta"]),
    ("Choose:", ["first"]),
    ("Is it a city?", ["Paris", "bread", "Tokyo", "chair"]),
]


class TestFalconH1MultiItemScoring(CustomTestCase):
    model = os.environ.get("MAMBA2_MIS_TEST_MODEL", "tiiuae/Falcon-H1-0.5B-Instruct")
    engine = None
    label_token_ids = []
    parity_token_cases = []
    parity_references = []
    concurrent_token_cases = []
    concurrent_references = []
    logprob_atol = 5e-2
    logprob_rtol = 1.5e-2

    @classmethod
    def setUpClass(cls):
        tokenizer = AutoTokenizer.from_pretrained(cls.model, trust_remote_code=True)
        cls.label_token_ids = [
            tokenizer.encode(label, add_special_tokens=False)[0]
            for label in (" yes", " no")
        ]
        cls.parity_token_cases = _tokenize_cases(
            tokenizer=tokenizer, text_cases=_PARITY_TEXT_CASES
        )
        cls.concurrent_token_cases = _tokenize_cases(
            tokenizer=tokenizer, text_cases=_CONCURRENT_TEXT_CASES
        )

        # Compute independent ground truth with MIS disabled, then tear it down
        # before starting the MIS engine to keep GPU memory bounded.
        ref_engine = None
        try:
            ref_engine = _start_engine(model_path=cls.model, enable_mis=False)
            cls.parity_references = _compute_pointwise_references(
                engine=ref_engine,
                token_cases=cls.parity_token_cases,
                label_token_ids=cls.label_token_ids,
            )
            cls.concurrent_references = _compute_pointwise_references(
                engine=ref_engine,
                token_cases=cls.concurrent_token_cases,
                label_token_ids=cls.label_token_ids,
            )
        finally:
            if ref_engine is not None:
                ref_engine.shutdown()
                del ref_engine
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        cls.engine = _start_engine(model_path=cls.model, enable_mis=True)

    @classmethod
    def tearDownClass(cls):
        if cls.engine is not None:
            cls.engine.shutdown()
            cls.engine = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def test_batched_matches_non_mis_reference_for_varied_requests(self):
        for idx, ((query_ids, items_ids), (ref_scores, ref_logprobs)) in enumerate(
            zip(self.parity_token_cases, self.parity_references)
        ):
            with self.subTest(
                case_idx=idx, query_len=len(query_ids), item_count=len(items_ids)
            ):
                result = self._score_with_logprobs(query=query_ids, items=items_ids)
                self._assert_result_matches_reference(
                    result=result,
                    expected_scores=ref_scores,
                    expected_logprobs=ref_logprobs,
                )

    def test_sibling_changes_and_reordering_do_not_change_target(self):
        query = "Rate each answer:"
        target = "The target answer remains unchanged."
        baseline = self._score_with_logprobs(
            query=query, items=[target, "sibling A", "sibling B"]
        )
        changed = self._score_with_logprobs(
            query=query,
            items=[target, "a completely different sibling " * 8, "sibling B"],
        )
        reordered = self._score_with_logprobs(
            query=query, items=["sibling B", "sibling A", target]
        )

        self._assert_result_matches_reference(
            result=changed,
            expected_scores=[baseline.scores[0]],
            expected_logprobs=[baseline.token_logprobs[0]],
            compare_rows=[0],
        )
        self._assert_result_matches_reference(
            result=reordered,
            expected_scores=[baseline.scores[0]],
            expected_logprobs=[baseline.token_logprobs[0]],
            compare_rows=[2],
        )

    def test_concurrent_requests_match_non_mis_reference(self):
        async def gather_scores():
            return await asyncio.gather(
                *(
                    self._async_score_with_logprobs(query=query_ids, items=items_ids)
                    for query_ids, items_ids in self.concurrent_token_cases
                )
            )

        actual_results = self.engine.loop.run_until_complete(gather_scores())
        for idx, (result, (ref_scores, ref_logprobs)) in enumerate(
            zip(actual_results, self.concurrent_references)
        ):
            with self.subTest(case_idx=idx):
                self._assert_result_matches_reference(
                    result=result,
                    expected_scores=ref_scores,
                    expected_logprobs=ref_logprobs,
                )

    def test_empty_items_rejected_and_engine_recovers(self):
        for query, bad_items in (
            ([1, 2], [[]]),
            ("Rate:", [""]),
            ("Rate:", [" valid", ""]),
        ):
            with self.subTest(query=query, bad_items=bad_items):
                with self.assertRaisesRegex(ValueError, "empty items"):
                    self._score_with_logprobs(query=query, items=bad_items)

        empty_result = self._score_with_logprobs(query="Rate:", items=[])
        self.assertEqual(empty_result.scores, [])
        self.assertEqual(empty_result.token_logprobs, [])

        recovery = self._score_with_logprobs(
            query="Decide whether each statement is true:",
            items=["The sky is blue.", "Water freezes."],
        )
        self._assert_non_degenerate_result(result=recovery)

    def _score_with_logprobs(self, *, query, items):
        return self.engine.score(
            query=query,
            items=items,
            label_token_ids=self.label_token_ids,
            apply_softmax=False,
            return_token_logprobs=True,
        )

    async def _async_score_with_logprobs(self, *, query, items):
        return await self.engine.async_score(
            query=query,
            items=items,
            label_token_ids=self.label_token_ids,
            apply_softmax=False,
            return_token_logprobs=True,
        )

    def _assert_non_degenerate_result(self, *, result, row_indices=None):
        self.assertIsNotNone(result.token_logprobs)
        rows = (
            list(range(len(result.scores)))
            if row_indices is None
            else list(row_indices)
        )
        scores_t = torch.tensor([result.scores[i] for i in rows], dtype=torch.float32)
        logprobs_t = torch.tensor(
            [result.token_logprobs[i] for i in rows], dtype=torch.float32
        )
        expected_shape = (len(rows), len(self.label_token_ids))
        self.assertEqual(tuple(scores_t.shape), expected_shape)
        self.assertEqual(tuple(logprobs_t.shape), expected_shape)
        self.assertTrue(torch.isfinite(scores_t).all().item())
        self.assertTrue(torch.isfinite(logprobs_t).all().item())
        self.assertTrue((logprobs_t < -1e-4).all().item())
        self.assertTrue((scores_t > 0.0).all().item())
        torch.testing.assert_close(
            torch.exp(logprobs_t), scores_t, atol=1e-5, rtol=1e-4
        )
        if len(rows) >= 2:
            self.assertFalse(
                torch.allclose(logprobs_t[0], logprobs_t[1], atol=1e-3, rtol=0.0)
            )
        return scores_t, logprobs_t

    def _assert_result_matches_reference(
        self,
        *,
        result,
        expected_scores,
        expected_logprobs,
        compare_rows=None,
    ):
        actual_scores_t, actual_logprobs_t = self._assert_non_degenerate_result(
            result=result, row_indices=compare_rows
        )
        if compare_rows is None:
            ref_scores_t = torch.tensor(expected_scores, dtype=torch.float32)
            ref_logprobs_t = torch.tensor(expected_logprobs, dtype=torch.float32)
        else:
            self.assertEqual(len(expected_scores), len(compare_rows))
            self.assertEqual(len(expected_logprobs), len(compare_rows))
            ref_scores_t = torch.tensor(expected_scores, dtype=torch.float32)
            ref_logprobs_t = torch.tensor(expected_logprobs, dtype=torch.float32)
        max_lp_diff = float((actual_logprobs_t - ref_logprobs_t).abs().max().item())
        max_sc_diff = float((actual_scores_t - ref_scores_t).abs().max().item())
        print(
            f"Falcon-H1 MIS max_logprob_abs_diff={max_lp_diff:.6e} "
            f"max_score_abs_diff={max_sc_diff:.6e}"
        )
        torch.testing.assert_close(
            actual_logprobs_t,
            ref_logprobs_t,
            atol=self.logprob_atol,
            rtol=self.logprob_rtol,
        )


def _start_engine(*, model_path: str, enable_mis: bool) -> Engine:
    # Falcon-H1 stacks 36 hybrid Mamba-2 + Attention layers; 7-bit-mantissa
    # bfloat16 batching shifts tail logprobs by ~0.15-0.35 nats even when
    # enable_mis=False, whereas 10-bit-mantissa float16 bounds multi-chunk
    # parity within atol=5e-2, rtol=1.5e-2 (bfloat16 parity is in test_mamba2.py).
    return Engine(
        model_path=model_path,
        trust_remote_code=True,
        dtype="float16",
        enable_mis=enable_mis,
        disable_cuda_graph=True,
        attention_backend="flashinfer",
        disable_radix_cache=True,
        chunked_prefill_size=-1,
        mem_fraction_static=0.40,
        log_level="error",
    )


def _tokenize_cases(*, tokenizer, text_cases):
    tokenized = []
    for query, items in text_cases:
        query_ids = [] if query == "" else tokenizer.encode(query)
        items_ids = [tokenizer.encode(item, add_special_tokens=False) for item in items]
        for item_ids in items_ids:
            assert len(item_ids) > 0
        tokenized.append((query_ids, items_ids))
    return tokenized


def _compute_pointwise_references(*, engine: Engine, token_cases, label_token_ids):
    references = []
    for query_ids, items_ids in token_cases:
        item_scores = []
        item_logprobs = []
        for item_ids in items_ids:
            res = engine.score(
                query=query_ids,
                items=[item_ids],
                label_token_ids=label_token_ids,
                apply_softmax=False,
                return_token_logprobs=True,
            )
            item_scores.append(res.scores[0])
            item_logprobs.append(res.token_logprobs[0])
        references.append((item_scores, item_logprobs))
    return references


if __name__ == "__main__":
    unittest.main()
