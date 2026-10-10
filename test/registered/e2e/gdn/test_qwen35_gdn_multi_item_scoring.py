"""End-to-end MIS coverage for the hybrid Qwen3.5 GDN architecture."""

import asyncio
import os
import unittest

import torch
from transformers import AutoTokenizer

from sglang.srt.entrypoints.engine import Engine
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_HYBRID_GDN_SMALL_MODEL_NAME_FOR_TEST,
    CustomTestCase,
)

register_cuda_ci(est_time=240, stage="extra-a", runner_config="1-gpu-large")


class TestQwen35GDNMultiItemScoring(CustomTestCase):
    model = os.environ.get(
        "QWEN35_GDN_TEST_MODEL", DEFAULT_HYBRID_GDN_SMALL_MODEL_NAME_FOR_TEST
    )
    atol = 2e-2
    rtol = 2e-2

    @classmethod
    def setUpClass(cls):
        cls.engine = Engine(
            model_path=cls.model,
            trust_remote_code=True,
            dtype="bfloat16",
            enable_mis=True,
            attention_backend="flashinfer",
            linear_attn_prefill_backend="triton",
            disable_radix_cache=True,
            chunked_prefill_size=-1,
            mem_fraction_static=0.8,
            log_level="error",
        )
        tokenizer = AutoTokenizer.from_pretrained(cls.model, trust_remote_code=True)
        cls.label_token_ids = [
            tokenizer.encode(label, add_special_tokens=False)[0]
            for label in (" yes", " no")
        ]

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "engine", None) is not None:
            cls.engine.shutdown()
        torch.cuda.empty_cache()

    def _score(self, query, items):
        return self.engine.score(
            query=query,
            items=items,
            label_token_ids=self.label_token_ids,
            apply_softmax=False,
        ).scores

    async def _async_score(self, query, items):
        result = await self.engine.async_score(
            query=query,
            items=items,
            label_token_ids=self.label_token_ids,
            apply_softmax=False,
        )
        return result.scores

    def _pointwise(self, query, items):
        return [self._score(query, [item])[0] for item in items]

    def _assert_scores_close(self, actual, expected):
        torch.testing.assert_close(
            torch.tensor(actual),
            torch.tensor(expected),
            atol=self.atol,
            rtol=self.rtol,
        )

    def test_batched_matches_pointwise_for_varied_requests(self):
        cases = [
            (
                "Decide whether each statement is true:",
                ["The sky is blue.", "Two plus two is five.", "Water freezes."],
            ),
            ("", ["empty query short", "empty query with a much longer item " * 8]),
            ("Classify:", ["one"]),
            (
                "Judge each passage:",
                ["tiny", "medium length passage " * 5, "long passage " * 24],
            ),
        ]

        for query, items in cases:
            with self.subTest(query=query, item_count=len(items)):
                self._assert_scores_close(
                    self._score(query, items), self._pointwise(query, items)
                )

    def test_sibling_changes_and_reordering_do_not_change_target(self):
        query = "Rate each answer:"
        target = "The target answer remains unchanged."
        baseline = self._score(query, [target, "sibling A", "sibling B"])[0]
        changed = self._score(
            query, [target, "a completely different sibling " * 8, "sibling B"]
        )[0]
        reordered = self._score(query, ["sibling B", "sibling A", target])[2]

        self._assert_scores_close(changed, baseline)
        self._assert_scores_close(reordered, baseline)

    def test_concurrent_requests_match_pointwise(self):
        cases = [
            ("Is it an animal?", ["cat", "table", "blue whale"]),
            ("", ["alpha", "beta"]),
            ("Choose:", ["first"]),
            ("Is it a city?", ["Paris", "bread", "Tokyo", "chair"]),
        ]
        expected = [self._pointwise(query, items) for query, items in cases]

        async def gather_scores():
            return await asyncio.gather(
                *(self._async_score(query, items) for query, items in cases)
            )

        actual = self.engine.loop.run_until_complete(gather_scores())
        for actual_scores, expected_scores in zip(actual, expected):
            self._assert_scores_close(actual_scores, expected_scores)


if __name__ == "__main__":
    unittest.main()
