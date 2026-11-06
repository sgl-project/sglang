# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0.

import unittest

import sglang as sgl
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=600, stage="base-b", runner_config="1-gpu-small")

TARGET = "Qwen/Qwen3-8B"
DRAFT = "tencent/Qwen3-8B-DSpine-b16"
PROMPTS = [
    "The capital of France is",
    "Write a Python function that returns Fibonacci numbers:",
    "Explain why the sky is blue:",
    "The sum of the first ten positive integers is",
]


class TestDSpine(unittest.TestCase):
    def test_greedy_parity_sampling_and_graph_batch_changes(self):
        common = dict(
            model_path=TARGET,
            tp_size=1,
            dtype="bfloat16",
            attention_backend="flashinfer",
            page_size=1,
            mem_fraction_static=0.65,
            context_length=4096,
            max_running_requests=4,
            disable_radix_cache=True,
        )
        params = {"temperature": 0.0, "max_new_tokens": 64, "ignore_eos": True}
        baseline = sgl.Engine(
            **common, disable_cuda_graph=True, disable_overlap_schedule=True
        )
        try:
            reference = baseline.generate(PROMPTS, params)
            reference = [result["output_ids"] for result in reference]
        finally:
            baseline.shutdown()

        for eager in (True, False):
            with self.subTest(eager=eager):
                engine = sgl.Engine(
                    **common,
                    speculative_algorithm="DSPINE",
                    speculative_draft_model_path=DRAFT,
                    speculative_num_draft_tokens=16,
                    disable_cuda_graph=eager,
                    disable_overlap_schedule=eager,
                    cuda_graph_bs_decode=[1, 2, 4],
                )
                try:
                    for batch_size in (4, 1, 2):
                        results = engine.generate(PROMPTS[:batch_size], params)
                        self.assertEqual(
                            [r["output_ids"] for r in results], reference[:batch_size]
                        )
                        self._check_metrics(results)
                    # A sampled batch must carry the realized proposal to verify,
                    # including a greedy request and target-side probability filters.
                    mixed = [
                        params,
                        {
                            "temperature": 0.8,
                            "top_p": 0.9,
                            "top_k": 40,
                            "min_p": 0.05,
                            "max_new_tokens": 64,
                            "ignore_eos": True,
                        },
                    ]
                    results = engine.generate(PROMPTS[:2], mixed)
                    self.assertEqual(results[0]["output_ids"], reference[0])
                    self.assertTrue(all(len(r["output_ids"]) == 64 for r in results))
                    self._check_metrics(results)
                finally:
                    engine.shutdown()

    def _check_metrics(self, results):
        for result in results:
            meta = result["meta_info"]
            self.assertGreater(meta["spec_verify_ct"], 0)
            self.assertGreater(meta["spec_num_proposed_drafts"], 0)
            self.assertGreaterEqual(meta["spec_num_correct_drafts"], 0)
            self.assertLessEqual(
                meta["spec_num_correct_drafts"], meta["spec_num_proposed_drafts"]
            )
            self.assertTrue(0.0 <= meta["spec_accept_rate"] <= 1.0)


if __name__ == "__main__":
    unittest.main()
