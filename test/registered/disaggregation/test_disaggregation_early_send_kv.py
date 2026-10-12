import re
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import requests

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.sgl_eval_utils import run_sgl_eval
from sglang.test.test_utils import DEFAULT_MODEL_NAME_FOR_TEST

register_cuda_ci(est_time=400, stage="base-b", runner_config="2-gpu-large")


class TestDisaggregationEarlySendKV(PDDisaggregationServerBase):
    extra_prefill_env = {"SGLANG_DISAGG_PREFILL_EARLY_SEND_KV": "1"}
    # Batch-invariant prefill so concurrent and sequential first tokens must match.
    extra_prefill_args = ["--enable-deterministic-inference"]

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.launch_all()

    def test_gsm8k(self):
        args = SimpleNamespace(
            base_url=f"http://{self.base_host}:{self.lb_port}",
            eval_name="gsm8k",
            max_tokens=512,
            num_examples=200,
            num_threads=128,
        )
        metrics = run_sgl_eval(args)
        print(f"Evaluation metrics: {metrics}")
        self.assertGreater(metrics["score"], 0.62)

    def test_greedy_matches_sequential(self):
        """Concurrent greedy outputs match the same prompts run one at a time."""
        prompts = [
            f"Shared instructions: answer briefly and precisely. Question {i}: "
            f"what is {i} times {i + 3}?"
            for i in range(32)
        ]

        def generate(prompt):
            res = requests.post(
                self.lb_url + "/generate",
                json={
                    "text": prompt,
                    "sampling_params": {"temperature": 0, "max_new_tokens": 1},
                },
            )
            res.raise_for_status()
            return res.json()["text"]

        sequential = [generate(p) for p in prompts]
        with ThreadPoolExecutor(16) as pool:
            concurrent = list(pool.map(generate, prompts))
        # A single output token is produced by prefill alone, so it does not
        # depend on decode batching; it checks the first token handed to decode.
        self.assertEqual(sequential, concurrent)

    def test_fallback_requests_concurrent(self):
        """Logprob and grammar requests take the scheduler path alongside plain ones."""
        regex = r"Answer: [0-9]{3}-[a-z]{4}\."

        def generate(i):
            body = {
                "text": f"Request {i}: give a short reply.",
                "sampling_params": {"temperature": 0, "max_new_tokens": 16},
            }
            if i % 3 == 1:
                body["return_logprob"] = True
            elif i % 3 == 2:
                body["sampling_params"]["regex"] = regex
            res = requests.post(self.lb_url + "/generate", json=body)
            res.raise_for_status()
            return i, res.json()

        with ThreadPoolExecutor(24) as pool:
            results = list(pool.map(generate, range(48)))

        for i, out in results:
            self.assertTrue(out["text"], f"empty output for request {i}")
            meta = out["meta_info"]
            if i % 3 == 1:
                self.assertEqual(
                    len(meta["output_token_logprobs"]), meta["completion_tokens"]
                )
            elif i % 3 == 2:
                self.assertIsNotNone(re.fullmatch(regex, out["text"]), out["text"])


if __name__ == "__main__":
    unittest.main()
