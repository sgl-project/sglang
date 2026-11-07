"""DFLASH DP replicas must preserve target outputs with sharded vocabulary tables."""

import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import requests

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

# TP4 / DP2 exercises both local attention TP and idle attention-DP peers.
register_cuda_ci(est_time=480, stage="nightly", runner_config="4-gpu-h100")


class TestDFlashDPAttention(CustomTestCase):
    @contextmanager
    def _server(self, model, revision, draft=None, draft_revision=None):
        args = [
            "--revision",
            revision,
            "--tp-size",
            "4",
            "--attn-dp-size",
            "2",
            "--enable-dp-lm-head",
            "--attention-backend",
            "fa3",
            "--enable-deterministic-inference",
            "--mem-fraction-static",
            "0.6",
            "--context-length",
            "4096",
            "--max-total-tokens",
            "16384",
            "--max-running-requests",
            "8",
            "--chunked-prefill-size",
            "1024",
            "--cuda-graph-backend-prefill",
            "disabled",
            "--cuda-graph-max-bs-decode",
            "4",
            "--random-seed",
            "42",
        ]
        if draft is not None:
            args += [
                "--speculative-algorithm",
                "DFLASH",
                "--speculative-draft-model-path",
                draft,
                "--speculative-draft-model-revision",
                draft_revision,
                "--speculative-draft-attention-backend",
                "flashinfer",
            ]
        process = popen_launch_server(
            model,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=args,
            env={"SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY": "1"},
        )
        try:
            yield
        finally:
            terminate_and_kill_process_tree(process)

    def _generate(self, prompt, rank, length, temperature=0):
        response = requests.post(
            DEFAULT_URL_FOR_TEST + "/generate",
            json={
                "text": prompt,
                "routed_dp_rank": rank,
                "sampling_params": {
                    "temperature": temperature,
                    "max_new_tokens": length,
                    "ignore_eos": True,
                    "sampling_seed": 42,
                },
            },
            timeout=180,
        )
        response.raise_for_status()
        result = response.json()
        self.assertEqual(len(result["output_ids"]), length)
        return result

    def _check_model(self, model, revision, draft, draft_revision):
        prompts = [
            ("The capital of France is", 16),
            (
                "Explain how a compiler translates source code into machine instructions. "
                * 12,
                64,
            ),
        ]
        with self._server(model, revision):
            reference = [
                self._generate(text, 0, n)["output_ids"] for text, n in prompts
            ]
        with self._server(model, revision, draft, draft_revision):
            # Isolated requests exercise idle peers, in both rank directions.
            for rank in (0, 1):
                for (text, n), expected in zip(prompts, reference):
                    result = self._generate(text, rank, n)
                    self.assertEqual(result["output_ids"], expected)
                    self.assertGreater(result["meta_info"]["spec_verify_ct"], 0)
            # Unequal lengths transition between active and idle peers, with
            # target graphs enabled and cached prefixes on both replicas.
            for rank in (0, 1):
                with ThreadPoolExecutor(2) as pool:
                    futures = [
                        pool.submit(self._generate, text, (rank + i) % 2, n)
                        for i, (text, n) in enumerate(prompts)
                    ]
                    for future, expected in zip(futures, reference):
                        self.assertEqual(future.result()["output_ids"], expected)
            sampled = self._generate(
                "Write a short story about an astronomer.", 1, 64, 0.7
            )
            self.assertGreater(sampled["meta_info"]["spec_verify_ct"], 0)

    def test_dflash(self):
        self._check_model(
            "Qwen/Qwen3-4B",
            "1cfa9a7208912126459214e8b04321603b3df60c",
            "z-lab/Qwen3-4B-DFlash-b16",
            "b74e3a329c4d963783143b1e970d95b002be72bd",
        )

    def test_dflash2_hybrid_target(self):
        self._check_model(
            "Qwen/Qwen3.8-27B",
            "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
            "incoai/Qwen3.8-27B-DFlash2",
            "015e795645c74b1a0eeef3b570031fb62e769bc5",
        )


if __name__ == "__main__":
    unittest.main()
