"""Compare default async and synchronous compiled MLX with eager Torch MPS."""

import importlib.util
import os
import tempfile
import unittest

import requests
import torch

from sglang.test.ci.ci_register import register_mps_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
    try_cached_model,
)

register_mps_ci(est_time=400, suite="stage-b-e2e-mps")


@unittest.skipUnless(
    torch.backends.mps.is_available() and importlib.util.find_spec("mlx") is not None,
    "Requires Torch MPS and MLX",
)
class TestCompiledMlxServing(CustomTestCase):
    backend = "mlx-compiled"
    disable_async = False
    model_name = "Qwen/Qwen3-0.6B"

    def test_decode_and_prefix_reuse_match_eager(self):
        model = try_cached_model(self.model_name)
        results = {}
        distributions = {}
        for backend in ("eager", self.backend):
            env = dict(os.environ, SGLANG_USE_MLX="0")
            log = self.enterContext(tempfile.TemporaryFile(mode="w+"))
            process = popen_launch_server(
                model,
                DEFAULT_URL_FOR_TEST,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                env=env,
                device="mps",
                return_stdout_stderr=(log, log),
                other_args=(
                    [
                        "--device",
                        "mps",
                        "--mps-execution-backend",
                        backend,
                        "--context-length",
                        "1024",
                        "--max-total-tokens",
                        "4096",
                        "--max-running-requests",
                        "4",
                        "--mem-fraction-static",
                        "0.6",
                    ]
                    + (
                        ["--disable-mps-graph-async"]
                        if self.disable_async and backend != "eager"
                        else []
                    )
                ),
            )
            try:
                outputs = []
                for repeat in range(2):
                    response = requests.post(
                        DEFAULT_URL_FOR_TEST + "/generate",
                        json={
                            "return_logprob": True,
                            "top_logprobs_num": 5,
                            "text": [
                                "The capital of France is",
                                "The first five prime numbers are",
                            ],
                            "sampling_params": {
                                "temperature": 0,
                                "max_new_tokens": 8,
                                "ignore_eos": True,
                            },
                        },
                        timeout=180,
                    )
                    response.raise_for_status()
                    payload = response.json()
                    if not repeat:
                        distributions[backend] = [
                            item["meta_info"]["output_top_logprobs"] for item in payload
                        ]
                    outputs.append([item["output_ids"] for item in payload])
                    if repeat:
                        self.assertTrue(
                            all(
                                item["meta_info"]["cached_tokens"] > 0
                                for item in payload
                            )
                        )
                self.assertEqual(outputs[0], outputs[1])
                results[backend] = outputs[0]
                response = requests.post(
                    DEFAULT_URL_FOR_TEST + "/generate",
                    json={
                        "text": ["A short answer:", "A longer answer:"],
                        "sampling_params": [
                            {
                                "temperature": 0,
                                "max_new_tokens": length,
                                "ignore_eos": True,
                                "logit_bias": {"42": 10000},
                            }
                            for length in (2, 7)
                        ],
                    },
                    timeout=180,
                )
                response.raise_for_status()
                self.assertEqual(
                    [item["output_ids"] for item in response.json()],
                    [[42] * 2, [42] * 7],
                )
            finally:
                terminate_and_kill_process_tree(process)
            if backend != "eager":
                log.seek(0)
                output = log.read()
                self.assertIn("Compiled MLX: executions=", output, output)
                self.assertNotIn("using eager MPS", output, output)
        if self.model_name == "Qwen/Qwen3-0.6B":
            self.assertEqual(results["eager"][1], results[self.backend][1])
        for request in range(2):
            for step, (eager_id, mlx_id) in enumerate(
                zip(results["eager"][request], results[self.backend][request])
            ):
                eager = {
                    token: value
                    for value, token, _ in distributions["eager"][request][step]
                }
                mlx = {
                    token: value
                    for value, token, _ in distributions[self.backend][request][step]
                }
                for token in eager.keys() & mlx.keys():
                    self.assertAlmostEqual(eager[token], mlx[token], delta=0.25)
                if eager_id != mlx_id:
                    # BF16 can break a tied greedy choice (e.g. the next country).
                    self.assertIn(mlx_id, eager)
                    self.assertIn(eager_id, mlx)
                    self.assertLessEqual(eager[eager_id] - eager[mlx_id], 0.125)
                    self.assertLessEqual(mlx[mlx_id] - mlx[eager_id], 0.125)
                    break  # Subsequent distributions have different prefixes.


class TestSynchronousCompiledMlxServing(TestCompiledMlxServing):
    disable_async = True


class TestLlamaCompiledMlxServing(TestCompiledMlxServing):
    model_name = "HuggingFaceTB/SmolLM2-135M"


class TestGpt2CompiledMlxServing(TestCompiledMlxServing):
    model_name = "openai-community/gpt2"


if __name__ == "__main__":
    unittest.main()
