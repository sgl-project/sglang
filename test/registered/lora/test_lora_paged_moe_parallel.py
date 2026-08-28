"""Paged MoE LoRA output parity across single-GPU, TP, and EP+TP.

Each configuration runs in a fresh server process. This matches serving usage
and prevents process-global CUDA or distributed state from leaking between the
flat reference and direct-paged runs.
"""

import math
import os
import subprocess
import sys
import time
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

import requests

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.lora_utils import (
    MOE_BASE_MODEL_PATH,
    MOE_LORA_PATH,
    MOE_LORA_TEST_PROMPTS,
)
from sglang.test.test_utils import CustomTestCase, kill_process_tree

register_cuda_ci(est_time=360, stage="extra-a", runner_config="2-gpu-large")

MODEL_PATH = os.environ.get("SGLANG_TEST_MOE_MODEL_PATH", MOE_BASE_MODEL_PATH)
LORA_PATH = os.environ.get("SGLANG_TEST_MOE_LORA_PATH", MOE_LORA_PATH)
LORA_NAME = "test"
MAX_NEW_TOKENS = 8
LONG_OUTPUT_TOKENS = 32
LONG_OUTPUT_CONCURRENCY = 4
LOGPROB_THRESHOLD = 5e-4


def _wait_for_server(process: subprocess.Popen, base_url: str) -> None:
    deadline = time.monotonic() + 300
    session = requests.Session()
    session.trust_env = False
    try:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"Server exited with code {process.returncode}")
            try:
                response = session.get(base_url + "/v1/models", timeout=2)
                if response.status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(2)
    finally:
        session.close()
    raise TimeoutError("Paged MoE test server did not become ready within 300s")


def _run_moe_lora(
    *,
    tp_size: int,
    ep_size: int = 1,
    paged: bool,
    port: int,
    max_new_tokens: int = MAX_NEW_TOKENS,
    concurrency: int = 1,
) -> List[Dict[str, object]]:
    base_url = f"http://127.0.0.1:{port}"
    command = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        MODEL_PATH,
        "--enable-lora",
        f"--lora-paths={LORA_NAME}={LORA_PATH}",
        "--max-lora-rank",
        "16",
        "--max-loras-per-batch",
        "2",
        "--max-loaded-loras",
        "2",
        "--lora-target-modules",
        "gate_up_proj",
        "down_proj",
        "--context-length",
        "512",
        "--mem-fraction-static",
        "0.70",
        "--attention-backend",
        "fa3",
        "--cuda-graph-backend-decode",
        "disabled",
        "--cuda-graph-backend-prefill",
        "disabled",
        "--disable-radix-cache",
        "--disable-custom-all-reduce",
        "--random-seed",
        "42",
        "--tp-size",
        str(tp_size),
        "--ep-size",
        str(ep_size),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    if paged:
        command.extend(["--lora-page-rank-size", "8", "--lora-pages", "4"])

    process = subprocess.Popen(command, env=os.environ.copy())
    try:
        _wait_for_server(process, base_url)

        def request_one(prompt):
            with requests.Session() as request_session:
                request_session.trust_env = False
                response = request_session.post(
                    base_url + "/generate",
                    json={
                        "text": prompt,
                        "lora_path": LORA_NAME,
                        "sampling_params": {
                            "max_new_tokens": max_new_tokens,
                            "temperature": 0,
                            "ignore_eos": True,
                        },
                        "return_logprob": True,
                    },
                    timeout=120,
                )
            response.raise_for_status()
            data = response.json()
            meta_info = data["meta_info"]
            return {
                "text": data["text"],
                "output_ids": data.get("output_ids")
                or meta_info.get("output_ids")
                or [],
                "logprobs": [item[0] for item in meta_info["output_token_logprobs"]],
            }

        prompts = MOE_LORA_TEST_PROMPTS[:8]
        if concurrency == 1:
            return [request_one(prompt) for prompt in prompts[:5]]

        indexed_outputs = {}
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {
                executor.submit(request_one, prompt): index
                for index, prompt in enumerate(prompts)
            }
            for future in as_completed(futures):
                indexed_outputs[futures[future]] = future.result()
        return [indexed_outputs[index] for index in range(len(prompts))]
    finally:
        kill_process_tree(process.pid)


class TestPagedMoELoRAParallelParity(CustomTestCase):
    def _assert_outputs_valid(self, outputs, label, expected_tokens):
        self.assertEqual(len(outputs), 8, label)
        for prompt_id, output in enumerate(outputs):
            self.assertEqual(
                len(output["output_ids"]),
                expected_tokens,
                f"{label}, prompt {prompt_id}",
            )
            self.assertEqual(
                len(output["logprobs"]),
                expected_tokens,
                f"{label}, prompt {prompt_id}",
            )
            self.assertTrue(
                all(math.isfinite(value) for value in output["logprobs"]),
                f"{label}, prompt {prompt_id}",
            )

    def _assert_outputs_match(self, expected, actual, label):
        self.assertEqual(len(expected), len(actual), label)
        for prompt_id, (reference, observed) in enumerate(zip(expected, actual)):
            self.assertEqual(
                reference["text"],
                observed["text"],
                f"{label}, prompt {prompt_id}",
            )
            self.assertEqual(
                reference["output_ids"],
                observed["output_ids"],
                f"{label}, prompt {prompt_id}",
            )
            expected_logprobs = reference["logprobs"]
            observed_logprobs = observed["logprobs"]
            self.assertEqual(
                len(expected_logprobs),
                len(observed_logprobs),
                f"{label}, prompt {prompt_id}",
            )
            max_diff = max(
                (
                    abs(lhs - rhs)
                    for lhs, rhs in zip(expected_logprobs, observed_logprobs)
                ),
                default=0.0,
            )
            self.assertLessEqual(
                max_diff,
                LOGPROB_THRESHOLD,
                f"{label}, prompt {prompt_id}: max logprob diff " f"{max_diff:.6e}",
            )

    def test_flat_vs_paged_tp_and_ep_parity(self):
        configurations = (
            ("Paged TP=1", 1, 1),
            ("Paged TP=2", 2, 1),
            ("Paged EP=2+TP=2", 2, 2),
        )
        for offset, (label, tp_size, ep_size) in enumerate(configurations):
            with self.subTest(configuration=label):
                reference = _run_moe_lora(
                    tp_size=tp_size,
                    ep_size=ep_size,
                    paged=False,
                    port=13550 + 2 * offset,
                )
                actual = _run_moe_lora(
                    tp_size=tp_size,
                    ep_size=ep_size,
                    paged=True,
                    port=13551 + 2 * offset,
                )
                self._assert_outputs_match(reference, actual, label)

    def test_long_output_concurrent_tp1_smoke(self):
        outputs = _run_moe_lora(
            tp_size=1,
            paged=True,
            port=13555,
            max_new_tokens=LONG_OUTPUT_TOKENS,
            concurrency=LONG_OUTPUT_CONCURRENCY,
        )
        self._assert_outputs_valid(outputs, "Paged TP=1 concurrent", LONG_OUTPUT_TOKENS)


if __name__ == "__main__":
    unittest.main()
