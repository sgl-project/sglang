"""End-to-end coverage for FlashInfer prefill context parallelism."""

import unittest

import requests
import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    CustomTestCase,
    find_available_port,
    kill_process_tree,
    popen_launch_server,
)

register_cuda_ci(est_time=300, stage="base-b", runner_config="2-gpu-large")

NUM_GPUS = 2
MODEL_PATH = "Qwen/Qwen3-0.6B"


class TestFlashInferPrefillCPServer(CustomTestCase):
    def _launch_server(self, enable_cp):
        base_url = f"http://127.0.0.1:{find_available_port(30000)}"
        other_args = [
            "--tp-size",
            str(NUM_GPUS),
            "--attention-backend",
            "flashinfer",
            "--cuda-graph-backend-prefill",
            "disabled",
            "--skip-server-warmup",
        ]
        env = None
        if enable_cp:
            other_args.extend(
                [
                    "--attn-cp-size",
                    str(NUM_GPUS),
                    "--moe-dense-tp-size",
                    "1",
                    "--enable-prefill-cp",
                    "--cp-strategy",
                    "zigzag",
                ]
            )
            env = {"SGLANG_ENABLE_CP_V2": "1"}

        process = popen_launch_server(
            MODEL_PATH,
            base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            env=env,
        )
        return process, base_url

    def _generate(self, base_url, input_ids):
        response = requests.post(
            base_url + "/generate",
            json={
                "input_ids": input_ids,
                "sampling_params": {"max_new_tokens": 4, "temperature": 0},
                "return_logprob": True,
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()

    def _assert_output_parity(self, expected, actual):
        self.assertEqual(actual["output_ids"], expected["output_ids"])
        expected_logprobs = expected["meta_info"]["output_token_logprobs"]
        actual_logprobs = actual["meta_info"]["output_token_logprobs"]
        self.assertEqual(len(actual_logprobs), len(expected_logprobs))
        for expected_item, actual_item in zip(expected_logprobs, actual_logprobs):
            self.assertEqual(int(actual_item[1]), int(expected_item[1]))
            self.assertAlmostEqual(
                float(actual_item[0]),
                float(expected_item[0]),
                delta=2e-1,
            )

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.device_count() >= NUM_GPUS,
        f"Need {NUM_GPUS} CUDA devices",
    )
    def test_qwen3_matches_tp_and_reuses_prefix(self):
        prefix = list(range(1000, 1064))
        prompt = prefix + list(range(2000, 2032))

        baseline_process, baseline_url = self._launch_server(enable_cp=False)
        try:
            baseline = self._generate(baseline_url, prompt)
        finally:
            kill_process_tree(baseline_process.pid, wait_timeout=60)

        cp_process, cp_url = self._launch_server(enable_cp=True)
        try:
            cold = self._generate(cp_url, prompt)
            requests.post(cp_url + "/flush_cache", timeout=30).raise_for_status()
            self._generate(cp_url, prefix)
            cached = self._generate(cp_url, prompt)
        finally:
            kill_process_tree(cp_process.pid, wait_timeout=60)

        self._assert_output_parity(baseline, cold)
        self._assert_output_parity(cold, cached)
        self.assertGreater(cached["meta_info"]["cached_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
