"""Compare NCCL FULL A1/F2 inference with matching native expert sharding."""

import json
import math
import unittest
from contextlib import ExitStack
from urllib.parse import urlparse

import requests

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_ENABLE_ROUTED_EXPERTS_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_pd_server,
    popen_launch_server,
    terminate_and_kill_process_tree,
    try_cached_model,
)
from sglang.utils import wait_for_http_ready

register_cuda_ci(est_time=300, stage="extra-b", runner_config="4-gpu-b200")


class TestAFDNCCL(CustomTestCase):
    prompts = (
        "The capital of France is",
        "The first five prime numbers are 2, 3, 5, 7,",
        "A compiler translates source code into executable instructions. " * 64
        + "\nIn one sentence, a compiler",
    )
    max_new_tokens = 16

    @classmethod
    def setUpClass(cls):
        cls.processes = []
        cls.model = try_cached_model(DEFAULT_ENABLE_ROUTED_EXPERTS_MODEL_NAME_FOR_TEST)
        cls.base_url = DEFAULT_URL_FOR_TEST
        url = urlparse(cls.base_url)
        cls.ffn_addr = f"{url.hostname}:{url.port + 1000}"
        cls.ffn_url = f"http://{cls.ffn_addr}"
        cls.nccl_port = url.port + 100
        # Keep routing and GEMM configuration consistent across different batch shapes.
        cls.common_args = [
            "--dtype",
            "bfloat16",
            "--enable-deterministic-inference",
            "--moe-runner-backend",
            "triton",
            "--moe-a2a-backend",
            "none",
            "--ep-size",
            "1",
            "--cuda-graph-backend-decode",
            "disabled",
            "--cuda-graph-backend-prefill",
            "disabled",
            "--disable-overlap-schedule",
            "--mem-fraction-static",
            "0.7",
            "--context-length",
            "4096",
            "--max-running-requests",
            "16",
            "--random-seed",
            "1",
        ]

    @classmethod
    def tearDownClass(cls):
        cls._cleanup_processes()

    @classmethod
    def _cleanup_processes(cls):
        processes, cls.processes = getattr(cls, "processes", []), []
        with ExitStack() as cleanup:
            for process in processes:
                cleanup.callback(terminate_and_kill_process_tree, process)

    def _collect_outputs(self, configuration):
        outputs = []
        for prompt in self.prompts:
            response = requests.post(
                self.base_url + "/generate",
                json={
                    "text": prompt,
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": self.max_new_tokens,
                        "ignore_eos": True,
                    },
                    "return_logprob": True,
                },
                timeout=120,
            )
            self.assertEqual(response.status_code, 200, response.text)
            meta = response.json()["meta_info"]
            logprobs = meta["output_token_logprobs"]
            self.assertEqual(meta["completion_tokens"], self.max_new_tokens)
            self.assertEqual(len(logprobs), self.max_new_tokens)
            self.assertTrue(all(math.isfinite(item[0]) for item in logprobs))
            outputs.append(logprobs)
        print(json.dumps({"configuration": configuration, "output_logprobs": outputs}))
        return [[item[1] for item in logprobs] for logprobs in outputs]

    def test_full_asymmetric_generation_matches_native(self):
        cls = type(self)
        try:
            # Attention TP1 and expert TP2 match A1/F2 on both sides.
            native = popen_launch_server(
                self.model,
                self.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=self.common_args
                + [
                    "--tp-size",
                    "2",
                    "--attn-dp-size",
                    "2",
                    "--enable-dp-lm-head",
                    "--nccl-port",
                    self.nccl_port,
                ],
            )
            cls.processes.append(native)
            reference = self._collect_outputs("native")
            terminate_and_kill_process_tree(native)
            cls.processes.remove(native)

            # F waits for A's handshake, so its launch must not wait for HTTP readiness.
            ffn = popen_launch_pd_server(
                self.model,
                self.ffn_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=self.common_args
                + [
                    "--afd-role",
                    "ffn",
                    "--afd-backend",
                    "nccl",
                    "--tp-size",
                    "2",
                    "--nccl-port",
                    self.nccl_port + 100,
                ],
            )
            cls.processes.append(ffn)
            attn = popen_launch_server(
                self.model,
                self.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=self.common_args
                + [
                    "--afd-role",
                    "attn",
                    "--afd-backend",
                    "nccl",
                    "--afd-ffn-addr",
                    self.ffn_addr,
                    "--tp-size",
                    "1",
                    "--base-gpu-id",
                    "2",
                    "--nccl-port",
                    self.nccl_port,
                ],
            )
            cls.processes.append(attn)
            wait_for_http_ready(
                self.ffn_url + "/health",
                timeout=60,
                process=ffn,
            )
            actual = self._collect_outputs("afd_full_a1_f2")
            for prompt, expected, observed in zip(
                self.prompts, reference, actual, strict=True
            ):
                with self.subTest(prompt=prompt[:80]):
                    self.assertEqual(observed, expected)
            for process in cls.processes:
                self.assertIsNone(process.poll())
        finally:
            cls._cleanup_processes()


if __name__ == "__main__":
    unittest.main()
