"""Qwen3.5-9B NEXTN and breadth-1 NGRAM decoding on Intel XPU."""

import unittest

import requests
import torch

from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)
from sglang.test.xpu.simple_eval_gsm8k_xpu_mixin import SimpleEvalGSM8KXPUMixin

register_xpu_ci(est_time=900, suite="nightly-xpu-kernel-main-4-gpu", nightly=True)

MODEL = "Qwen/Qwen3.5-9B"
SERVER_ARGS = SimpleEvalGSM8KXPUMixin.other_args + [
    "--tp-size",
    "1",
    "--page-size",
    "128",
    "--linear-attn-backend",
    "intel_xpu",
    "--mem-fraction-static",
    "0.98",
    "--max-running-requests",
    "4",
    "--max-mamba-cache-size",
    "8",
    "--max-total-tokens",
    "8192",
    "--max-prefill-tokens",
    "8192",
    "--chunked-prefill-size",
    "-1",
    "--startup-weight-load-mode",
    "serial",
    "--skip-server-warmup",
]
NEXTN_ARGS = [
    "--speculative-algorithm",
    "NEXTN",
    "--speculative-num-steps",
    "3",
    "--speculative-eagle-topk",
    "1",
    "--speculative-num-draft-tokens",
    "4",
]
NGRAM_ARGS = [
    "--speculative-algorithm",
    "NGRAM",
    "--speculative-num-draft-tokens",
    "4",
    "--speculative-ngram-max-bfs-breadth",
    "1",
]


@unittest.skipUnless(torch.xpu.is_available(), "Intel XPU is required")
class TestQwen3_5_9BXPUSpecDecode(CustomTestCase):
    def generate(self, extra_args):
        process = popen_launch_server(
            MODEL,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=SERVER_ARGS + extra_args,
        )
        try:
            response = requests.post(
                DEFAULT_URL_FOR_TEST + "/generate",
                json={
                    "text": "The capital of France is",
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 64,
                        "ignore_eos": True,
                    },
                },
                timeout=180,
            )
            response.raise_for_status()
            return response.json()
        finally:
            terminate_and_kill_process_tree(process)

    def test_speculative_modes_match_greedy(self):
        baseline = self.generate([])
        self.assertEqual(baseline["meta_info"]["completion_tokens"], 64)

        for algorithm, args in (("NEXTN", NEXTN_ARGS), ("NGRAM", NGRAM_ARGS)):
            with self.subTest(algorithm=algorithm):
                speculative = self.generate(args)
                self.assertEqual(speculative["meta_info"]["completion_tokens"], 64)
                self.assertEqual(speculative["text"], baseline["text"])
                self.assertGreater(speculative["meta_info"]["spec_verify_ct"], 0)
                self.assertGreater(
                    speculative["meta_info"]["spec_num_correct_drafts"], 0
                )


if __name__ == "__main__":
    unittest.main()
