"""
Usage:
SGLANG_USE_CPU_ENGINE=1 python3 -m unittest test_autoround

CPU accuracy test for AutoRound INT4 checkpoints. Covers both AutoRound packing
formats (auto_round:auto_gptq / auto_round:auto_awq) by launching a server and
running an MMLU eval. Dense linear layers run on any x86 CPU (AVX512 path or
scalar fallback); AMX is only required for INT4 MoE. Skipped on non-x86 hosts.
"""

import os
import unittest
from types import SimpleNamespace

from sglang.srt.utils import is_host_cpu_x86, kill_process_tree
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_AUTOROUND_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cpu_ci(est_time=330, suite="base-a-test-cpu")


@unittest.skipUnless(
    is_host_cpu_x86(),
    "AutoRound dense linear on CPU requires an x86 CPU.",
)
class TestAutoRoundCPU(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST

    def test_mmlu(self):
        device = "cpu"
        for model in DEFAULT_AUTOROUND_MODEL_NAME_FOR_TEST:
            with self.subTest(model=model):
                print(f"\n[INFO] Launching server for model: {model}")
                process = popen_launch_server(
                    model,
                    self.base_url,
                    timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                    other_args=["--trust-remote-code", "--quantization", "auto-round"],
                    device=device,
                )

                try:
                    args = SimpleNamespace(
                        base_url=self.base_url,
                        model=model,
                        eval_name="mmlu",
                        num_examples=32,
                        num_threads=32,
                        device=device,
                    )
                    metrics = run_eval(args)
                    self.assertGreaterEqual(metrics["score"], 0.25)
                finally:
                    kill_process_tree(process.pid)
                    print(f"[INFO] Server for {model} stopped.")


if __name__ == "__main__":
    os.environ.setdefault("SGLANG_USE_CPU_ENGINE", "1")
    unittest.main()
