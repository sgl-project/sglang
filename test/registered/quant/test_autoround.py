"""
Usage:
python3 -m unittest test_autoround.TestAutoRound.test_mmlu
"""

import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_AUTOROUND_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=77, suite="stage-b-test-large-1-gpu")


class TestAutoRound(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST

    @classmethod
    def tearDownClass(cls):
        pass

    def test_mmlu(self):
        device = "auto"
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
                    if "Llama" in model:
                        self.assertGreaterEqual(metrics["score"], 0.6)
                    else:
                        self.assertGreaterEqual(metrics["score"], 0.26)
                finally:
                    kill_process_tree(process.pid)
                    print(f"[INFO] Server for {model} stopped.")


if __name__ == "__main__":
    unittest.main()
