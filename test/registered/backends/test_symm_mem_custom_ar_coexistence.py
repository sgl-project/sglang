"""Test that custom all-reduce can coexist with NCCL symmetric memory.

When --enable-symm-mem is set, the default behavior is to use NCCL symm-mem
for all allreduce operations. Setting SGLANG_ENABLE_CA_WITH_SYMM=1 allows
custom AR to handle small tensors (decode) while NCCL symm-mem handles the rest.

This test verifies that the opt-in CA-with-symm mode does not crash and
maintains accuracy.
"""

import os
import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    try_cached_model,
)

register_cuda_ci(est_time=600, suite="nightly-8-gpu-h200", nightly=True)

MODEL_PATH = "meta-llama/Llama-3.1-70B-Instruct"
SERVER_LAUNCH_TIMEOUT = 600


class TestSymmMemCustomARCoexistence(CustomTestCase):
    """Launch server with symm-mem + CA-with-symm enabled, verify accuracy."""

    @classmethod
    def setUpClass(cls):
        cls.model = try_cached_model(MODEL_PATH)
        cls.base_url = DEFAULT_URL_FOR_TEST
        # Enable CA-with-symm via environment variable
        os.environ["SGLANG_ENABLE_CA_WITH_SYMM"] = "1"
        other_args = [
            "--tensor-parallel-size",
            "8",
            "--enable-symm-mem",
            # custom all-reduce is enabled by default (not passing --disable-custom-all-reduce)
            "--mem-fraction-static",
            "0.85",
            "--disable-radix-cache",
        ]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        os.environ.pop("SGLANG_ENABLE_CA_WITH_SYMM", None)

    def test_gsm8k_accuracy(self):
        """GSM8K accuracy should be >= 0.80 with symm-mem + CA-with-symm."""
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=200,
            num_threads=64,
        )
        metrics = run_eval(args)
        print(f"Eval accuracy of GSM8K (symm-mem + CA-with-symm): {metrics=}")
        self.assertGreater(metrics["score"], 0.80)


if __name__ == "__main__":
    unittest.main()
