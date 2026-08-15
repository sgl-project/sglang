"""Archived test classes split out of test/registered/quant/test_awq.py.

Originally registered with `register_cuda_ci(...)`. Moved here as part of
the per-commit pruning effort to keep the code reachable manually.
Run with `python3 test/manual/quant/test_awq_archived.py`.
"""

import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_amd_ci,
    popen_launch_server,
)


@unittest.skipIf(is_in_amd_ci(), "AWQ Marlin is not supported on AMD GPUs")
class TestAWQMarlinFloat16(CustomTestCase):
    """
    Verify that the model can be loaded with float16 dtype and awq_marlin quantization
    """

    @classmethod
    def setUpClass(cls):
        cls.model = "QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--dtype", "float16", "--quantization", "awq_marlin"],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_mmlu(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            num_examples=64,
            num_threads=32,
        )

        metrics = run_eval(args)
        self.assertGreater(metrics["score"], 0.85)


if __name__ == "__main__":
    unittest.main()
