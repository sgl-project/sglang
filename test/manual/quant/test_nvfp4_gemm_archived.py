"""Archived test classes split out of test/registered/quant/test_nvfp4_gemm.py.

Originally registered with `register_cuda_ci(...)`. Moved here as part of
the per-commit pruning effort to keep the code reachable manually.
Run with `python3 test/manual/quant/test_nvfp4_gemm_archived.py`.
"""

import unittest
from types import SimpleNamespace
from urllib.parse import urlparse

from sglang.srt.utils import get_device_sm, kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
    try_cached_model,
)

MODEL_PATH = "nvidia/Llama-3.1-8B-Instruct-NVFP4"


class FP4GemmBase:
    backend = None

    @classmethod
    def setUpClass(cls):
        if cls.backend is None:
            raise NotImplementedError("Subclass must set 'backend' attribute")
        cls.model = try_cached_model(MODEL_PATH)
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--trust-remote-code",
            "--quantization",
            "modelopt_fp4",
            "--fp4-gemm-backend",
            cls.backend,
        ]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        parsed_url = urlparse(self.base_url)
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=1319,
            num_threads=200,
        )
        metrics = run_eval(args)
        print(metrics)

        self.assertGreater(metrics["score"], 0.64)


@unittest.skipIf(get_device_sm() < 100, "Test requires CUDA SM 100 or higher")
class TestFP4GemmAuto(FP4GemmBase, unittest.TestCase):
    backend = "auto"


if __name__ == "__main__":
    unittest.main()
