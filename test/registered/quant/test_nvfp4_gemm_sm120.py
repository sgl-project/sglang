import unittest
from types import SimpleNamespace
from urllib.parse import urlparse

import torch

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    try_cached_model,
)

register_cuda_ci(est_time=360, suite="stage-b-test-small-1-gpu")

MODEL_PATH = "nvidia/Llama-3.1-8B-Instruct-NVFP4"


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is not available")
class TestNVFP4GemmSM120(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        compute_capability = torch.cuda.get_device_capability()
        if compute_capability != (12, 0):
            raise unittest.SkipTest(
                f"NVFP4 SM120 test requires SM 12.0, but found {compute_capability[0]}.{compute_capability[1]}"
            )
        cls.model = try_cached_model(MODEL_PATH)
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--quantization",
                "modelopt_fp4",
                "--fp4-gemm-backend",
                "flashinfer_cudnn",
                "--disable-piecewise-cuda-graph",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process"):
            kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        parsed_url = urlparse(self.base_url)
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=200,
            host=parsed_url.hostname,
            port=parsed_url.port,
        )
        metrics = run_eval_few_shot_gsm8k(args)
        print(f"{metrics=}")
        self.assertGreater(metrics["accuracy"], 0.64)


if __name__ == "__main__":
    unittest.main(verbosity=3)
