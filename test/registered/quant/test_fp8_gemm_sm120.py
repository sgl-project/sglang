import unittest
from types import SimpleNamespace
from urllib.parse import urlparse

from sglang.srt.utils import get_device_sm, kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
    try_cached_model,
)

register_cuda_ci(est_time=144, suite="stage-b-test-1-gpu-small")

PERTENSOR_MODEL_PATH = "nvidia/Llama-3.1-8B-Instruct-FP8"
BLOCKWISE_MODEL_PATH = "Qwen/Qwen3-4B-Instruct-2507-FP8"


class FP8GemmSM120Base:
    model_path = None
    backend = None
    quantization = None

    @classmethod
    def setUpClass(cls):
        if cls.backend is None:
            raise NotImplementedError("Subclass must set 'backend' attribute")
        cls.model = try_cached_model(cls.model_path)
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--trust-remote-code",
            "--fp8-gemm-backend",
            cls.backend,
            "--disable-piecewise-cuda-graph",
        ]
        if cls.quantization:
            other_args += ["--quantization", cls.quantization]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process"):
            kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        parsed_url = urlparse(self.base_url)
        args = SimpleNamespace(
            num_shots=self.num_shots,
            data_path=None,
            num_questions=1319,
            max_new_tokens=512,
            parallel=200,
            host=parsed_url.hostname,
            port=parsed_url.port,
        )
        metrics = run_eval_few_shot_gsm8k(args)
        print(f"{metrics=}")
        self.assertGreaterEqual(metrics["accuracy"], self.accuracy_threshold)


@unittest.skipIf(get_device_sm() < 100, "Test requires CUDA SM 100 or higher")
class TestFP8PerTensorGemmSM120Auto(FP8GemmSM120Base, unittest.TestCase):
    model_path = PERTENSOR_MODEL_PATH
    backend = "auto"
    quantization = "modelopt_fp8"
    num_shots = 5
    accuracy_threshold = 0.73


@unittest.skipIf(get_device_sm() < 100, "Test requires CUDA SM 100 or higher")
class TestFP8BlockwiseGemmSM120Auto(FP8GemmSM120Base, unittest.TestCase):
    model_path = BLOCKWISE_MODEL_PATH
    backend = "auto"
    num_shots = 8
    accuracy_threshold = 0.87


if __name__ == "__main__":
    unittest.main()
