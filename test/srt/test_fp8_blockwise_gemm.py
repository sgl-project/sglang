import unittest
from types import SimpleNamespace
from urllib.parse import urlparse

from sglang.srt.utils import get_device_sm, kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
    try_cached_model,
)

MODEL_PATH = "Qwen/Qwen3-4B-Instruct-2507-FP8"


class FP8BlockwiseGemmBase:
    backend = None

    @classmethod
    def setUpClass(cls):
        if cls.backend is None:
            raise NotImplementedError("Subclass must set 'backend' attribute")
        cls.model = try_cached_model(MODEL_PATH)
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--trust-remote-code",
            "--fp8-gemm-backend",
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
            num_shots=8,
            data_path=None,
            num_questions=1319,
            max_new_tokens=512,
            parallel=200,
            host=f"{parsed_url.scheme}://{parsed_url.hostname}",
            port=parsed_url.port,
        )
        metrics = run_eval_few_shot_gsm8k(args)
        print(metrics)

        self.assertGreaterEqual(metrics["accuracy"], 0.41)


class TestFP8BlockwiseGemmTriton(FP8BlockwiseGemmBase, unittest.TestCase):
    backend = "triton"


class TestFP8BlockwiseGemmDeepGemm(FP8BlockwiseGemmBase, unittest.TestCase):
    backend = "deep_gemm"


@unittest.skipIf(get_device_sm() < 100, "Test requires CUDA SM 100 or higher")
class TestFP8BlockwiseGemmFlashinferTrtllm(FP8BlockwiseGemmBase, unittest.TestCase):
    backend = "flashinfer_trtllm"


if __name__ == "__main__":
    unittest.main()
