import unittest
from types import SimpleNamespace

from sglang.srt.utils import get_device_sm, kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.few_shot_gsm8k import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=500, suite="nightly-4-gpu-b200", nightly=True)

QWEN3_NEXT_MODEL_FP4 = "nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4"

ACC_THRESHOLDS = {
    QWEN3_NEXT_MODEL_FP4: {"kl_div": 0.0025, "gsm8k": 0.93},
}


@unittest.skipIf(
    get_device_sm() < 100, "Test requires CUDA SM 100 or higher (Blackwell)"
)
class TestQwen3NextFp4(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = QWEN3_NEXT_MODEL_FP4
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tp-size",
                "4",
                "--chunked-prefill-size",
                "2048",
                "--quantization",
                "modelopt_fp4",
                "--mamba-scheduler-strategy",
                "extra_buffer",
                "--mamba-track-interval",
                "128",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval(args)
        print(f"{metrics=}")
        self.assertGreaterEqual(
            metrics["accuracy"], ACC_THRESHOLDS[self.model]["gsm8k"]
        )


if __name__ == "__main__":
    unittest.main()
