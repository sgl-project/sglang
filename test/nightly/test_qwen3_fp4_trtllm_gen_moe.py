import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestFlashinferTrtllmGenMoeBackend(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = "nvidia/Qwen3-30B-A3B-NVFP4"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--moe-runner-backend",
                "flashinfer_trtllm",
                "--quantization",
                "modelopt_fp4",
                "--trust-remote-code",
                "--disable-radix-cache",
                "--max-running-requests",
                "1024",
                "--chunked-prefill-size",
                "16384",
                "--mem-fraction-static",
                "0.89",
                "--max-prefill-tokens",
                "16384",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        args = SimpleNamespace(
            num_shots=8,
            data_path=None,
            num_questions=1319,
            max_new_tokens=512,
            parallel=1319,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval(args)
        print(f"{metrics=}")
        self.assertGreater(metrics["accuracy"], 0.88)


if __name__ == "__main__":
    unittest.main()
