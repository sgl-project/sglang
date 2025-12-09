import unittest
from urllib.parse import urlparse

from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    SimpleNamespace,
    popen_launch_server,
    run_bench_one_batch,
)

MODEL = "Qwen/Qwen2.5-7B-Instruct"
GSM8K_EXP_ACCURACY = 0.84
EXP_PREFILL_LATENCY = 0.045
TOKENS_TO_CAPTURE = [i for i in range(128, 4096, 128)]


class TestPiecewiseGraphPrefillCorrectness(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.url = urlparse(DEFAULT_URL_FOR_TEST)
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--mem-fraction-static",
                0.8,
                "--attention-backend",
                "ascend",
                "--cuda-graph-bs",
                128,
                "--enable-piecewise-cuda-graph",
                "--piecewise-cuda-graph-tokens",
                TOKENS_TO_CAPTURE,
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        print(f"##=== Testing accuracy: {self.model} ===##")
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=1319,
            max_new_tokens=512,
            parallel=128,
            host=f"http://{self.url.hostname}",
            port=int(self.url.port),
        )

        metrics = run_eval_few_shot_gsm8k(args)
        self.assertGreaterEqual(
            metrics["accuracy"],
            GSM8K_EXP_ACCURACY,
        )


class TestPiecewiseGraphPrefillBenchmark(CustomTestCase):

    def test_latency(self):
        print(f"##=== Testing prefill latency: {MODEL} ===##")
        prefill_latency, _, _ = run_bench_one_batch(
            MODEL,
            other_args=[
                "--trust-remote-code",
                "--mem-fraction-static",
                0.8,
                "--attention-backend",
                "ascend",
                "--enable-piecewise-cuda-graph",
                "--piecewise-cuda-graph-tokens",
                TOKENS_TO_CAPTURE,
            ],
        )
        self.assertLess(prefill_latency, EXP_PREFILL_LATENCY)


if __name__ == "__main__":
    unittest.main()
