import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=300, stage="extra-b", runner_config="8-gpu-h200")

GIGACHAT35_FP8_MODEL = "ai-sage/GigaChat3.5-432B-A28B"

GIGACHAT35_FP8_ARGS = [
    "--tp-size",
    "8",
    "--ep-size",
    "8",
    "--trust-remote-code",
    "--tool-call-parser",
    "gigachat35",
    # GigaChat35's fused silu_and_mul_clamp JIT kernel can't be traced by the
    # default tc_piecewise prefill CUDA-graph backend; disable prefill graph
    "--cuda-graph-backend-prefill",
    "disabled",
]


class TestGigaChat35FP8(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = GIGACHAT35_FP8_MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=GIGACHAT35_FP8_ARGS,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        args = SimpleNamespace(
            model=self.model,
            eval_name="gsm8k",
            num_shots=5,
            num_examples=200,
            max_tokens=16000,
            num_threads=200,
            repeat=1,
            temperature=1.0,
            top_p=0.95,
            base_url=self.base_url,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval(args)
        print(f"{metrics=}")
        # FP8 checkpoint scores ~0.97 on this config (8xH100)
        self.assertGreaterEqual(metrics["score"], 0.93)


if __name__ == "__main__":
    unittest.main()
