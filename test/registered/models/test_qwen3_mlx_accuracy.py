import os
import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    try_cached_model,
)

MODEL_PATH = "Qwen/Qwen3-0.6B"

# TODO: hook this into the appropriate Apple/Mac CI suite once maintainer guidance is confirmed.


class TestQwen3MlxAccuracy(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = try_cached_model(MODEL_PATH)
        cls.base_url = DEFAULT_URL_FOR_TEST

        env = os.environ.copy()
        env["SGLANG_USE_MLX"] = "1"

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--tp-size",
                "1",
                "--disable-radix-cache",
                "--disable-cuda-graph",
            ],
            env=env,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process is not None:
            kill_process_tree(cls.process.pid)

    def test_gsm8k_accuracy(self):
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=25,
            max_new_tokens=256,
            parallel=16,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
            temperature=0.0,
        )

        metrics = run_eval(args)
        print(f"{metrics=}")

        # Lightweight regression guard for the Apple MLX path.
        # Thresholds are calibrated from repeated local runs and are intentionally conservative.
        self.assertGreater(metrics["accuracy"], 0.40)
        self.assertLess(metrics["invalid"], 0.10)


if __name__ == "__main__":
    unittest.main()
