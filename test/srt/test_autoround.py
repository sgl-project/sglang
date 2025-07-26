import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_AUTOROUND_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestAutoRound(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST

    @classmethod
    def tearDownClass(cls):
        pass

    def test_mmlu(self):
        for model in DEFAULT_AUTOROUND_MODEL_NAME_FOR_TEST:
            with self.subTest(model=model):
                print(f"\n[INFO] Launching server for model: {model}")
                process = popen_launch_server(
                    model,
                    self.base_url,
                    timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                    other_args=["--trust-remote-code"],
                )

                try:
                    args = SimpleNamespace(
                        base_url=self.base_url,
                        model=model,
                        eval_name="mmlu",
                        num_examples=64,
                        num_threads=32,
                    )
                    metrics = run_eval(args)
                    if "Llama" in model:
                        self.assertGreater(metrics["score"], 0.6)
                    else:
                        self.assertGreater(metrics["score"], 0.26)
                finally:
                    kill_process_tree(process.pid)
                    print(f"[INFO] Server for {model} stopped.")


if __name__ == "__main__":
    unittest.main()
