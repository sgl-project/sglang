import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestGrok(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = "lmzheng/grok-1"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--load-format",
                "dummy",
                "--json-model-override-args",
                '{"num_hidden_layers": 2}',
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=64,
            num_threads=128,
        )
        metrics = run_eval(args)
        print(f"{metrics=}")

        # It is dummy weights so we only assert the output throughput instead of accuracy.
        self.assertGreater(metrics["output_throughput"], 1000)


if __name__ == "__main__":
    unittest.main()
