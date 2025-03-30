import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestEvalAccuracyLarge(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--log-level-http", "warning"],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_math(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="math",
            num_examples=5000,
            num_threads=1024,
        )

        metrics = run_eval(args)
        self.assertGreaterEqual(
            metrics["score"], 0.519 - 0.02
        )  # -2% to account for sampling variance


if __name__ == "__main__":
    unittest.main()
