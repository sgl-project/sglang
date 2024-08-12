import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_child_process
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)


class TestEvalAccuracyLarge(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.base_url = "http://127.0.0.1:9157"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=300,
            other_args=["--log-level-http", "warning"],
        )

    @classmethod
    def tearDownClass(cls):
        kill_child_process(cls.process.pid)

    def test_mmlu(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            num_examples=None,
            num_threads=2048,
        )

        metrics = run_eval(args)
        assert metrics["score"] >= 0.70

    def test_human_eval(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="humaneval",
            num_examples=None,
            num_threads=2048,
        )

        metrics = run_eval(args)
        assert metrics["score"] >= 0.65


if __name__ == "__main__":
    unittest.main()
