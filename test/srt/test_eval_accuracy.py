import json
import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_child_process
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import popen_launch_server


class TestAccuracy(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        port = 30000

        cls.model = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        cls.base_url = f"http://localhost:{port}"
        cls.process = popen_launch_server(cls.model, port, timeout=300)

    @classmethod
    def tearDownClass(cls):
        kill_child_process(cls.process.pid)

    def test_mmlu(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            num_examples=20,
            num_threads=20,
        )

        metrics = run_eval(args)
        assert metrics["score"] >= 0.5


if __name__ == "__main__":
    unittest.main(warnings="ignore")

    # t = TestAccuracy()
    # t.setUpClass()
    # t.test_mmlu()
    # t.tearDownClass()
