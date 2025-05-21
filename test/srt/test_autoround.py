import unittest
from types import SimpleNamespace
# import sys
# import unittest
# import sglang
# sys.path.insert(0, "../../python/sglang")
from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_AUTOROUND_MLLM_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestAutoRound(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = None
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--trust-remote-code"],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_mmlu(self):
        for model in DEFAULT_AUTOROUND_MODEL_NAME_FOR_TEST:
            self.model = model
            args = SimpleNamespace(
                base_url=self.base_url,
                model=self.model,
                eval_name="lambada_openai",
                num_examples=64,
                num_threads=32,
            )

            metrics = run_eval(args)
            self.assertGreater(metrics["score"], 0.4)


if __name__ == "__main__":
    unittest.main()
