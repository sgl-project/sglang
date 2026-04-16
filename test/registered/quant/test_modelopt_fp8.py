import unittest
from types import SimpleNamespace
from urllib.parse import urlparse

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.few_shot_gsm8k import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=120, suite="stage-b-test-large-1-gpu")


class TestModeloptFP8(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = "nvidia/Llama-3.1-8B-Instruct-FP8"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--quantization", "modelopt_fp8"],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        parsed_url = urlparse(self.base_url)
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=200,
            host=f"{parsed_url.scheme}://{parsed_url.hostname}",
            port=parsed_url.port,
        )
        metrics = run_eval(args)
        print(f"{metrics=}")
        self.assertGreater(metrics["accuracy"], 0.70)


if __name__ == "__main__":
    unittest.main()
