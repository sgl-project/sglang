import unittest
from types import SimpleNamespace
from urllib.parse import urlparse

from sglang.srt.utils import get_device_sm, kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)


@unittest.skipIf(get_device_sm() < 100, "Test requires CUDA SM 100 or higher")
class TestFlashAttention4(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = "Qwen/Qwen3-8B"
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--trust-remote-code",
            "--prefill-attention-backend",
            "fa4",
            "--decode-attention-backend",
            "flashinfer",
        ]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        parsed_url = urlparse(self.base_url)
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=1319,
            max_new_tokens=512,
            parallel=200,
            host=f"{parsed_url.scheme}://{parsed_url.hostname}",
            port=parsed_url.port,
        )
        metrics = run_eval_few_shot_gsm8k(args)
        print(metrics)

        self.assertGreater(metrics["accuracy"], 0.89)


if __name__ == "__main__":
    unittest.main()
