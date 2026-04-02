import unittest
from types import SimpleNamespace
from urllib.parse import urlparse

from sglang.srt.utils import get_device_sm, kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)

# FlashAttention4 integration test (requires SM 100+ / Blackwell B200)
register_cuda_ci(est_time=200, suite="stage-b-test-4-gpu-b200")


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
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=1319,
            num_threads=200,
        )
        metrics = run_eval(args)
        print(metrics)

        self.assertGreater(metrics["score"], 0.89)


if __name__ == "__main__":
    unittest.main()
