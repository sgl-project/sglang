import unittest
from urllib.parse import urlparse

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_deterministic import (
    BenchArgs,
)
from sglang.test.test_deterministic import test_deterministic as run_deterministic
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=180, stage="base-b", runner_config="2-gpu-large")

QWEN35_MODEL = "Qwen/Qwen3.5-9B"
SERVER_LAUNCH_TIMEOUT = 600


class TestQwen35EagleRSDeterministic(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--trust-remote-code",
            "--tp",
            "2",
            "--attention-backend",
            "triton",
            "--enable-deterministic-inference",
            "--speculative-algorithm",
            "NEXTN",
            "--speculative-num-steps",
            "3",
            "--speculative-eagle-topk",
            "1",
            "--speculative-num-draft-tokens",
            "4",
            "--speculative-use-rejection-sampling",
            "--mem-fraction-static",
            "0.8",
            "--disable-radix-cache",
        ]
        with envs.SGLANG_ENABLE_ASYNC_ASSERT.override(True):
            cls.process = popen_launch_server(
                QWEN35_MODEL,
                cls.base_url,
                timeout=SERVER_LAUNCH_TIMEOUT,
                other_args=other_args,
            )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_seeded_determinism(self):
        endpoint = urlparse(self.base_url)
        results = run_deterministic(
            BenchArgs(
                host=endpoint.hostname,
                port=endpoint.port,
                temperature=1.0,
                sampling_seed=42,
                max_new_tokens=32,
                test_mode="single",
                n_trials=6,
            )
        )
        self.assertEqual(results, [1])


if __name__ == "__main__":
    unittest.main()
