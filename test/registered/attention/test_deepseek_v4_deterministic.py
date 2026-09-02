"""
Usage:
cd test/registered/attention
python3 -m unittest test_deepseek_v4_deterministic.TestDeepseekV4Deterministic
"""

import unittest
from urllib.parse import urlparse

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_deterministic import BenchArgs, test_deterministic
from sglang.test.test_deterministic_utils import (
    COMMON_SERVER_ARGS,
    TestDeterministicBase,
)
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_cuda_ci(est_time=2400, stage="weekly", runner_config="4-gpu-h100")

DEEPSEEK_V4_MODEL = "sgl-project/DeepSeek-V4-Flash-FP8"
DEEPSEEK_V4_LAUNCH_TIMEOUT = 3600


class TestDeepseekV4Deterministic(TestDeterministicBase):
    """DeepSeek-V4 on its own dsv4 attention backend.

    The backend is forced by the model's config-time overrides, so this covers
    the one attention backend DeepSeek-V4 can run on rather than a choice
    between several.
    """

    @classmethod
    def get_model(cls):
        return DEEPSEEK_V4_MODEL

    @classmethod
    def get_launch_timeout(cls):
        return DEEPSEEK_V4_LAUNCH_TIMEOUT

    @classmethod
    def get_server_args(cls):
        return COMMON_SERVER_ARGS + ["--tp", "4"]


class TestDeepseekV4SameKernelPD(CustomTestCase):
    """Exact prefill-vs-decode parity with both modes on FlashMLA KV-cache.

    Exact selected-token logprob equality implies zero K3 KL for every token.
    This isolates KV-cache state and metadata correctness from numerical drift
    between the production sparse-prefill and KV-cache attention kernels.
    """

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            DEEPSEEK_V4_MODEL,
            cls.base_url,
            timeout=DEEPSEEK_V4_LAUNCH_TIMEOUT,
            other_args=[
                "--trust-remote-code",
                "--cuda-graph-max-bs-decode",
                "4",
                "--enable-deterministic-inference",
                "--tp",
                "4",
                "--max-running-requests",
                "4",
                "--dsv4-prefill-backend",
                "flashmla_kv",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        terminate_and_kill_process_tree(cls.process, wait_timeout=60)

    def test_prefill_decode_logprobs_are_bitexact(self):
        url = urlparse(self.base_url)
        args = BenchArgs(
            host=url.hostname,
            port=url.port,
            test_mode="p_vs_d",
            n_trials=4,
            max_new_tokens=100,
        )
        results = test_deterministic(args)
        self.assertTrue(all(result == 1 for result in results), results)
        print("Same-kernel P/D logprobs are bit-exact; K3 KL=0.")


if __name__ == "__main__":
    unittest.main()
