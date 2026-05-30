import os
import time
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.kits.kl_divergence_kit import KLDivergenceMixin
from sglang.test.kits.prefix_cache_branching_kit import PrefixCacheBranchingMixin
from sglang.test.server_fixtures.default_fixture import DefaultServerBase, openai_api_env
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

QWEN3_NEXT_MODEL = "Qwen/Qwen3-Next-80B-A3B-Instruct"


class TestQwen3Next(
    GSM8KMixin, KLDivergenceMixin, PrefixCacheBranchingMixin, DefaultServerBase
):
    model = QWEN3_NEXT_MODEL
    cache_chunk_size = 64
    gsm8k_accuracy_thres = 0.93
    kl_div_thres = 0.0025
    other_args = [
        "--tp-size",
        "4",
        "--chunked-prefill-size",
        "1024",
        "--mamba-scheduler-strategy",
        "extra_buffer",
        "--mamba-track-interval",
        "64",
        "--moe-runner-backend",
        "triton",
    ]


class TestQwen3NextLazyExtraBuffer(
    GSM8KMixin, KLDivergenceMixin, PrefixCacheBranchingMixin, CustomTestCase
):
    model = QWEN3_NEXT_MODEL
    cache_chunk_size = 64
    base_url = DEFAULT_URL_FOR_TEST
    timeout = DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH
    api_key = "sk-123456"
    gsm8k_accuracy_thres = 0.93
    kl_div_thres = 0.0025
    other_args = [
        "--tp-size",
        "4",
        "--chunked-prefill-size",
        "1024",
        "--mamba-scheduler-strategy",
        "extra_buffer",
        "--mamba-track-interval",
        "64",
        "--moe-runner-backend",
        "triton",
    ]

    @classmethod
    def setUpClass(cls):
        env = os.environ.copy()
        env["SGLANG_MAMBA_LAZY_EXTRA_BUFFER"] = "true"
        with openai_api_env(cls.api_key):
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=cls.timeout,
                other_args=cls.other_args,
                env=env,
            )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid, wait_timeout=60)
        time.sleep(2)

    @classmethod
    def flush_cache(cls):
        requests.post(cls.base_url + "/flush_cache")


if __name__ == "__main__":
    unittest.main()
