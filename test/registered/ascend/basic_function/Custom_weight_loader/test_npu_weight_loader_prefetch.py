import os
import tempfile
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    QWEN3_8B_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=300, suite="full-4-npu-a3", nightly=True)

# Common ascend launch args — --attention-backend ascend is a fixture
_COMMON_ARGS = [
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--dtype",
    "bfloat16",
    "--trust-remote-code",
    "--mem-fraction-static",
    "0.78",
]


class TestWeightLoaderPrefetchTp4(CustomTestCase):
    """--weight-loader-prefetch-checkpoints with tp-size=4 — verify prefetch
    log across 4 ranks and custom thread count.

    [Test Category] Parameter
    [Test Target] --weight-loader-prefetch-checkpoints, --weight-loader-prefetch-num-threads
    """

    model = QWEN3_8B_WEIGHTS_PATH
    base_url = DEFAULT_URL_FOR_TEST

    @classmethod
    def setUpClass(cls):
        cls.out_file = tempfile.NamedTemporaryFile(
            mode="w+", suffix=".txt", delete=False
        )
        cls.err_file = tempfile.NamedTemporaryFile(
            mode="w+", suffix=".txt", delete=False
        )
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                *_COMMON_ARGS,
                "--tp-size",
                "4",  # 4 ranks split prefetch across shards
                "--weight-loader-prefetch-checkpoints",
                "--weight-loader-prefetch-num-threads",
                "3",  # Custom thread count, matches 2.log pattern
            ],
            return_stdout_stderr=(cls.out_file, cls.err_file),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        cls.out_file.close()
        cls.err_file.close()
        os.unlink(cls.out_file.name)
        os.unlink(cls.err_file.name)

    def test_prefetch_log_4ranks_and_generate(self):
        """Verify prefetch log with per-rank shard count and custom threads."""
        with open(self.err_file.name) as f:
            log_content = f.read()

        # Each rank announces its shard assignment with this exact format
        self.assertIn(
            "prefetching",
            log_content,
            "Prefetch log keyword not found in stderr",
        )
        self.assertIn(
            "4 local ranks sharing the work, 3 threads per rank",
            log_content,
            "Expected '4 local ranks sharing the work, 3 threads per rank' not found",
        )
        # Rank 0 gets 2/5 shards (5 files / 4 ranks = extra 1), others get 1/5
        self.assertIn(
            "Rank 0: prefetching 2/5 checkpoint shards",
            log_content,
            "Rank 0 shard assignment (2/5) not found",
        )
        self.assertIn(
            "Rank 1: prefetching 1/5 checkpoint shards",
            log_content,
            "Rank 1 shard assignment (1/5) not found",
        )
        # Prefetch percentage progress: confirms the process ran to completion
        self.assertIn(
            "Rank 0: prefetching checkpoint files: 100% (2/2)",
            log_content,
            "Rank 0 prefetch completion not found",
        )
        self.assertIn(
            "Rank 1: prefetching checkpoint files: 100% (1/1)",
            log_content,
            "Rank 1 prefetch completion (1/1) not found",
        )

        # Server must be healthy and able to generate
        resp = requests.get(self.base_url + "/health", timeout=30)
        self.assertEqual(resp.status_code, 200)

        data = {
            "text": "Hello, my name is",
            "sampling_params": {"temperature": 0, "max_new_tokens": 8},
        }
        resp = requests.post(self.base_url + "/generate", json=data, timeout=30)
        self.assertEqual(resp.status_code, 200)
        self.assertIn("text", resp.json())


if __name__ == "__main__":
    unittest.main()
