"""MI35x DeepSeek-V4-Pro FP4 weight-loading experiment.

This uses the FP4-expert Pro checkpoint, not the larger Pro FP8 checkpoint. It
starts one TP=8 server, records time to ``/health_generate``, and exits without
running accuracy or performance suites.

Registry: nightly-amd-8-gpu-mi35x-deepseek-v4-pro-fp4-load-exp
"""

import os
import time

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)

register_amd_ci(
    est_time=3600,
    suite="nightly-amd-8-gpu-mi35x-deepseek-v4-pro-fp4-load-exp",
    nightly=True,
)

MODEL_PATH = os.environ.get(
    "DEEPSEEK_V4_PRO_MODEL_PATH_FP4", "deepseek-ai/DeepSeek-V4-Pro"
)
EXPERIMENT = os.environ.get("SGLANG_DSV4_LOAD_EXPERIMENT", "baseline")
ENABLE_PREFETCH = os.environ.get("SGLANG_DSV4_LOAD_PREFETCH") == "1"
PREFETCH_THREADS = int(os.environ.get("SGLANG_DSV4_PREFETCH_THREADS", "2"))


class TestDeepseekV4ProFp4LoadOnly(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        env = os.environ.copy()
        env.update(
            {
                "AITER_BF16_FP8_MOE_BOUND": "0",
                "SGLANG_DP_USE_GATHERV": "1",
                "SGLANG_DSV4_FP4_EXPERTS": "true",
                "SGLANG_USE_ROCM700A": "0",
            }
        )

        other_args = [
            "--trust-remote-code",
            "--tp",
            "8",
            "--disable-cuda-graph",
            "--disable-radix-cache",
            "--attention-backend",
            "dsv4",
            "--max-running-requests",
            "16",
            "--page-size",
            "256",
            "--mem-fraction-static",
            "0.90",
            "--swa-full-tokens-ratio",
            "0.1",
            "--chunked-prefill-size",
            "8192",
            "--disable-shared-experts-fusion",
        ]
        if ENABLE_PREFETCH:
            other_args.extend(
                [
                    "--weight-loader-prefetch-checkpoints",
                    "--weight-loader-prefetch-num-threads",
                    str(PREFETCH_THREADS),
                ]
            )

        start = time.perf_counter()
        cls.process = popen_launch_server(
            MODEL_PATH,
            cls.base_url,
            timeout=3600,
            other_args=other_args,
            env=env,
        )
        cls.ready_elapsed = time.perf_counter() - start

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_server_ready(self):
        self.assertIsNone(self.process.poll())
        h2d_threads = os.environ.get("SGLANG_DSV4_WEIGHT_LOAD_THREADS", "0")
        sort_weight_files = os.environ.get("SGLANG_SORT_WEIGHT_FILES", "0")
        result = (
            f"experiment={EXPERIMENT} ready_elapsed={self.ready_elapsed:.2f}s "
            f"prefetch={int(ENABLE_PREFETCH)} "
            f"prefetch_threads={PREFETCH_THREADS} "
            f"h2d_threads={h2d_threads} sort_weight_files={sort_weight_files}"
        )
        print(f"DSV4_PRO_FP4_LOAD_RESULT {result}")
        if is_in_ci():
            write_github_step_summary(
                "### DeepSeek-V4-Pro FP4 load-only experiment\n" f"`{result}`\n"
            )


if __name__ == "__main__":
    import unittest

    unittest.main()
