import os
import unittest
from types import SimpleNamespace
from urllib.parse import urlparse

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    MOONLIGHT_16B_A3B_MODEL_PATH,
)
from sglang.test.ascend.npu_eval_accuracy_kit import run_npu_pr_smoke
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_npu_ci(
    est_time=800,
    suite="base-b-test-2-npu-a3",
)
register_npu_ci(est_time=800, suite="nightly-2-npu-a3", nightly=True)

TEST_MODEL_MATRIX = {
    MOONLIGHT_16B_A3B_MODEL_PATH: {
        "accuracy": 0.83,
    },
}

# The nightly pipeline is also triggered by pull requests via path filters, so the
# pipeline is identified by GITHUB_WORKFLOW_REF (inherited from the caller's github
# context) instead of the triggering event.
_is_pr_pipeline = (
    "/.github/workflows/nightly-test-npu.yml"
    not in os.environ.get("GITHUB_WORKFLOW_REF", "")
)


class TestAscendMlaFiaHicacheBf16(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.models = TEST_MODEL_MATRIX.keys()
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.url = urlparse(DEFAULT_URL_FOR_TEST)
        cls.fia_args = [
            "--trust-remote-code",
            "--disable-cuda-graph",
            "--mem-fraction-static",
            0.8,
            "--attention-backend",
            "ascend",
            "--device",
            "npu",
            "--dtype",
            "bfloat16",
            "--tp-size",
            2,
            "--disable-radix-cache",
        ]
        # HiCache is mutually exclusive with --disable-radix-cache.
        cls.hicache_args = [
            "--trust-remote-code",
            "--mem-fraction-static",
            0.8,
            "--attention-backend",
            "ascend",
            "--device",
            "npu",
            "--dtype",
            "bfloat16",
            "--tp-size",
            2,
            "--enable-hierarchical-cache",
            "--hicache-size",
            30,
        ]

    def test_a_gsm8k(self):
        os.environ["ASCEND_USE_FIA"] = "true"
        for model in self.models:
            with self.subTest(model=model):
                process = popen_launch_server(
                    model,
                    self.base_url,
                    timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                    other_args=[
                        *self.fia_args,
                    ],
                )

                try:
                    if _is_pr_pipeline:
                        run_npu_pr_smoke(self.base_url)
                    else:
                        print(f"##=== Testing accuracy: {model} ===##")

                        args = SimpleNamespace(
                            num_shots=5,
                            data_path=None,
                            num_questions=1319,
                            max_new_tokens=512,
                            parallel=128,
                            host=self.url.hostname,
                            port=int(self.url.port),
                        )

                        metrics = run_eval_few_shot_gsm8k(args)
                        self.assertGreaterEqual(
                            metrics["accuracy"],
                            TEST_MODEL_MATRIX[model]["accuracy"],
                        )
                finally:
                    kill_process_tree(process.pid)

    def test_b_gsm8k_hicache(self):
        os.environ.pop("ASCEND_USE_FIA", None)
        for model in self.models:
            with self.subTest(model=model):
                process = popen_launch_server(
                    model,
                    self.base_url,
                    timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                    other_args=[
                        *self.hicache_args,
                    ],
                )

                try:
                    if _is_pr_pipeline:
                        run_npu_pr_smoke(self.base_url)
                    else:
                        print(f"##=== Testing accuracy with HiCache: {model} ===##")

                        args = SimpleNamespace(
                            num_shots=5,
                            data_path=None,
                            num_questions=1319,
                            max_new_tokens=512,
                            parallel=128,
                            host=self.url.hostname,
                            port=int(self.url.port),
                        )

                        metrics = run_eval_few_shot_gsm8k(args)
                        self.assertGreaterEqual(
                            metrics["accuracy"],
                            TEST_MODEL_MATRIX[model]["accuracy"],
                        )
                finally:
                    terminate_and_kill_process_tree(process)


if __name__ == "__main__":
    unittest.main()
