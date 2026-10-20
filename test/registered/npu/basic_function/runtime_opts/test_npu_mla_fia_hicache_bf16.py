import os
import unittest
from urllib.parse import urlparse

from sglang.test.ascend.e2e.test_npu_accuracy_utils import run_evalscope
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    MOONLIGHT_16B_A3B_MODEL_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
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

GSM8K_NUM_QUESTIONS = 1319
# Match the nightly accuracy case so both numbers come from the same harness.
GSM8K_GENERATION_CONFIG = {"max_tokens": 7168, "temperature": 1.0}
GSM8K_EVAL_BATCH_SIZE = 64

# The PR pipeline evaluates the first 20% of GSM8K (263 questions).
PR_GSM8K_RATIO = 0.2
PR_ACCURACY_THRESHOLD = 0.77

# The nightly pipeline is also triggered by pull requests via path filters, so the
# pipeline is identified by GITHUB_WORKFLOW_REF (inherited from the caller's github
# context) instead of the triggering event.
_is_pr_pipeline = "/.github/workflows/nightly-test-npu.yml" not in os.environ.get(
    "GITHUB_WORKFLOW_REF", ""
)


class TestAscendMlaFiaHicacheBf16(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.models = TEST_MODEL_MATRIX.keys()
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.url = urlparse(DEFAULT_URL_FOR_TEST)
        cls.fia_hicache_args = [
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
            2 if _is_pr_pipeline else 10,
        ]
        # The PR pipeline runs eager to shorten startup; the nightly pipeline
        # covers the default decode graph path with FIA + HiCache.
        if _is_pr_pipeline:
            cls.fia_hicache_args.append("--disable-cuda-graph")

    def test_gsm8k_fia_hicache(self):
        os.environ["ASCEND_USE_FIA"] = "true"
        for model in self.models:
            with self.subTest(model=model):
                process = popen_launch_server(
                    model,
                    self.base_url,
                    timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                    other_args=[
                        *self.fia_hicache_args,
                    ],
                )

                try:
                    num_questions = GSM8K_NUM_QUESTIONS
                    accuracy_thres = TEST_MODEL_MATRIX[model]["accuracy"]
                    if _is_pr_pipeline:
                        num_questions = int(GSM8K_NUM_QUESTIONS * PR_GSM8K_RATIO)
                        accuracy_thres = PR_ACCURACY_THRESHOLD
                    print(
                        f"##=== Testing accuracy: {model} "
                        f"({num_questions}/{GSM8K_NUM_QUESTIONS} questions) ===##"
                    )

                    metrics = run_evalscope(
                        host=self.url.hostname,
                        port=int(self.url.port),
                        model=os.path.basename(model),
                        datasets=["gsm8k"],
                        eval_batch_size=GSM8K_EVAL_BATCH_SIZE,
                        limit=num_questions,
                        generation_config=GSM8K_GENERATION_CONFIG,
                    )
                    self.assertIn(
                        "accuracy", metrics, "Failed to parse evalscope accuracy"
                    )
                    self.assertGreaterEqual(metrics["accuracy"], accuracy_thres)
                finally:
                    terminate_and_kill_process_tree(process)


if __name__ == "__main__":
    unittest.main()
