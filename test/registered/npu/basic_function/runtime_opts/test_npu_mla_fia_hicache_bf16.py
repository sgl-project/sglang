import os
import unittest

from sglang.test.ascend.e2e.test_npu_performance_utils import (
    MOONLIGHT_16B_A3B_MODEL_PATH,
)
from sglang.test.ascend.npu_eval_accuracy_kit import NPUGSM8KMixin
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
register_npu_ci(est_time=1400, suite="nightly-2-npu-a3", nightly=True)

# The nightly pipeline is also triggered by pull requests via path filters, so the
# pipeline is identified by GITHUB_WORKFLOW_REF (inherited from the caller's github
# context) instead of the triggering event.
_is_pr_pipeline = "/.github/workflows/nightly-test-npu.yml" not in os.environ.get(
    "GITHUB_WORKFLOW_REF", ""
)


class TestAscendMlaFiaHicacheBf16(CustomTestCase, NPUGSM8KMixin):
    # The nightly pipeline runs the full GSM8K set to avoid sampling variance; the
    # PR pipeline only runs the mixin smoke request.
    gsm8k_num_examples = 1319
    # evalscope is the harness the 0.83 Moonlight bf16 baseline was calibrated
    # with (4-shot chat, temperature 1.0).
    gsm8k_accuracy_thres = 0.83
    gsm8k_backend = "evalscope"

    @classmethod
    def setUpClass(cls):
        os.environ["ASCEND_USE_FIA"] = "true"
        cls.model = MOONLIGHT_16B_A3B_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
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
            other_args.append("--disable-cuda-graph")
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            env={
                **os.environ,
                "PYTORCH_NPU_ALLOC_CONF": (
                    "expandable_segments:True,pin_memory_expandable_segments:True"
                ),
            },
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        terminate_and_kill_process_tree(cls.process)


if __name__ == "__main__":
    unittest.main()
