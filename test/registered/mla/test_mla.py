import os
import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import MGSMEnMixin
from sglang.test.test_utils import (
    DEFAULT_MLA_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
)

# MLA attention test with MGSM evaluation
register_cuda_ci(est_time=181, suite="stage-b-test-1-gpu-large")
register_amd_ci(est_time=1100, suite="stage-b-test-1-gpu-small-amd")

_CUDA_PR_UT_EVENTS = ("pull_request", "workflow_dispatch")


@unittest.skipIf(
    is_in_ci() and os.getenv("GITHUB_EVENT_NAME") in _CUDA_PR_UT_EVENTS,
    "MLA MGSM accuracy is unstable on current CUDA PR UT H100 runners",
)
class TestMLA(CustomTestCase, MGSMEnMixin):
    mgsm_en_score_threshold = 0.8

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MLA_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--enable-torch-compile",
                "--torch-compile-max-bs",
                "4",
                "--chunked-prefill-size",
                "256",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


if __name__ == "__main__":
    unittest.main()
