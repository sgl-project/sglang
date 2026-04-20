from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=99, suite="stage-b-test-1-gpu-small")
register_amd_ci(est_time=300, suite="stage-b-test-1-gpu-small-amd")

"""
# TODO: Segmentation fault occurs when upgraded to Cu13. Ref: https://github.com/sgl-project/sglang/actions/runs/24603159715/job/71945537414?pr=23119")
# Should move back to registered test after it's fixed
"""

import time
import unittest

from sglang.srt.utils import is_hip, kill_process_tree
from sglang.test.kits.eval_accuracy_kit import MMLUMixin
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

_is_hip = is_hip()


class TestHiCache(CustomTestCase, MMLUMixin):
    mmlu_score_threshold = 0.65
    mmlu_num_examples = 64
    mmlu_num_threads = 32

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--enable-hierarchical-cache",
                "--mem-fraction-static",
                0.7,
                "--hicache-size",
                100 if not _is_hip else 200,
                "--page-size",
                "64",
                "--hicache-storage-backend",
                "file",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        time.sleep(5)


if __name__ == "__main__":
    unittest.main()
