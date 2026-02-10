import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.few_shot_gsm8k import run_eval as run_gsm8k
from sglang.test.ascend.test_ascend_utils import QWEN3_30B_A3B_INSTRUCT_2507_WEIGHTS_PATH
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-2-npu-a3", nightly=True)


class TestHiCache(CustomTestCase):
    """
    Testcase：Verify the correctness of --enable-hierarchical-cache (HiCache) and  dataset accuracy (gsm8k) meets the
    requirement.

    [Test Category] Parameter
    [Test Target] --enable-hierarchical-cache, --hicache-size
    """

    @classmethod
    def setUpClass(cls):
        # Test class initialization: Launch the service with HiCache enabled and related NPU/HIP configurations
        cls.model = QWEN3_30B_A3B_INSTRUCT_2507_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
                "--enable-hierarchical-cache",
                "--hicache-size",
                100,
                "--tp-size",
                2,
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        # Verify gsm8k dataset evaluation accuracy meets the minimum requirement (score ≥ 0.9) with HiCache enabled
        expect_accuracy = 0.9
        args = SimpleNamespace(
            num_shots=8,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        print("Starting gsm8k test...")
        metrics = run_gsm8k(args)
        self.assertGreaterEqual(
            metrics["accuracy"],
            expect_accuracy,
            f'Accuracy of {self.model} is {str(metrics["accuracy"])}, is lower than {expect_accuracy}',
        )


if __name__ == "__main__":
    unittest.main()
