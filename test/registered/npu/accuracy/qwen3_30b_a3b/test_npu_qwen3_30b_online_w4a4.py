import unittest

from sglang.test.ascend.test_ascend_utils import (
    QWEN3_30B_A3B_INSTRUCT_2507_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_npu_ci(est_time=400, suite="full-2-npu-a3", nightly=True)


class TestQwen330BOnlineW4A4(GSM8KMixin, CustomTestCase):
    model = QWEN3_30B_A3B_INSTRUCT_2507_WEIGHTS_PATH
    gsm8k_backend = "run_eval"
    gsm8k_score_threshold = 0.75
    gsm8k_num_examples = 200
    gsm8k_num_shots = 5
    gsm8k_num_threads = 128
    gsm8k_temperature = 0

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--device",
                "npu",
                "--dtype",
                "float16",
                "--attention-backend",
                "ascend",
                "--tp-size",
                "2",
                "--mem-fraction-static",
                "0.7",
                "--max-running-requests",
                "32",
                "--online-quantization",
                "w4a4_int",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "process", None):
            terminate_and_kill_process_tree(cls.process)


if __name__ == "__main__":
    unittest.main()
