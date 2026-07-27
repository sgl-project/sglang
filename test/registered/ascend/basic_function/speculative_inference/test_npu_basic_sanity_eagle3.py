import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    QWEN3_8B_EAGLE3_WEIGHTS_PATH,
    QWEN3_8B_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.kits.basic_api_contract_kit import BasicAPIContractMixin
from sglang.test.kits.basic_decode_correctness_kit import BasicDecodeCorrectnessMixin
from sglang.test.kits.basic_scheduler_stress_kit import BasicSchedulerStressMixin
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=200, suite="full-1-npu-a3", nightly=True)


class TestBasicSanityEagle3(
    BasicAPIContractMixin,
    BasicDecodeCorrectnessMixin,
    BasicSchedulerStressMixin,
    GSM8KMixin,
    CustomTestCase,
):
    served_model_name = QWEN3_8B_WEIGHTS_PATH

    model = QWEN3_8B_WEIGHTS_PATH
    gsm8k_num_questions = 1400
    gsm8k_accuracy_thres = 0.74

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            QWEN3_8B_WEIGHTS_PATH,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--dtype",
                "float16",
                "--attention-backend",
                "ascend",
                "--speculative-algorithm",
                "EAGLE3",
                "--speculative-draft-model-path",
                QWEN3_8B_EAGLE3_WEIGHTS_PATH,
                "--speculative-num-steps",
                "1",
                "--speculative-eagle-topk",
                "1",
                "--speculative-num-draft-tokens",
                "2",
                "--cuda-graph-max-bs",
                "4",
                "--mem-fraction-static",
                "0.65",
                "--disable-piecewise-cuda-graph",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


if __name__ == "__main__":
    unittest.main()
