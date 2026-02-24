import unittest

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ascend.test_ascend_utils import KIMI_LINEAR_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    CustomTestCase,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)

register_npu_ci(
    est_time=400,
    suite="nightly-1-npu-a3",
    nightly=True,
    disabled="run failed",
)



class TestKimiLinear(GSM8KAscendMixin, CustomTestCase):
    """Testcase: Verify that the inference accuracy of the moonshotai/Kimi-Linear-48B-A3B-Instruct model on the GSM8K dataset is no less than 0.88.

    [Test Category] Model
    [Test Target] moonshotai/Kimi-Linear-48B-A3B-Instruct
    """

    model = KIMI_LINEAR_WEIGHTS_PATH
    accuracy = 0.88
    other_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.8",
        "--attention-backend",
        "ascend",
        "--tp-size",
        "2",
        "--disable-cuda-graph",
        "--disable-radix-cache",
        "--max-running-requests",
        "16",
    ]

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        os.environ["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:False"
        os.environ["ASCEND_MF_STORE_URL"] = "tcp://127.0.0.1:24666"
        os.environ["HCCL_BUFFSIZE"] = "200"
        os.environ["SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK"] = "24"
        os.environ["USE_VLLM_CUSTOM_ALLREDUCE"] = "1"
        os.environ["HCCL_EXEC_TIMEOUT"] = "200"
        os.environ["STREAMS_PER_DEVICE"] = "32"
        os.environ["SGLANG_ENBLE_TORCH_COMILE"] = "1"
        os.environ["AUTO_USE_UC_MEMORY"] = "0"
        os.environ["P2P_HCCL_BUFFSIZE"] = "20"
        env = os.environ.copy()

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=cls.timeout_for_server_launch,
            other_args=cls.other_args,
            env=env,
        )


if __name__ == "__main__":
    unittest.main()
