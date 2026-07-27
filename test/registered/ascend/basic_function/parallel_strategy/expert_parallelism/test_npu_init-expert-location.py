import unittest

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ascend.test_ascend_utils import (
    HOT_MAP_JSON,
    HOT_MAP_PT,
    HOT_MAP_STRING,
    QWEN3_30B_A3B_INSTRUCT_2507_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=200, suite="full-2-npu-a3", nightly=True)

# Read the content of the expert distribution file
with open(HOT_MAP_STRING, "r") as f:
    init_expert_location = f.read()


class TestInitExpertLocationString(GSM8KAscendMixin, CustomTestCase):
    """Testcase: Verify set --init-expert-location the inference accuracy of the model on the
    GSM8K dataset is no less than 0.90.

    [Test Category] Parameters
    [Test Target] --init-expert-location
    """

    init_expert_location = init_expert_location

    model = QWEN3_30B_A3B_INSTRUCT_2507_WEIGHTS_PATH
    accuracy = 0.90
    other_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        0.7,
        "--max-running-requests",
        32,
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--cuda-graph-max-bs",
        32,
        "--tp-size",
        2,
        "--ep-dispatch-algorithm",
        "static",
        "--moe-a2a-backend",
        "deepep",
        "--deepep-mode",
        "normal",
        "--init-expert-location",
        init_expert_location,
    ]

    env = {
        "HCCL_BUFFSIZE": "1024",
    }


class TestInitExpertLocationJson(TestInitExpertLocationString):
    """test json format"""

    init_expert_location = HOT_MAP_JSON


class TestInitExpertLocationPt(TestInitExpertLocationString):
    """test pt format"""

    init_expert_location = HOT_MAP_PT


class TestInitExpertLocationTrivial(TestInitExpertLocationString):
    init_expert_location = "trivial"


if __name__ == "__main__":
    unittest.main()
