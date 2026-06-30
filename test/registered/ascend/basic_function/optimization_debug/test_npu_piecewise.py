import unittest

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ascend.test_ascend_utils import LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(
    est_time=400,
    suite="nightly-1-npu-a3",
    nightly=True,
)


class TestPiecewise(GSM8KAscendMixin, CustomTestCase):
    """Test Case: Verify that the Llama-3.1-8B-Instruct model maintains normal accuracy
    when enabling the parameters: --torch-compile-max-bs, --piecewise-cuda-graph-max-tokens,
    --piecewise-cuda-graph-tokens, --enforce-piecewise-cuda-graph, --piecewise-cuda-graph-compiler

    [Test Category] Parameter
    [Test Target] --torch-compile-max-bs; --piecewise-cuda-graph-max-tokens; --piecewise-cuda-graph-tokens; --enforce-piecewise-cuda-graph; --piecewise-cuda-graph-compiler;
    """

    model = LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH
    accuracy = 0.65
    other_args = [
        "--trust-remote-code",
        "--torch-compile-max-bs",
        "2",
        "--cuda-graph-bs",
        "16",
        "--mem-fraction-static",
        0.8,
        "--piecewise-cuda-graph-max-tokens",
        "128",
        "--piecewise-cuda-graph-tokens",
        "64",
        "128",
        "--enforce-piecewise-cuda-graph",
        "--tp",
        "1",
        "--piecewise-cuda-graph-compiler",
        "eager",
        "--disable-radix-cache",
        "--attention-backend",
        "ascend",
    ]


if __name__ == "__main__":
    unittest.main()
