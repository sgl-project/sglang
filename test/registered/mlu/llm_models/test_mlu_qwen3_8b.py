import unittest

import torch_mlu  # noqa: F401

from sglang.test.ci.ci_register import register_mlu_ci
from sglang.test.mlu.gsm8k_mlu_mixin import GSM8KMLUMixin
from sglang.test.mlu.test_mlu_utils import QWEN3_8B_WEIGHTS_PATH
from sglang.test.test_utils import CustomTestCase

register_mlu_ci(est_time=900, suite="nightly-test-mlu", nightly=True)


class TestMLUQwen38B(GSM8KMLUMixin, CustomTestCase):
    """Verify Qwen3-8B GSM8K accuracy on the MLU backend."""

    model = QWEN3_8B_WEIGHTS_PATH
    accuracy = 0.80
    random_seed = 42
    temperature = 0.6
    top_p = 0.95
    other_args = [
        "--device",
        "mlu",
        "--trust-remote-code",
        "--attention-backend",
        "mlu",
        "--sampling-backend",
        "pytorch",
        "--skip-server-warmup",
    ]


if __name__ == "__main__":
    unittest.main()
