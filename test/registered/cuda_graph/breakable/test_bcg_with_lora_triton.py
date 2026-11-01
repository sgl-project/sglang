"""Extra legacy Triton breakable-prefill checks; csgmv runs per PR."""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.bcg_lora_kit import BCGLoRAServerMixin
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=150, stage="extra-a", runner_config="1-gpu-large")


class TestBCGLoRATriton(BCGLoRAServerMixin, CustomTestCase):
    lora_backend = "triton"


if __name__ == "__main__":
    unittest.main()
