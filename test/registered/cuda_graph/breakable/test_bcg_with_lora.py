"""Per-PR csgmv breakable-prefill checks; legacy Triton runs in Extra."""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.bcg_lora_kit import BCGLoRAServerMixin
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=90, stage="base-b", runner_config="1-gpu-large")


class TestBCGLoRACsgmv(BCGLoRAServerMixin, CustomTestCase):
    lora_backend = "csgmv"


if __name__ == "__main__":
    unittest.main()
