"""The csgmv LoRA backend under the breakable (BCG) prefill CUDA graph.

Per-PR arm of sglang.test.kits.bcg_lora_kit (graph-vs-eager prompt logprobs
for LoRA, base and mixed batches, with the log-scraping guards against a
vacuous pass). The legacy triton arm is test_bcg_with_lora_triton.py (extra).
"""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.bcg_lora_kit import BCGLoRAServerMixin
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=90, stage="base-b", runner_config="1-gpu-large")


class TestBCGLoRACsgmv(BCGLoRAServerMixin, CustomTestCase):
    lora_backend = "csgmv"


if __name__ == "__main__":
    unittest.main()
