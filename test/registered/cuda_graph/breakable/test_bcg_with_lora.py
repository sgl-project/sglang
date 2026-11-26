"""Nightly csgmv breakable-prefill checks; the legacy Triton arm is
test_bcg_with_lora_triton.py."""

import unittest

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("Test requires CUDA", allow_module_level=True)


from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.bcg_lora_kit import BCGLoRAServerMixin
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=90, stage="nightly", runner_config="1-gpu-small")


class TestBCGLoRACsgmv(BCGLoRAServerMixin, CustomTestCase):
    lora_backend = "csgmv"


if __name__ == "__main__":
    unittest.main()
