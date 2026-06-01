import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.kits.spec_decoding_kit import SpecDecodingMixin
from sglang.test.server_fixtures.dsa_mtp_fixture import (
    DsaMtpEvalConfigDefaults,
    DsaMtpServerBase,
)

register_cuda_ci(
    est_time=400,
    stage="base-c",
    runner_config="8-gpu-h200",
)


@unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
class TestGLM5TPMTP(
    DsaMtpServerBase, DsaMtpEvalConfigDefaults, GSM8KMixin, SpecDecodingMixin
):
    model = "zai-org/GLM-5-FP8"
    mem_fraction_static = 0.8
    bs_1_speed_thres = 150


if __name__ == "__main__":
    unittest.main()
