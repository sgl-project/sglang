import unittest

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
    runner_config="4-gpu-b200",
)


class TestGLM52NVFP4TPMTP(
    DsaMtpServerBase, DsaMtpEvalConfigDefaults, GSM8KMixin, SpecDecodingMixin
):
    model = "nvidia/GLM-5.2-NVFP4"
    tp_size = 4
    mem_fraction_static = 0.8
    bs_1_speed_thres = 280
    extra_server_args = [
        "--moe-runner-backend",
        "flashinfer_trtllm",
        "--quantization",
        "modelopt_fp4",
    ]


if __name__ == "__main__":
    unittest.main()
