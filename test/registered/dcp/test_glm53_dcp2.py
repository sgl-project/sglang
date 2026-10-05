import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.server_fixtures.dsa_mtp_fixture import (
    DsaMtpEvalConfigDefaults,
    DsaMtpServerBase,
)

register_cuda_ci(est_time=600, stage="extra-b", runner_config="4-gpu-b200")


class TestGLM53NVFP4DCP2MTP(DsaMtpServerBase, DsaMtpEvalConfigDefaults, GSM8KMixin):
    model = "nvidia/GLM-5.3-NVFP4"
    tp_size = 4
    mem_fraction_static = 0.8
    # TP4 without DCP scores 0.914 on this eval; TP4 + DCP2 scores 0.904.
    gsm8k_accuracy_thres = 0.88
    extra_server_args = ["--quantization", "modelopt_fp4", "--dcp-size", "2"]


if __name__ == "__main__":
    unittest.main()
