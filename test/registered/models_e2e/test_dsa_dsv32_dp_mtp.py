import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.kits.spec_decoding_kit import SpecDecodingMixin
from sglang.test.server_fixtures.dsa_mtp_fixture import (
    DsaMtpEvalConfigDefaults,
    DsaMtpServerBase,
)

register_cuda_ci(
    est_time=600,
    stage="extra-b",
    runner_config="8-gpu-h200",
)


class TestDeepseekV32DPMTP(
    DsaMtpServerBase, DsaMtpEvalConfigDefaults, GSM8KMixin, SpecDecodingMixin
):
    model = "deepseek-ai/DeepSeek-V3.2"
    mem_fraction_static = 0.7
    enable_dp_attention = True
    bs_1_speed_thres = 90


if __name__ == "__main__":
    unittest.main()
