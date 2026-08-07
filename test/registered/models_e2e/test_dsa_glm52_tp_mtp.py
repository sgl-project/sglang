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
    stage="extra-b",
    runner_config="8-gpu-h200",
)


class TestGLM52TPMTP(
    DsaMtpServerBase, DsaMtpEvalConfigDefaults, GSM8KMixin, SpecDecodingMixin
):
    """TP-only counterpart of the per-commit DP-attention launch in
    test_dsa_glm52_dp_mtp.py. Label-gated: the two differ only by
    --enable-dp-attention, and DSA + MTP is deployed with DP attention on."""

    model = "zai-org/GLM-5.2-FP8"
    mem_fraction_static = 0.8
    bs_1_speed_thres = 150


if __name__ == "__main__":
    unittest.main()
