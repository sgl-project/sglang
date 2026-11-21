import os
import unittest

from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.server_fixtures.dsa_mtp_fixture import DsaMtpServerBase

register_amd_ci(est_time=400, suite="base-c-test-large-8-gpu-amd-mi35x")

GLM52_MODEL_PATH = os.environ.get("GLM52_MODEL_PATH", "amd/GLM-5.2-MXFP4")


class TestGLM52MXFP4DCP4MTPBreakablePrefill(DsaMtpServerBase, GSM8KMixin):
    """GLM-5.2 TP4 + DCP4 + MTP on ROCm with breakable prefill graphs: the DCP
    MLA attention returns (output, LSE), and the prefill graph wrapper must
    carry the pair through extend and target-verify."""

    model = GLM52_MODEL_PATH
    tp_size = 4
    mem_fraction_static = 0.8
    gsm8k_accuracy_thres = 0.9
    extra_server_args = [
        "--cuda-graph-backend-prefill",
        "breakable",
        "--kv-cache-dtype",
        "fp8_e4m3",
        "--dsa-prefill-backend",
        "triton",
        "--dsa-decode-backend",
        "triton",
        "--dcp-size",
        "4",
        "--random-seed",
        "0",
    ]


if __name__ == "__main__":
    unittest.main()
