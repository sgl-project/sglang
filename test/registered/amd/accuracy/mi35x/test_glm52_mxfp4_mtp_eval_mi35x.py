"""MI35x GLM-5.2-MXFP4 EAGLE-MTP GSM8K evaluation (8-GPU nightly).

AMD-nightly sibling of test/registered/models_e2e/test_dsa_glm52_tp_mtp.py:
the same DSA + EAGLE-MTP eval fixture, run against the AMD Quark-MXFP4
checkpoint on MI35x. This is the GPU coverage of the GLM-5.2 Quark-MXFP4 MTP
load path requested on #30265 -- the bf16 nextn layer must load without shape
mismatch under Quark MXFP4, and MTP must still accept drafts (accept length is
asserted by GSM8KMixin via gsm8k_accept_length_thres).

Registry: nightly-amd-8-gpu-mi35x-glm52-mxfp4-mtp suite
"""

import unittest

from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.server_fixtures.dsa_mtp_fixture import (
    DsaMtpEvalConfigDefaults,
    DsaMtpServerBase,
)

register_amd_ci(
    est_time=5400,
    suite="nightly-amd-8-gpu-mi35x-glm52-mxfp4-mtp",
    nightly=True,
)


class TestGLM52MXFP4MTPMI35x(DsaMtpServerBase, DsaMtpEvalConfigDefaults, GSM8KMixin):
    model = "amd/GLM-5.2-MXFP4"
    mem_fraction_static = 0.8
    # Thresholds from the reported amd/GLM-5.2-MXFP4 runs (MI355x, fp8_e4m3 KV):
    # #29781 measured GSM8K 0.950 at EAGLE 5/1/6 (the fixture's config) -> 0.935
    # floor; #30265 measured accept length 5.94 -> 5.0 floor.
    gsm8k_accuracy_thres = 0.935
    gsm8k_accept_length_thres = 5.0
    # GLM-5.2 DSA on gfx950 runs the tilelang NSA backends; fp8 index-K cache.
    extra_server_args = (
        "--nsa-prefill-backend",
        "tilelang",
        "--nsa-decode-backend",
        "tilelang",
        "--kv-cache-dtype",
        "fp8_e4m3",
    )


if __name__ == "__main__":
    unittest.main()
