"""Varlen absorbed-MLA extend under a breakable captured prefill graph."""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.attention_unittest.attention_methods.varlen_absorbed_extend_kit import (
    VarlenAbsorbedExtendMixin,
    cases,
    supported,
)
from sglang.test.test_utils import CustomTestCase

_SUPPORTED, _SKIP_REASON = supported()

# 4-gpu-b200 is SM 10.0, the only per-commit runner where _supported() is true;
# 1-gpu-large (H100, SM 9.0) only exercises the skip path. Mirrors the
# registration of test_trtllm_mla_piecewise.py.
register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="4-gpu-b200")
register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@unittest.skipIf(not _SUPPORTED, _SKIP_REASON)
class TestTRTLLMMLABreakableExtend(VarlenAbsorbedExtendMixin, CustomTestCase):
    CASES = cases("trtllm_mla", "bcg")
    MODE_KWARGS = {"breakable": True}
    MODE_NAME = "breakable"


if __name__ == "__main__":
    unittest.main()
