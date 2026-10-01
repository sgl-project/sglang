"""Varlen absorbed-MLA extend under a tc_piecewise captured prefill graph."""

import unittest

import torch

from sglang.srt.layers.attention.trtllm_mla_backend import (
    varlen_absorbed_mla_supported,
)
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
# registration of test_trtllm_mla.py / test_tokenspeed_mla.py.
register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="4-gpu-b200")
register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@unittest.skipIf(not _SUPPORTED, _SKIP_REASON)
class TestTRTLLMMLAPiecewiseExtend(VarlenAbsorbedExtendMixin, CustomTestCase):
    CASES = cases("trtllm_mla", "pcg")
    MODE_KWARGS = {"piecewise": True}
    MODE_NAME = "tc_piecewise"


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestVarlenAbsorbedArchGate(CustomTestCase):
    """Off SM100, FlashInfer's auto backend is XQA, which rejects cum_seq_lens_q."""

    def test_arch_gate_matches_flashinfer_resolution(self):
        major, minor = torch.cuda.get_device_capability()
        expected = major == 10
        # Call the predicate rather than instantiating a backend (construction
        # needs a full ModelRunner). fp8 KV is the shipped configuration, so this
        # isolates the arch half of the gate from the FP4-KV half.
        self.assertEqual(
            varlen_absorbed_mla_supported(torch.float8_e4m3fn),
            expected,
            f"the arch gate disagrees with SM {major}.{minor}",
        )
        if expected:
            self.assertTrue(
                _SUPPORTED,
                f"SM {major}.{minor} is SM 10.x, so the numerical cases above "
                "must not be skipped",
            )
        else:
            self.assertFalse(
                _SUPPORTED,
                f"SM {major}.{minor} must take the FlashInfer fallback, "
                "not varlen absorbed MLA",
            )


if __name__ == "__main__":
    unittest.main()
