"""Numerical tests for the varlen absorbed-MLA extend path.

Under a captured tc_piecewise prefill graph, trtllm_mla runs absorbed MLA over a
ragged q against a freshly built paged block table. These cases check the output
still matches the reference across prefix lengths, page boundaries, ragged batches
and shuffled pages.

Backends without a varlen kernel are covered by test_mla_varlen_absorbed_gate.py:
the kit builds MLA shapes (576, 512), which they reject, so every case here would
skip.

The case list and assertions are shared with test_trtllm_mla_breakable.py via
sglang.test.kits.attention_unittest.attention_methods.varlen_absorbed_extend_kit:
the two exercise the same numerical contract under the two halves of
use_varlen_absorbed (is_in_tc_piecewise_cuda_graph() vs
is_in_breakable_cuda_graph()), so only the capture-mode kwarg and the case-name
prefix differ.
"""

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
register_cuda_ci(est_time=15, stage="base-b", runner_config="4-gpu-b200")
register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-large")


@unittest.skipIf(not _SUPPORTED, _SKIP_REASON)
class TestTRTLLMMLAPiecewiseExtend(VarlenAbsorbedExtendMixin, CustomTestCase):
    CASES = cases("trtllm_mla", "pcg")
    MODE_KWARGS = {"piecewise": True}
    MODE_NAME = "tc_piecewise"


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestVarlenAbsorbedArchGate(CustomTestCase):
    """The arch gate must be asserted, not merely skipped over.

    On SM != 10.x flashinfer resolves backend="auto" to XQA, which rejects
    cum_seq_lens_q. The class above skips there, so without this test the
    non-SM10 half of the gate would have no coverage at all -- and that gate
    exists because trusting the ``backend == "trtllm-gen"`` string was wrong.
    1-gpu-large (H100, SM 9.0) is a per-commit runner, so this runs every PR.
    """

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
