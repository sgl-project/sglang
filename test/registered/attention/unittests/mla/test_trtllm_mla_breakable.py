"""Numerical tests for the varlen absorbed-MLA extend path under a *breakable*
(segmented, non-torch.compile) captured prefill graph.

Mirrors test_trtllm_mla_piecewise.py exactly, but exercises
is_in_breakable_cuda_graph() instead of is_in_tc_piecewise_cuda_graph(): the
two are the OR'd halves of TRTLLMMLABackend's use_varlen_absorbed predicate,
and neither existing test forces the breakable half to be true in isolation.

Backends without a varlen kernel are covered by test_mla_varlen_absorbed_gate.py:
the kit builds MLA shapes (576, 512), which they reject, so every case here would
skip.

The case list and assertions are shared with test_trtllm_mla_piecewise.py via
sglang.test.kits.attention_unittest.attention_methods.varlen_absorbed_extend_kit:
see that module's docstring.
"""

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
register_cuda_ci(est_time=15, stage="base-b", runner_config="4-gpu-b200")
register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-large")


@unittest.skipIf(not _SUPPORTED, _SKIP_REASON)
class TestTRTLLMMLABreakableExtend(VarlenAbsorbedExtendMixin, CustomTestCase):
    CASES = cases("trtllm_mla", "bcg")
    MODE_KWARGS = {"breakable": True}
    MODE_NAME = "breakable"


if __name__ == "__main__":
    unittest.main()
