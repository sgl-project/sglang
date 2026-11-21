"""UNO speculative decoding on Intel XPU (intel_xpu attention, XPU decode graphs)."""

import os
import sys
import unittest

from sglang.srt.utils import is_xpu

# Put the `test/` root on sys.path so `registered.<...>` resolves regardless of
# cwd: CI runs each file as `python3 <full_path>` (only the file's own dir is on
# the path), and pytest inserts only the file's dir too. `test/` is three levels
# up from this file's dir (test/registered/e2e/xpu/<this>).
_TEST_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _TEST_ROOT not in sys.path:
    sys.path.insert(0, _TEST_ROOT)

# Reference the shared mixin as a module attribute rather than importing the
# Test* names: pytest collects by class __name__, so importing the base module's
# CUDA TestUnoCudaGraph here would re-collect it under the XPU file.
from registered.spec.uno import test_uno as _uno_base

from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=480, suite="nightly-xpu-kernel-main-1-gpu", nightly=True)


@unittest.skipUnless(is_xpu(), "Intel XPU required")
class TestUnoIntelXPU(_uno_base._UnoDecodingMixin, CustomTestCase):
    # The BMG cookbook command minus its prefill graph flag, which this decode
    # test does not exercise. The 24 GB card needs a higher static fraction to
    # hold the 16.4 GB model plus the 1.4 GB UNO LoRA pool.
    platform_args = (
        "--attention-backend",
        "intel_xpu",
        "--page-size",
        "64",
        "--cuda-graph-backend-decode",
        "full",
        "--mem-fraction-static",
        "0.88",
    )
    # XPU bf16 GEMMs are not batch-invariant, so AR's one-row decode and UNO's
    # F-row verify can flip a near-tie; AR alone flipped at a 0.25 gap on B60.
    # Same bound as SpecServerKit's decode-vs-prefill logprob_match_delta.
    near_tie_logprob_gap = 0.5


if __name__ == "__main__":
    unittest.main()
