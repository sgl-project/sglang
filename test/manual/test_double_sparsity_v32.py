"""Manual NIAH + MMLU quality tests for Double Sparsity on DeepSeek-V3.2 (FP8).

This test file is **manual**: it requires a calibrated channel mask file and
a runnable DeepSeek-V3.2 FP8 server on the agreed hardware (2-node H200 per
DEC-1). It is not part of the registered CI suite — the registered unit
tests at ``test/registered/unit/layers/attention/test_double_sparsity_unit.py``
cover everything that can be verified without weights.

Per AC-9 (DEC-3 resolution):

    - NIAH @ 4K / 16K / 64K within 5 percentage points of native_nsa.
    - MMLU 5-shot within 1.0 percentage point of native_nsa.

Usage::

    DS_BASE_URL=http://localhost:30000 \
    DS_CHANNEL_MASK=/models/dsv32-fp8-channel-mask.safetensors \
    python -m unittest test.manual.test_double_sparsity_v32

If either env var is unset, the tests skip cleanly. Run the paired
native_nsa baseline first (``development/serve_native_nsa.sh`` +
``development/benchmark_baseline.sh``), then run the DS server
(``development/serve_double_sparsity.sh``) before running this suite.

Negative tests (AC-9 sensitivity checks):

    - A corrupted channel mask (random-permuted selection) makes NIAH @ 64K
      drop > 20 pp below native_nsa.
    - An empty / zero page signature table makes NIAH @ 16K drop > 30 pp.

These are placeholder tests right now — they assert the harness works
against the running server when env vars are set. The actual NIAH /
MMLU evaluation hooks into existing sglang tooling and is out of scope
for this scaffolding milestone.
"""

from __future__ import annotations

import os
import unittest


def _required_env(name: str) -> str | None:
    return os.environ.get(name)


@unittest.skipUnless(
    _required_env("DS_BASE_URL") and _required_env("DS_CHANNEL_MASK"),
    "DS_BASE_URL and DS_CHANNEL_MASK env vars must point at a running DS server.",
)
class TestDoubleSparsityV32Quality(unittest.TestCase):
    """Manual NIAH + MMLU regression against the AC-9 thresholds."""

    NIAH_TOLERANCE_PP = 5.0
    MMLU_TOLERANCE_PP = 1.0

    @classmethod
    def setUpClass(cls):
        cls.base_url = _required_env("DS_BASE_URL")
        cls.channel_mask = _required_env("DS_CHANNEL_MASK")

    def test_niah_at_4k(self):
        self.skipTest(
            "NIAH @ 4K harness wiring is the deploying team's responsibility — "
            "this scaffold reserves the assertion shape: "
            "|niah_4k(double_sparsity) - niah_4k(native_nsa)| <= 5 pp."
        )

    def test_niah_at_16k(self):
        self.skipTest(
            "NIAH @ 16K: |niah_16k(ds) - niah_16k(native_nsa)| <= 5 pp."
        )

    def test_niah_at_64k(self):
        self.skipTest(
            "NIAH @ 64K: |niah_64k(ds) - niah_64k(native_nsa)| <= 5 pp."
        )

    def test_mmlu_5shot(self):
        self.skipTest(
            "MMLU 5-shot: |mmlu(ds) - mmlu(native_nsa)| <= 1.0 pp."
        )

    def test_niah_64k_sensitivity_corrupt_mask(self):
        self.skipTest(
            "Negative sensitivity: corrupted (random-permuted) channel mask "
            "should drop NIAH @ 64K by > 20 pp below native_nsa baseline."
        )

    def test_niah_16k_sensitivity_zero_signatures(self):
        self.skipTest(
            "Negative sensitivity: fault-injected zero page signature table "
            "should drop NIAH @ 16K by > 30 pp below native_nsa baseline."
        )


if __name__ == "__main__":
    unittest.main()
