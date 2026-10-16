"""Hybrid near-bit-exact KL guards: MTP.

Group related scenarios so CI can partition device, HiCache, and MTP coverage
under its 1800-second per-file limit. Runtime estimates need GPU calibration.
"""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.unified_hybrid_bitexact_kit import UnifiedHybridMTPBitExactMixin
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=800, stage="extra-a", runner_config="1-gpu-large")


class TestUnifiedHybridMTPBitExact(UnifiedHybridMTPBitExactMixin, CustomTestCase):
    pass


class TestRustUnifiedHybridMTPBitExact(UnifiedHybridMTPBitExactMixin, CustomTestCase):
    tree_core_backend = "rust"


if __name__ == "__main__":
    unittest.main()
