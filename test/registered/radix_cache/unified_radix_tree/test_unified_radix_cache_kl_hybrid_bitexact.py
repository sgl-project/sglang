"""Hybrid near-bit-exact KL guards: device cache strategies and unified memory.

Group related scenarios so CI can partition device, HiCache, and MTP coverage
under its 1800-second per-file limit. Runtime estimates need GPU calibration.
"""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.unified_hybrid_bitexact_kit import (
    UnifiedHybridBitExactMixin,
    UnifiedHybridLazyBitExactMixin,
    UnifiedMemoryHybridBitExactMixin,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=1550, stage="extra-a", runner_config="1-gpu-large")


class TestUnifiedHybridBitExact(UnifiedHybridBitExactMixin, CustomTestCase):
    pass


class TestRustUnifiedHybridBitExact(UnifiedHybridBitExactMixin, CustomTestCase):
    tree_core_backend = "rust"


class TestUnifiedHybridLazyBitExact(UnifiedHybridLazyBitExactMixin, CustomTestCase):
    pass


class TestRustUnifiedHybridLazyBitExact(UnifiedHybridLazyBitExactMixin, CustomTestCase):
    tree_core_backend = "rust"


class TestUnifiedMemoryHybridBitExact(UnifiedMemoryHybridBitExactMixin, CustomTestCase):
    pass


if __name__ == "__main__":
    unittest.main()
