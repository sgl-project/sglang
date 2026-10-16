"""Hybrid near-bit-exact KL guards: HiCache host cache and buffer-only L3 restores.

Group related scenarios so CI can partition device, HiCache, and MTP coverage
under its 1800-second per-file limit. Runtime estimates need GPU calibration.
"""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.unified_hybrid_bitexact_kit import (
    UnifiedHybridBufferOnlyBitExactMixin,
    UnifiedHybridHiCacheBitExactMixin,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=1100, stage="extra-a", runner_config="1-gpu-large")


class TestUnifiedHybridHiCacheBitExact(
    UnifiedHybridHiCacheBitExactMixin, CustomTestCase
):
    pass


class TestRustUnifiedHybridHiCacheBitExact(
    UnifiedHybridHiCacheBitExactMixin, CustomTestCase
):
    tree_core_backend = "rust"


class TestUnifiedHybridBufferOnlyBitExact(
    UnifiedHybridBufferOnlyBitExactMixin, CustomTestCase
):
    pass


if __name__ == "__main__":
    unittest.main()
