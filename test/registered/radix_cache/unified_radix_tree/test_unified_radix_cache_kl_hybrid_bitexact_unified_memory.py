"""Hybrid near-bit-exact KL guards: unified memory.

Keep scenarios in separate files so CI can partition them independently under
its 1800-second per-file limit. Runtime estimates need GPU calibration.
"""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.unified_hybrid_bitexact_kit import (
    UnifiedMemoryHybridBitExactMixin,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=350, stage="extra-a", runner_config="1-gpu-large")


class TestUnifiedMemoryHybridBitExact(UnifiedMemoryHybridBitExactMixin, CustomTestCase):
    pass


if __name__ == "__main__":
    unittest.main()
