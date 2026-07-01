"""Unit tests for the DeepEP low_latency RDMA buffer-size replica.

estimate_low_latency_rdma_size_bytes is a pure-Python replica of DeepEP's C++
LowLatencyLayout.total_bytes, used by the auto mem_fraction reservation before
deep_ep is importable. These tests pin its closed form and, when deep_ep is
available, assert it matches the native Buffer.get_low_latency_rdma_size_hint
across a grid including ep_size (which the native hint must ignore).
"""

from __future__ import annotations

import unittest

from sglang.srt.layers.moe.token_dispatcher.deepep import (
    estimate_low_latency_rdma_size_bytes,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_NUM_MAX = (128, 256, 512, 1024)
_HIDDEN = (2048, 4096, 5120, 6144, 7168)
_EXPERTS = (64, 128, 256)
_EP_SIZE = (1, 4, 8)

try:
    from deep_ep import Buffer

    _HAS_DEEP_EP = True
except Exception:
    _HAS_DEEP_EP = False


class TestDeepEPBufferSizeReplica(unittest.TestCase):
    def test_replica_is_128_byte_aligned_and_positive(self):
        for num_max in _NUM_MAX:
            for hidden in _HIDDEN:
                for num_experts in _EXPERTS:
                    size = estimate_low_latency_rdma_size_bytes(
                        num_max, hidden, num_experts
                    )
                    self.assertGreater(size, 0)
                    self.assertEqual(size % 128, 0)

    def test_replica_monotonic_in_num_max(self):
        for hidden in _HIDDEN:
            for num_experts in _EXPERTS:
                sizes = [
                    estimate_low_latency_rdma_size_bytes(nm, hidden, num_experts)
                    for nm in _NUM_MAX
                ]
                self.assertEqual(sizes, sorted(sizes))

    def test_known_value_glm52(self):
        # GLM-5.2 (hidden=6144, experts=256) at the num_max ceiling = 12.18 GiB.
        self.assertEqual(
            estimate_low_latency_rdma_size_bytes(1024, 6144, 256), 13086230656
        )

    @unittest.skipUnless(_HAS_DEEP_EP, "deep_ep not installed")
    def test_replica_matches_native(self):
        for num_max in _NUM_MAX:
            for hidden in _HIDDEN:
                for num_experts in _EXPERTS:
                    replica = estimate_low_latency_rdma_size_bytes(
                        num_max, hidden, num_experts
                    )
                    for ep_size in _EP_SIZE:
                        native = Buffer.get_low_latency_rdma_size_hint(
                            num_max, hidden, ep_size, num_experts
                        )
                        self.assertEqual(
                            replica,
                            native,
                            msg=(
                                f"replica != native at num_max={num_max} "
                                f"hidden={hidden} experts={num_experts} "
                                f"ep_size={ep_size}: {replica} vs {native}"
                            ),
                        )


if __name__ == "__main__":
    unittest.main()
