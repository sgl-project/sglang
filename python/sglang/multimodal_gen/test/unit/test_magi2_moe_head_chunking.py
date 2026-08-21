# SPDX-License-Identifier: Apache-2.0
"""A rank owns ``(12 / ep_size) * 256`` experts, so ep=1, 2 and 3 cross the MoE
kernel's 1023-expert ceiling; head chunking is what keeps them under it.
"""

import unittest

from sglang.multimodal_gen.runtime.layers.moe_multihead import (
    MAX_KERNEL_EXPERTS,
    Magi2MultiHeadExperts,
)

MOE_HEADS = 12
EXPERTS_PER_HEAD = 256


class TestHeadChunking(unittest.TestCase):
    def _experts(self, num_heads: int) -> Magi2MultiHeadExperts:
        return Magi2MultiHeadExperts(
            num_heads=num_heads,
            num_experts_per_head=EXPERTS_PER_HEAD,
            head_dim=256,
            intermediate_size=1280,
        )

    def test_every_ep_degree_stays_under_the_ceiling(self):
        for ep in (1, 2, 3, 4, 6, 12):
            experts = self._experts(MOE_HEADS // ep)
            per_call = experts.heads_per_call * EXPERTS_PER_HEAD
            self.assertLessEqual(
                per_call,
                MAX_KERNEL_EXPERTS,
                f"ep={ep} calls the kernel with {per_call} experts",
            )

    def test_ep4_and_below_still_run_in_one_call(self):
        for ep in (4, 6, 12):
            experts = self._experts(MOE_HEADS // ep)
            self.assertLessEqual(experts.num_heads, experts.heads_per_call)

    def test_ep3_chunks_rather_than_calling_with_1024(self):
        experts = self._experts(MOE_HEADS // 3)
        self.assertEqual(experts.num_local_experts, 1024)
        self.assertGreater(experts.num_heads, experts.heads_per_call)

    def test_single_head_over_the_ceiling_raises(self):
        with self.assertRaises(ValueError):
            Magi2MultiHeadExperts(
                num_heads=1,
                num_experts_per_head=MAX_KERNEL_EXPERTS + 1,
                head_dim=256,
                intermediate_size=1280,
            )


if __name__ == "__main__":
    unittest.main()
