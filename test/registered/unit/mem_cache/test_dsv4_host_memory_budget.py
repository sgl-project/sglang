import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.memory_pool_host import (
    DeepSeekV4PagedHostPool,
    DeepSeekV4StateHostPool,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestV4HostBudget(CustomTestCase):
    def test_paged_and_state_pools_share_one_allocation_budget(self):
        from sglang.srt.mem_cache.pool_host.base import (
            host_memory_budget_bytes,
            host_memory_budget_scope,
        )

        def build(kind, layout):
            if kind == "paged":
                return DeepSeekV4PagedHostPool(
                    "kv",
                    [torch.empty((4, 6), dtype=torch.uint8)],
                    item_bytes=6,
                    num_host_pages=4,
                    slot_page_size=2,
                    layout=layout,
                    pin_memory=False,
                )
            return DeepSeekV4StateHostPool(
                "state",
                [
                    SimpleNamespace(
                        ring_size=2,
                        kv_score_buffer=SimpleNamespace(
                            kv_score=torch.empty((4, 3), dtype=torch.uint8)
                        ),
                    )
                ],
                num_host_pages=4,
                swa_page_size=2,
                layout=layout,
                pin_memory=False,
            )

        for layout in ("layer_first", "page_first", "page_first_direct"):
            for first, second in (("paged", "state"), ("state", "paged")):
                with (
                    self.subTest(layout=layout, first=first),
                    host_memory_budget_scope(40),
                ):
                    pool = build(first, layout)
                    buffers = pool.get_hybrid_pool_buffer()
                    self.assertEqual(sum(buffer.nbytes for buffer in buffers), 24)
                    self.assertEqual(host_memory_budget_bytes(), 16)
                    # Each pool fits alone, but their sum exceeds the snapshot.
                    with self.assertRaisesRegex(ValueError, "Not enough host memory"):
                        build(second, layout)
                    self.assertEqual(host_memory_budget_bytes(), 16)


if __name__ == "__main__":
    unittest.main()
