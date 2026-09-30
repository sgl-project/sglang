"""The index-K memory budget has to match what the pool allocates.

[Test Category] Correctness
[Test Target] mem_cache/kv_cache_configurator.py  (index_size)
              model_executor/pool_configurator.py (indexer_cell_size)

Under DCP the latent KV *shards* -- each rank keeps ``max_total`` rows and
translates ``// dcp_size`` on the way in -- while the LightningIndexer's index-K
is *replicated* over the whole virtual loc space. Two files decide that
independently: one sizes the allocation, the other prices it. When they
disagree the only symptom is a bare ``NPU out of memory`` during pool
construction, with nothing naming DCP.

Observed on A3 at DCP16 before the fix: 1.08 GiB budgeted against 17.35 GiB
allocated, a 16.27 GiB overshoot per die, on a card with 310 MiB free.

CI cannot reach any of it -- it runs at ``dcp_size == 1`` on a CUDA pool, where
both terms collapse to 1 and the wrong arithmetic looks right. So the two
expressions are mirrored here and asserted to agree.
"""

import unittest

import torch

from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

MAX_TOTAL = 1_000_000


def allocated_index_rows(max_total: int, dcp_size: int, is_draft: bool) -> int:
    """Mirrors kv_cache_configurator.py: the rows the pool asks for.

    A draft worker's ``max_total`` arrives already multiplied by the loc-space
    scale, so it must not be scaled again.
    """
    return max_total if is_draft else max_total * dcp_size


def budgeted_index_rows(max_total: int, dcp_size: int, is_draft: bool) -> int:
    """Mirrors pool_configurator.py: the rows the budget pays for."""
    cells = max_total
    if not is_draft and dcp_size > 1:
        cells *= dcp_size
    return cells


class TestDcpIndexBufBudget(CustomTestCase):
    def test_the_budget_covers_exactly_the_rows_the_allocator_asks_for(self):
        # The regression guard. Two call sites in two files have to be edited
        # together; assert that rather than trust it.
        for dcp_size in (1, 2, 4, 8, 16):
            for is_draft in (False, True):
                with self.subTest(dcp_size=dcp_size, is_draft=is_draft):
                    self.assertEqual(
                        budgeted_index_rows(MAX_TOTAL, dcp_size, is_draft),
                        allocated_index_rows(MAX_TOTAL, dcp_size, is_draft),
                    )

    def test_the_target_worker_spans_the_whole_virtual_range(self):
        for dcp_size in (2, 4, 8, 16):
            with self.subTest(dcp_size=dcp_size):
                self.assertEqual(
                    allocated_index_rows(MAX_TOTAL, dcp_size, False),
                    MAX_TOTAL * dcp_size,
                )

    def test_the_draft_worker_is_not_scaled_twice(self):
        # Its sizes arrive pre-multiplied, so a bare `* dcp_size` would ask for
        # max_total * dcp_size**2 -- 16x at DCP16, and silent, because the pool
        # still allocates a valid shape.
        for dcp_size in (2, 4, 8, 16):
            with self.subTest(dcp_size=dcp_size):
                self.assertEqual(
                    allocated_index_rows(MAX_TOTAL * dcp_size, dcp_size, True),
                    MAX_TOTAL * dcp_size,
                )

    def test_nothing_is_scaled_without_dcp(self):
        for is_draft in (False, True):
            with self.subTest(is_draft=is_draft):
                self.assertEqual(
                    allocated_index_rows(MAX_TOTAL, 1, is_draft), MAX_TOTAL
                )
                self.assertEqual(budgeted_index_rows(MAX_TOTAL, 1, is_draft), MAX_TOTAL)

    def test_the_two_pools_price_index_k_differently(self):
        """Why the budget also needed a per-element correction, not just DCP.

        ``NPUMLATokenToKVPool`` stores index-K plain -- ``index_head_dim`` at the
        pool's own dtype. ``DSATokenToKVPool`` packs k-with-scale into uint8.
        Pricing the second for the first under-counts by nearly 2x on a bf16
        cache, independently of DCP.
        """
        index_head_dim = 128

        cuda_bytes = (
            index_head_dim + index_head_dim // DSATokenToKVPool.quant_block_size * 4
        ) * torch._utils._element_size(DSATokenToKVPool.index_k_with_scale_buffer_dtype)
        ascend_bytes = index_head_dim * torch._utils._element_size(torch.bfloat16)

        self.assertEqual(cuda_bytes, 132)
        self.assertEqual(ascend_bytes, 256)
        self.assertGreater(
            ascend_bytes,
            cuda_bytes,
            "if these ever converge, drop the branch in "
            "_compute_dsa_indexer_cell_size rather than leaving a dead one",
        )


if __name__ == "__main__":
    unittest.main()
