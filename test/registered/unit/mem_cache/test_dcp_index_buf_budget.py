import unittest

import torch

from sglang.srt.mem_cache.kv_cache_configurator import (
    dcp_index_buf_widening_factor,
    dcp_virtual_loc_extent,
)
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

MAX_TOTAL = 1_000_000


class TestDcpIndexBufBudget(CustomTestCase):
    """The memory budget for index-K has to match what the pool allocates.

    Under DCP the latent KV *shards* -- each rank keeps ``max_total`` rows and
    translates ``// dcp_size`` on the way in -- while the LightningIndexer's
    index-K is *replicated* over the whole virtual loc space, ``max_total *
    dcp_size``. A per-token cost that counts the indexer once therefore derives
    a ``max_total`` the pool cannot honour, and the only symptom is a bare
    ``NPU out of memory`` during pool construction with nothing naming DCP.

    Observed on A3 at DCP16 before the fix: 1.08 GiB budgeted for the indexer
    against 17.35 GiB allocated, a 16.27 GiB overshoot per die, on a card with
    310 MiB free at the point it gave up.

    None of this is reachable by CI, which runs at ``dcp_size == 1`` on a CUDA
    pool, where both correction terms collapse to 1 and the wrong arithmetic
    looks right.
    """

    def test_the_factor_is_inert_without_dcp(self):
        for replicated in (True, False):
            with self.subTest(replicated=replicated):
                self.assertEqual(
                    dcp_index_buf_widening_factor(
                        1, index_buf_is_replicated=replicated
                    ),
                    1,
                )

    def test_a_sharded_index_buffer_is_never_scaled(self):
        """CUDA does not widen index-K, so its budget must not change."""
        for dcp_size in (2, 4, 8, 16):
            with self.subTest(dcp_size=dcp_size):
                self.assertEqual(
                    dcp_index_buf_widening_factor(
                        dcp_size, index_buf_is_replicated=False
                    ),
                    1,
                )

    def test_a_replicated_index_buffer_costs_dcp_size_times_more(self):
        for dcp_size in (2, 4, 8, 16):
            with self.subTest(dcp_size=dcp_size):
                self.assertEqual(
                    dcp_index_buf_widening_factor(
                        dcp_size, index_buf_is_replicated=True
                    ),
                    dcp_size,
                )

    def test_the_budget_covers_exactly_the_rows_the_allocator_asks_for(self):
        """The regression guard, and the reason both helpers live side by side.

        ``dcp_virtual_loc_extent`` decides how many rows are allocated;
        ``dcp_index_buf_widening_factor`` decides how many are paid for. If they
        ever disagree the pool OOMs at load, so assert they agree rather than
        trusting two call sites in two files to be edited together.
        """
        for dcp_size in (1, 2, 4, 8, 16):
            with self.subTest(dcp_size=dcp_size):
                paid_for = MAX_TOTAL * dcp_index_buf_widening_factor(
                    dcp_size, index_buf_is_replicated=True
                )
                # loc_space_scale 1: the target worker, whose sizes reach the
                # builder unscaled. The draft worker's arrive pre-multiplied and
                # dcp_virtual_loc_extent divides that back out.
                allocated = dcp_virtual_loc_extent(MAX_TOTAL, dcp_size, 1)
                self.assertEqual(paid_for, allocated)

    def test_the_two_pools_price_index_k_differently(self):
        """Why the budget also needed a per-element correction, not just DCP.

        ``NPUMLATokenToKVPool`` stores index-K plain -- ``index_head_dim`` at the
        pool's own dtype. ``DSATokenToKVPool`` packs k-with-scale into uint8.
        Pricing the second for the first under-counts by nearly 2x on a bf16
        cache, independently of DCP, which is why the shortfall was visible even
        at ``dcp_size == 1`` where every DCP term is inert.
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
