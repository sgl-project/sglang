"""``rust_server.config._partition_cores``: the pool cores must
be a *bounded* slice of this rank's allowed cores, not the whole remainder —
sibling TP ranks share the NUMA node, so an unbounded mask lets MM preprocessing
bursts preempt a sibling's CUDA-launch thread (measured: ~20 ms of ViT wall time
per image request on TP4). Pure computation, so no Rust extension needed."""

import unittest
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.rust_server.config import (  # noqa: E402
    _frontend_cpu_share,
    _partition_cores,
)

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


def partition(node_cores, **kwargs):
    with patch("os.sched_getaffinity", return_value=set(node_cores), create=True):
        return _partition_cores(**kwargs)


def listener_pools(masks):
    """Frontend pools of the listener ranks whose affinity masks are ``masks``
    (``None`` for a rank hosting no listener), as one host computes them."""
    keys = [None if m is None else ("node", tuple(sorted(m))) for m in masks]
    return [
        partition(mask, cpu_share=_frontend_cpu_share(keys, rank))[1]
        for rank, mask in enumerate(masks)
        if mask is not None
    ]


class TestPartitionCores(CustomTestCase):
    def test_pool_is_bounded_not_the_node_remainder(self):
        # A 120-core NUMA node shared with sibling TP ranks: the pools must
        # NOT get cores 2..119.
        launch, pool = partition(range(120), mm_workers=8)
        self.assertEqual(launch, [0, 1])
        self.assertEqual(pool, list(range(2, 14)))  # max(8, 8 + 4) after reserve

    def test_budget_scales_with_mm_workers_with_a_floor(self):
        _, pool_text = partition(range(120), mm_workers=0)
        _, pool_mm = partition(range(120), mm_workers=16)
        self.assertEqual(len(pool_text), 8)  # the floor covers the I/O threads
        self.assertEqual(len(pool_mm), 20)

    def test_small_allowance_degrades_gracefully(self):
        # Fewer allowed cores than the budget: take what exists after the
        # launch reserve, never raise.
        launch, pool = partition(range(6), mm_workers=8)
        self.assertEqual(launch, [0])  # min(2, 6 // 4) == 1
        self.assertEqual(pool, list(range(1, 6)))
        # Below the split threshold: unpinned.
        self.assertEqual(partition(range(3), mm_workers=8), (None, None))

    def test_launch_and_pool_cores_are_disjoint(self):
        launch, pool = partition(range(32), mm_workers=8)
        self.assertFalse(set(launch) & set(pool))


class TestFrontendCpuShare(CustomTestCase):
    def test_attention_dp_listeners_on_one_mask_get_disjoint_pools(self):
        """DP attention puts a listener on every DP rank; ranks launched with
        one affinity mask (no NUMA bind) all pinned their frontends to cores
        2..9 of it (GLM-5.3 DPA8 on 8xH200: 8 frontends on 8 of 128 cores)."""
        pools = listener_pools([range(128)] * 8)
        self.assertEqual(pools[0], list(range(2, 10)))
        self.assertEqual(pools[7], list(range(114, 122)))
        cores = [core for pool in pools for core in pool]
        self.assertEqual(len(cores), len(set(cores)), "pools overlap")

    def test_split_only_among_listeners_that_share_a_mask(self):
        """NUMA bind gives four ranks per socket the same mask: each socket's
        mask is split four ways, never across sockets, and ranks hosting no
        listener take no block."""
        s0, s1 = range(64), range(64, 128)
        masks = [s0, None, s0, s1, s0, s1, s1, s0, s1]
        pools = listener_pools(masks)
        cores = [core for pool in pools for core in pool]
        self.assertEqual(len(cores), len(set(cores)), "pools overlap")
        listener_masks = [m for m in masks if m is not None]
        for pool, mask in zip(pools, listener_masks):
            self.assertLessEqual(set(pool), set(mask))
        # 64 cores / 4 listeners: blocks of 16, pool = cores 2..9 of a block.
        self.assertEqual(pools[2], list(range(66, 74)))
        self.assertEqual(pools[7], list(range(114, 122)))

    def test_distinct_masks_are_not_shrunk(self):
        """Per-rank affinity (e.g. SGLANG_SET_CPU_AFFINITY) already separates
        the listeners; splitting would cut each one's pool for nothing."""
        masks = [range(16 * r, 16 * (r + 1)) for r in range(8)]
        self.assertEqual(listener_pools(masks)[3], partition(masks[3])[1])

    def test_too_few_cores_per_listener_keeps_the_whole_mask(self):
        """Blocks under the split threshold would leave pools unpinned or one
        core wide; listeners then overlap on the full mask as before."""
        pools = listener_pools([range(12)] * 4)
        self.assertEqual(pools[0], partition(range(12))[1])
        self.assertEqual(pools[3], pools[0])


if __name__ == "__main__":
    unittest.main()
