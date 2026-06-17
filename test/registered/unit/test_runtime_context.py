"""Unit tests for the unified runtime_context — parallel subsystem. CPU-only.

Covers the lifecycle (set/get/has/reset), get_parallel() reads, and the
build_parallel_context factory (sizes/ranks read off the groups + dp-attention,
dp_size off the runner — all stubbed, no GPU/distributed).
"""

import unittest
from unittest import mock

from sglang.srt.runtime_context import (
    ParallelContext,
    RuntimeContext,
    build_parallel_context,
    get_context,
    get_parallel,
    has_context,
    reset_context,
    set_context,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _publish(**parallel_kw) -> None:
    set_context(RuntimeContext(parallel=ParallelContext(**parallel_kw)))


class TestLifecycle(unittest.TestCase):
    def setUp(self):
        reset_context()

    def tearDown(self):
        reset_context()

    def test_get_before_init_raises(self):
        self.assertFalse(has_context())
        with self.assertRaises(ValueError):
            get_context()
        with self.assertRaises(ValueError):
            get_parallel()

    def test_set_publishes(self):
        _publish(tp_size=4, tp_rank=2, pp_size=2)
        self.assertTrue(has_context())
        self.assertEqual(get_parallel().tp_size, 4)
        self.assertEqual(get_parallel().tp_rank, 2)
        self.assertEqual(get_parallel().pp_size, 2)

    def test_reset_clears(self):
        _publish(tp_size=2)
        reset_context()
        self.assertFalse(has_context())
        with self.assertRaises(ValueError):
            get_context()

    def test_republish_overwrites(self):
        _publish(tp_size=2)
        _publish(tp_size=8)
        self.assertEqual(get_parallel().tp_size, 8)


class TestParallelDefaults(unittest.TestCase):
    def setUp(self):
        reset_context()

    def tearDown(self):
        reset_context()

    def test_defaults_are_single_device(self):
        set_context(RuntimeContext(parallel=ParallelContext()))
        p = get_parallel()
        self.assertEqual(p.world_size, 1)
        self.assertEqual(p.tp_size, 1)
        self.assertEqual(p.dp_size, 1)
        self.assertEqual(p.moe_ep_size, 1)
        self.assertEqual(p.attn_tp_size, 1)
        self.assertEqual(p.attn_dp_size, 1)


class TestBuildParallelContext(unittest.TestCase):
    """The factory: grouped dims read off the process groups directly, attn_dp off the
    dp_attention getters, the folded dp_size off the runner. All stubbed."""

    def setUp(self):
        reset_context()

    def tearDown(self):
        reset_context()

    def test_snapshots_from_groups_and_runner(self):
        class FakeMR:
            dp_size = 3
            dp_rank = 1

        def grp(world_size, rank):
            return mock.Mock(world_size=world_size, rank_in_group=rank)

        ps = "sglang.srt.distributed.parallel_state"
        dpa = "sglang.srt.layers.dp_attention"
        with mock.patch.multiple(
            ps,
            get_world_group=lambda: grp(8, 5),
            get_tp_group=lambda: grp(2, 1),
            get_pp_group=lambda: grp(2, 0),
            get_moe_ep_group=lambda: grp(4, 3),
            get_moe_dp_group=lambda: grp(1, 0),
            get_moe_tp_group=lambda: grp(2, 1),
            get_attn_tp_group=lambda: grp(2, 1),
            get_attn_cp_group=lambda: grp(1, 0),
        ):
            with mock.patch.multiple(
                dpa,
                get_attention_dp_size=lambda: 2,
                get_attention_dp_rank=lambda: 1,
                get_local_attention_dp_size=lambda: 2,
                get_local_attention_dp_rank=lambda: 0,
            ):
                p = build_parallel_context(FakeMR())

        self.assertEqual(p.world_size, 8)
        self.assertEqual(p.world_rank, 5)
        self.assertEqual(p.tp_size, 2)
        self.assertEqual(p.tp_rank, 1)
        self.assertEqual(p.pp_size, 2)
        self.assertEqual(p.moe_ep_size, 4)
        self.assertEqual(p.moe_ep_rank, 3)
        self.assertEqual(p.moe_tp_size, 2)
        self.assertEqual(p.attn_tp_size, 2)
        self.assertEqual(p.attn_cp_size, 1)
        self.assertEqual(p.attn_dp_size, 2)
        self.assertEqual(p.attn_dp_rank, 1)
        self.assertEqual(p.local_attn_dp_size, 2)
        self.assertEqual(p.local_attn_dp_rank, 0)
        self.assertEqual(p.dp_size, 3)  # folded value off the runner
        self.assertEqual(p.dp_rank, 1)


class TestDpAttentionGettersReadContext(unittest.TestCase):
    """The dp_attention attn_dp / local_attn_dp getters read the published
    context snapshot (with module-global fallback when no context)."""

    def setUp(self):
        reset_context()

    def tearDown(self):
        reset_context()

    def test_attn_dp_getters_read_context(self):
        from sglang.srt.layers.dp_attention import (
            get_attention_dp_rank,
            get_attention_dp_size,
            get_local_attention_dp_rank,
            get_local_attention_dp_size,
        )

        set_context(
            RuntimeContext(
                parallel=ParallelContext(
                    attn_dp_size=4,
                    attn_dp_rank=3,
                    local_attn_dp_size=2,
                    local_attn_dp_rank=1,
                )
            )
        )
        self.assertEqual(get_attention_dp_size(), 4)
        self.assertEqual(get_attention_dp_rank(), 3)
        self.assertEqual(get_local_attention_dp_size(), 2)
        self.assertEqual(get_local_attention_dp_rank(), 1)


if __name__ == "__main__":
    unittest.main()
