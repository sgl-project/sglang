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
    """The factory: grouped dims read off the parallel_state getters, attn_dp off the
    dp_attention getters, dp_size off the runner. All stubbed."""

    def setUp(self):
        reset_context()

    def tearDown(self):
        reset_context()

    def test_snapshots_from_getters_and_runner(self):
        class FakeMR:
            dp_size = 3
            dp_rank = 1

        ps = "sglang.srt.distributed.parallel_state"
        dpa = "sglang.srt.layers.dp_attention"
        with mock.patch.multiple(
            ps,
            get_world_size=lambda: 8,
            get_world_rank=lambda: 5,
            get_tensor_model_parallel_world_size=lambda: 2,
            get_tensor_model_parallel_rank=lambda: 1,
            get_pipeline_model_parallel_world_size=lambda: 2,
            get_pipeline_model_parallel_rank=lambda: 0,
            get_moe_expert_parallel_world_size=lambda: 4,
            get_moe_expert_parallel_rank=lambda: 3,
            get_moe_data_parallel_world_size=lambda: 1,
            get_moe_data_parallel_rank=lambda: 0,
            get_moe_tensor_parallel_world_size=lambda: 2,
            get_moe_tensor_parallel_rank=lambda: 1,
            get_attn_tensor_model_parallel_world_size=lambda: 2,
            get_attn_tensor_model_parallel_rank=lambda: 1,
            get_attn_context_model_parallel_world_size=lambda: 1,
            get_attn_context_model_parallel_rank=lambda: 0,
        ):
            with mock.patch.multiple(
                dpa,
                get_attention_dp_size=lambda: 2,
                get_attention_dp_rank=lambda: 1,
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
        self.assertEqual(p.dp_size, 3)  # folded value off the runner
        self.assertEqual(p.dp_rank, 1)


if __name__ == "__main__":
    unittest.main()
