"""Unit tests for runtime_context — PoC M0.1 (parallelism). CPU-only, no torch.

Covers the lifecycle (init/get/reset guards), the parallelism accessors, and the
two-stage fill (engine fields at seed; attention fields None until
with_attention fills them).
"""

import unittest

from sglang.srt.model_executor.runtime_context import (
    ConfigSection,
    ParallelConfig,
    RuntimeContext,
    get_attn_cp_rank,
    get_attn_cp_size,
    get_attn_dp_size,
    get_attn_tp_rank,
    get_attn_tp_size,
    get_dp_rank,
    get_dp_size,
    get_local_attn_dp_size,
    get_moe_dp_rank,
    get_moe_dp_size,
    get_moe_ep_rank,
    get_moe_ep_size,
    get_pp_rank,
    get_pp_size,
    get_runtime_context,
    get_tp_rank,
    get_tp_size,
    has_runtime_context,
    init_runtime_context,
    reset_runtime_context,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _engine_parallel(**kw) -> ParallelConfig:
    """ParallelConfig with engine fields set (attention fields default None)."""
    base = dict(
        tp_size=1,
        tp_rank=0,
        pp_size=1,
        pp_rank=0,
        dp_size=1,
        dp_rank=None,
        moe_ep_size=1,
        moe_ep_rank=0,
        moe_dp_size=1,
        moe_dp_rank=None,
        attn_cp_size=1,
        attn_cp_rank=None,
    )
    base.update(kw)
    return ParallelConfig(**base)


def _publish(par: ParallelConfig) -> None:
    init_runtime_context(RuntimeContext(config=ConfigSection(parallel=par)))


class TestLifecycle(unittest.TestCase):
    def setUp(self):
        reset_runtime_context()

    def tearDown(self):
        reset_runtime_context()

    def test_get_before_init_raises(self):
        self.assertFalse(has_runtime_context())
        with self.assertRaises(ValueError):
            get_runtime_context()
        with self.assertRaises(ValueError):
            get_tp_size()

    def test_init_publishes(self):
        par = _engine_parallel(tp_size=4, tp_rank=2)
        _publish(par)
        self.assertTrue(has_runtime_context())
        self.assertIs(get_runtime_context().config.parallel, par)

    def test_reset_clears(self):
        _publish(_engine_parallel())
        reset_runtime_context()
        self.assertFalse(has_runtime_context())
        with self.assertRaises(ValueError):
            get_runtime_context()

    def test_republish_overwrites(self):
        _publish(_engine_parallel(tp_size=2))
        _publish(_engine_parallel(tp_size=8))
        self.assertEqual(get_tp_size(), 8)


class TestEngineAccessors(unittest.TestCase):
    def setUp(self):
        reset_runtime_context()

    def tearDown(self):
        reset_runtime_context()

    def test_all_engine_accessors(self):
        _publish(
            _engine_parallel(
                tp_size=8,
                tp_rank=3,
                pp_size=2,
                pp_rank=1,
                dp_size=4,
                dp_rank=2,
                moe_ep_size=8,
                moe_ep_rank=5,
                moe_dp_size=2,
                moe_dp_rank=1,
                attn_cp_size=2,
                attn_cp_rank=1,
            )
        )
        self.assertEqual(get_tp_size(), 8)
        self.assertEqual(get_tp_rank(), 3)
        self.assertEqual(get_pp_size(), 2)
        self.assertEqual(get_pp_rank(), 1)
        self.assertEqual(get_dp_size(), 4)
        self.assertEqual(get_dp_rank(), 2)
        self.assertEqual(get_moe_ep_size(), 8)
        self.assertEqual(get_moe_ep_rank(), 5)
        self.assertEqual(get_moe_dp_size(), 2)
        self.assertEqual(get_moe_dp_rank(), 1)
        self.assertEqual(get_attn_cp_size(), 2)
        self.assertEqual(get_attn_cp_rank(), 1)

    def test_optional_ranks_may_be_none(self):
        # tp_worker does not pass attn_cp_rank / moe_dp_rank → commonly None
        _publish(_engine_parallel(dp_rank=None, moe_dp_rank=None, attn_cp_rank=None))
        self.assertIsNone(get_dp_rank())
        self.assertIsNone(get_moe_dp_rank())
        self.assertIsNone(get_attn_cp_rank())


class TestTwoStageFill(unittest.TestCase):
    """Engine fields at seed; attention fields None until with_attention."""

    def setUp(self):
        reset_runtime_context()

    def tearDown(self):
        reset_runtime_context()

    def test_attention_none_before_fill(self):
        _publish(_engine_parallel(tp_size=4))
        # engine dim available immediately
        self.assertEqual(get_tp_size(), 4)
        # attention dims not yet filled
        self.assertIsNone(get_attn_tp_size())
        self.assertIsNone(get_attn_dp_size())
        self.assertIsNone(get_local_attn_dp_size())

    def test_with_attention_fills_and_republishes(self):
        par = _engine_parallel(tp_size=8)
        _publish(par)
        ctx = get_runtime_context()
        filled = ctx.config.parallel.with_attention(
            attn_tp_size=4,
            attn_tp_rank=1,
            attn_dp_size=2,
            attn_dp_rank=0,
            local_attn_dp_size=2,
            local_attn_dp_rank=0,
        )
        # frozen → with_attention returns a new instance, engine fields preserved
        self.assertIsNot(filled, par)
        self.assertEqual(filled.tp_size, 8)
        init_runtime_context(RuntimeContext(config=ConfigSection(parallel=filled)))
        self.assertEqual(get_attn_tp_size(), 4)
        self.assertEqual(get_attn_tp_rank(), 1)
        self.assertEqual(get_attn_dp_size(), 2)
        self.assertEqual(get_local_attn_dp_size(), 2)
        self.assertEqual(get_tp_size(), 8)  # engine dim unchanged

    def test_parallel_config_is_frozen(self):
        par = _engine_parallel()
        with self.assertRaises(Exception):  # FrozenInstanceError
            par.tp_size = 99


class TestPlaceholders(unittest.TestCase):
    def setUp(self):
        reset_runtime_context()

    def tearDown(self):
        reset_runtime_context()

    def test_freeze_and_apply_model_overrides_are_placeholders(self):
        _publish(_engine_parallel())
        ctx = get_runtime_context()
        ctx.apply_model_overrides(model=object())  # no-op placeholder, no raise
        ctx.freeze()  # placeholder, only flips sentinel
        self.assertTrue(ctx._frozen)


if __name__ == "__main__":
    unittest.main()
