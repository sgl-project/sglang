"""Unit tests for runtime_context: delegation, singletons, and override()."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=7, suite="base-a-test-cpu")

import unittest
from unittest.mock import patch

from sglang.srt.runtime_context import (
    ParallelContext,
    RuntimeContext,
    get_context,
    get_parallel,
)
from sglang.test.test_utils import CustomTestCase

_PS = "sglang.srt.distributed.parallel_state"
_DP = "sglang.srt.layers.dp_attention"

SIZE_RANK_DELEGATIONS = [
    ("world_size", f"{_PS}.get_world_size"),
    ("world_rank", f"{_PS}.get_world_rank"),
    ("tp_size", f"{_PS}.get_tensor_model_parallel_world_size"),
    ("tp_rank", f"{_PS}.get_tensor_model_parallel_rank"),
    ("pp_size", f"{_PS}.get_pipeline_model_parallel_world_size"),
    ("pp_rank", f"{_PS}.get_pipeline_model_parallel_rank"),
    ("moe_ep_size", f"{_PS}.get_moe_expert_parallel_world_size"),
    ("moe_ep_rank", f"{_PS}.get_moe_expert_parallel_rank"),
    ("moe_dp_size", f"{_PS}.get_moe_data_parallel_world_size"),
    ("moe_dp_rank", f"{_PS}.get_moe_data_parallel_rank"),
    ("moe_tp_size", f"{_PS}.get_moe_tensor_parallel_world_size"),
    ("moe_tp_rank", f"{_PS}.get_moe_tensor_parallel_rank"),
    ("attn_tp_size", f"{_PS}.get_attn_tensor_model_parallel_world_size"),
    ("attn_tp_rank", f"{_PS}.get_attn_tensor_model_parallel_rank"),
    ("attn_cp_size", f"{_PS}.get_attn_context_model_parallel_world_size"),
    ("attn_cp_rank", f"{_PS}.get_attn_context_model_parallel_rank"),
    ("attn_dp_size", f"{_DP}.get_attention_dp_size"),
    ("attn_dp_rank", f"{_DP}.get_attention_dp_rank"),
]

GROUP_DELEGATIONS = [
    ("world_group", f"{_PS}.get_world_group"),
    ("tp_group", f"{_PS}.get_tp_group"),
    ("pp_group", f"{_PS}.get_pp_group"),
    ("moe_ep_group", f"{_PS}.get_moe_ep_group"),
    ("moe_dp_group", f"{_PS}.get_moe_dp_group"),
    ("moe_tp_group", f"{_PS}.get_moe_tp_group"),
    ("attn_tp_group", f"{_PS}.get_attn_tp_group"),
    ("attn_cp_group", f"{_PS}.get_attn_cp_group"),
]


class TestRuntimeContextSingletons(CustomTestCase):
    def test_singletons(self):
        self.assertIs(get_parallel(), get_parallel())
        self.assertIsInstance(get_parallel(), ParallelContext)
        self.assertIsInstance(get_context(), RuntimeContext)
        self.assertIs(get_context().parallel, get_parallel())


class _IsolatedOverrides(CustomTestCase):
    """Give each test a clean override map, restoring afterward only the overrides
    installed outside it (e.g. by another test file sharing the process)."""

    def setUp(self):
        super().setUp()
        p = get_parallel()
        self._saved_overrides = dict(p._overrides)
        p._overrides.clear()

    def tearDown(self):
        p = get_parallel()
        p._overrides.clear()
        p._overrides.update(self._saved_overrides)
        super().tearDown()


class TestParallelDelegation(_IsolatedOverrides):
    def test_size_rank_delegate_to_canonical_getters(self):
        # Patch each getter to a distinct sentinel: a miswired attribute would read
        # a different (unpatched) getter and fail.
        for i, (attr, target) in enumerate(SIZE_RANK_DELEGATIONS):
            sentinel = 1000 + i
            with patch(target, return_value=sentinel):
                self.assertEqual(
                    getattr(get_parallel(), attr),
                    sentinel,
                    msg=f"{attr} must delegate to {target}",
                )

    def test_groups_delegate_to_canonical_getters(self):
        for attr, target in GROUP_DELEGATIONS:
            sentinel = object()
            with patch(target, return_value=sentinel):
                self.assertIs(
                    getattr(get_parallel(), attr),
                    sentinel,
                    msg=f"{attr} must delegate to {target}",
                )

    def test_wrapper_holds_no_resolved_state(self):
        # __slots__: no __dict__; the only instance state is the override hook.
        self.assertFalse(hasattr(get_parallel(), "__dict__"))
        # tp_group IS exposed: live delegation handles PD-multiplexing / the tp patch.
        self.assertTrue(hasattr(ParallelContext, "tp_group"))
        # local_attn_dp is intentionally not part of the wrapper surface.
        self.assertFalse(hasattr(ParallelContext, "local_attn_dp_size"))


class TestParallelOverride(_IsolatedOverrides):
    def test_override_takes_precedence(self):
        p = get_parallel()
        with p.override(tp_size=99, tp_rank=3, attn_dp_size=8):
            self.assertEqual(p.tp_size, 99)
            self.assertEqual(p.tp_rank, 3)
            self.assertEqual(p.attn_dp_size, 8)
            # same singleton: a fresh get_parallel() sees the override too
            self.assertEqual(get_parallel().tp_size, 99)
        self.assertEqual(p._overrides, {})

    def test_override_can_force_groups(self):
        sentinel = object()
        with get_parallel().override(tp_group=sentinel):
            self.assertIs(get_parallel().tp_group, sentinel)

    def test_override_nests_and_restores(self):
        p = get_parallel()
        with p.override(tp_size=2):
            self.assertEqual(p.tp_size, 2)
            with p.override(tp_size=4, pp_size=2):
                self.assertEqual(p.tp_size, 4)
                self.assertEqual(p.pp_size, 2)
            self.assertEqual(p.tp_size, 2)
            self.assertNotIn("pp_size", p._overrides)

    def test_override_unknown_key_raises_and_does_not_mutate(self):
        p = get_parallel()
        with self.assertRaises(ValueError):
            with p.override(tp_sizee=1):  # typo
                pass
        self.assertEqual(p._overrides, {})


if __name__ == "__main__":
    unittest.main()
