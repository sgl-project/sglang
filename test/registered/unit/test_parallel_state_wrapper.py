import dataclasses
import unittest

from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


def _make_default(**overrides):
    base = dict(
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        dp_rank=None,
        dp_size=1,
        attn_tp_rank=0,
        attn_tp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        attn_dp_rank=0,
        attn_dp_size=1,
        moe_ep_rank=0,
        moe_ep_size=1,
        moe_dp_rank=None,
        moe_dp_size=1,
        gpu_id=0,
    )
    base.update(overrides)
    return ParallelState(**base)


class TestParallelStateConstruction(CustomTestCase):
    def test_minimal_single_rank(self):
        state = _make_default()
        self.assertEqual(state.tp_rank, 0)
        self.assertEqual(state.tp_size, 1)
        self.assertEqual(state.gpu_id, 0)
        self.assertIsNone(state.dp_rank)
        self.assertIsNone(state.moe_dp_rank)

    def test_multi_rank_full(self):
        state = _make_default(
            tp_rank=3,
            tp_size=8,
            pp_rank=1,
            pp_size=2,
            dp_rank=2,
            dp_size=4,
            attn_tp_rank=3,
            attn_tp_size=4,
            attn_cp_rank=0,
            attn_cp_size=1,
            attn_dp_rank=2,
            attn_dp_size=2,
            moe_ep_rank=1,
            moe_ep_size=2,
            moe_dp_rank=2,
            moe_dp_size=4,
            gpu_id=3,
        )
        self.assertEqual(state.tp_rank, 3)
        self.assertEqual(state.dp_rank, 2)
        self.assertEqual(state.moe_dp_rank, 2)
        self.assertEqual(state.attn_dp_size, 2)


class TestParallelStateImmutability(CustomTestCase):
    def test_frozen_rejects_mutation(self):
        state = _make_default()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            state.tp_rank = 9  # type: ignore[misc]

    def test_slots_rejects_new_attribute(self):
        state = _make_default()
        with self.assertRaises((AttributeError, dataclasses.FrozenInstanceError)):
            state.extra_field = 1  # type: ignore[attr-defined]


class TestParallelStateKeywordOnly(CustomTestCase):
    def test_positional_args_rejected(self):
        with self.assertRaises(TypeError):
            ParallelState(0, 1)  # type: ignore[call-arg]


if __name__ == "__main__":
    unittest.main()
