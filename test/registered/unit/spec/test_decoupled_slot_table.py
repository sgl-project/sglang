"""CPU unit tests for the decoupled-spec verifier-side slot table + landing plan.

``DecoupledSlotTable`` is the ``rid -> (pool_idx, generation)`` map the recv
daemon uses to route an incoming enumeration block to GPU seats;
``plan_landing`` is the pure-host routing that decides which rows land (and where)
vs which are dropped because their request has left. Both are torch-free, so
these tests exercise every routing outcome without a GPU or transport.
"""

import unittest

from sglang.srt.speculative.decoupled_slot_table import (
    DecoupledSlotTable,
    LandingPlan,
    PlannedWrite,
    SlotBinding,
    plan_landing,
)
from sglang.srt.speculative.decoupled_spec_io import DraftEnumerationBufferBatch
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _block(rids, bases, *, num_steps=2, fanout=2) -> DraftEnumerationBufferBatch:
    """Build a well-formed enumeration block; token payload is filler (routing
    only reads rids / base_committed_lens / batch_size)."""
    row_stride = (num_steps + 1) * fanout * num_steps
    tokens = tuple(range(len(rids) * row_stride))
    return DraftEnumerationBufferBatch(
        src_drafter_rank=0,
        dst_verifier_rank=0,
        num_steps=num_steps,
        fanout=fanout,
        rids=list(rids),
        base_committed_lens=list(bases),
        tokens=tokens,
    )


class TestDecoupledSlotTable(CustomTestCase):
    def test_assign_then_lookup(self):
        table = DecoupledSlotTable()
        table.assign("a", pool_idx=5, generation=1)
        self.assertEqual(table.lookup("a"), SlotBinding(pool_idx=5, generation=1))
        self.assertEqual(len(table), 1)

    def test_lookup_missing_returns_none(self):
        table = DecoupledSlotTable()
        self.assertIsNone(table.lookup("nope"))

    def test_remove_then_lookup_none(self):
        table = DecoupledSlotTable()
        table.assign("a", pool_idx=5, generation=1)
        table.remove("a")
        self.assertIsNone(table.lookup("a"))
        self.assertEqual(len(table), 0)

    def test_remove_absent_is_noop(self):
        table = DecoupledSlotTable()
        table.remove("ghost")  # must not raise
        self.assertEqual(len(table), 0)

    def test_reassign_overwrites(self):
        # Re-open after retraction: same rid, new seat + generation, last wins.
        table = DecoupledSlotTable()
        table.assign("a", pool_idx=5, generation=1)
        table.assign("a", pool_idx=8, generation=2)
        self.assertEqual(table.lookup("a"), SlotBinding(pool_idx=8, generation=2))
        self.assertEqual(len(table), 1)

    def test_int_coercion(self):
        table = DecoupledSlotTable()
        table.assign("a", pool_idx=True, generation=False)  # bool -> int
        binding = table.lookup("a")
        self.assertEqual(binding.pool_idx, 1)
        self.assertEqual(binding.generation, 0)


class TestPlanLanding(CustomTestCase):
    def test_all_hit(self):
        table = DecoupledSlotTable()
        table.assign("a", pool_idx=3, generation=7)
        table.assign("b", pool_idx=4, generation=9)
        plan = plan_landing(_block(["a", "b"], [10, 20]), table)

        self.assertEqual(plan.dropped_rids, [])
        self.assertEqual(
            plan.writes,
            [
                PlannedWrite(
                    row_index=0, pool_idx=3, generation=7, base_committed_len=10
                ),
                PlannedWrite(
                    row_index=1, pool_idx=4, generation=9, base_committed_len=20
                ),
            ],
        )

    def test_all_dropped_when_table_empty(self):
        table = DecoupledSlotTable()
        plan = plan_landing(_block(["a", "b"], [10, 20]), table)
        self.assertEqual(plan.writes, [])
        self.assertEqual(plan.dropped_rids, ["a", "b"])

    def test_mixed_hit_and_drop(self):
        table = DecoupledSlotTable()
        table.assign("b", pool_idx=4, generation=9)  # only b is live
        plan = plan_landing(_block(["a", "b", "c"], [10, 20, 30]), table)

        self.assertEqual(plan.dropped_rids, ["a", "c"])
        self.assertEqual(
            plan.writes,
            [
                PlannedWrite(
                    row_index=1, pool_idx=4, generation=9, base_committed_len=20
                )
            ],
        )

    def test_row_index_tracks_original_position(self):
        # A dropped leading row must not shift the surviving row's row_index:
        # row_index must always point back into the block's own rows.
        table = DecoupledSlotTable()
        table.assign("b", pool_idx=4, generation=9)
        plan = plan_landing(_block(["a", "b"], [10, 20]), table)
        self.assertEqual(plan.writes[0].row_index, 1)

    def test_stamp_carries_block_base_and_table_generation(self):
        # base comes from the block (freshness), generation from the table
        # (occupancy) -- they must not be crossed.
        table = DecoupledSlotTable()
        table.assign("a", pool_idx=3, generation=7)
        plan = plan_landing(_block(["a"], [42]), table)
        write = plan.writes[0]
        self.assertEqual(write.base_committed_len, 42)
        self.assertEqual(write.generation, 7)
        self.assertEqual(write.pool_idx, 3)

    def test_empty_block(self):
        table = DecoupledSlotTable()
        plan = plan_landing(_block([], []), table)
        self.assertEqual(plan, LandingPlan(writes=[], dropped_rids=[]))


if __name__ == "__main__":
    unittest.main()
