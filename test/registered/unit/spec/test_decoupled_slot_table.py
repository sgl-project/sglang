"""CPU unit tests for the decoupled-spec verifier-side slot table + landing plan.

DecoupledSlotTable is the rid -> pool_idx map the recv daemon uses to route an
incoming enumeration block to GPU seats; plan_landing is the pure-host routing
that decides which rows land (and where) vs which are dropped because their
request has left. Both are torch-free, so these tests exercise every routing
outcome without a GPU or transport.
"""

import unittest

from sglang.srt.speculative.decoupled_slot_table import (
    DecoupledSlotTable,
    LandingPlan,
    PlannedWrite,
    plan_landing,
)
from sglang.srt.speculative.decoupled_spec_io import DraftEnumerationBufferBatch
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _block(
    rids, bases, *, num_steps=2, fanout=2, dst_verifier_rank=0
) -> DraftEnumerationBufferBatch:
    """Build a well-formed enumeration block; token payload is filler (routing
    only reads rids / base_committed_lens / batch_size)."""
    row_stride = (num_steps + 1) * fanout * num_steps
    tokens = tuple(range(len(rids) * row_stride))
    return DraftEnumerationBufferBatch(
        src_drafter_rank=0,
        dst_verifier_rank=dst_verifier_rank,
        num_steps=num_steps,
        fanout=fanout,
        rids=list(rids),
        base_committed_lens=list(bases),
        tokens=tokens,
    )


class TestDecoupledSlotTable(CustomTestCase):
    def test_assign_then_lookup(self):
        table = DecoupledSlotTable()
        table.assign("a", pool_idx=5)
        self.assertEqual(table.lookup("a"), 5)
        self.assertEqual(len(table), 1)

    def test_lookup_missing_returns_none(self):
        table = DecoupledSlotTable()
        self.assertIsNone(table.lookup("nope"))

    def test_remove_then_lookup_none(self):
        table = DecoupledSlotTable()
        table.assign("a", pool_idx=5)
        table.remove("a")
        self.assertIsNone(table.lookup("a"))
        self.assertEqual(len(table), 0)

    def test_remove_absent_is_noop(self):
        table = DecoupledSlotTable()
        table.remove("ghost")  # must not raise
        self.assertEqual(len(table), 0)

    def test_reassign_overwrites(self):
        # Re-open after retraction: same rid, new seat, last write wins.
        table = DecoupledSlotTable()
        table.assign("a", pool_idx=5)
        table.assign("a", pool_idx=8)
        self.assertEqual(table.lookup("a"), 8)
        self.assertEqual(len(table), 1)

    def test_clear(self):
        # clear() is the cache-flush hook: after it, no binding may survive
        # (a stale seat would route a late block into a reused seat post-flush).
        table = DecoupledSlotTable()
        table.assign("a", pool_idx=5)
        table.assign("b", pool_idx=6)
        table.clear()
        self.assertIsNone(table.lookup("a"))
        self.assertIsNone(table.lookup("b"))
        self.assertEqual(len(table), 0)

    def test_lookup_many_aligned_with_input(self):
        # Positionally aligned with the input rids, None where unbound -- the
        # daemon routes a whole block against this one snapshot.
        table = DecoupledSlotTable()
        table.assign("a", pool_idx=3)
        table.assign("c", pool_idx=5)
        self.assertEqual(table.lookup_many(["a", "b", "c"]), [3, None, 5])


class TestPlanLanding(CustomTestCase):
    def test_all_hit(self):
        table = DecoupledSlotTable()
        table.assign("a", pool_idx=3)
        table.assign("b", pool_idx=4)
        plan = plan_landing(_block(["a", "b"], [10, 20]), table, verifier_rank=0)

        self.assertEqual(plan.dropped_rids, [])
        self.assertEqual(
            plan.writes,
            [
                PlannedWrite(row_index=0, pool_idx=3, base_committed_len=10),
                PlannedWrite(row_index=1, pool_idx=4, base_committed_len=20),
            ],
        )

    def test_all_dropped_when_table_empty(self):
        table = DecoupledSlotTable()
        plan = plan_landing(_block(["a", "b"], [10, 20]), table, verifier_rank=0)
        self.assertEqual(plan.writes, [])
        self.assertEqual(plan.dropped_rids, ["a", "b"])

    def test_mixed_hit_and_drop(self):
        table = DecoupledSlotTable()
        table.assign("b", pool_idx=4)  # only b is live
        plan = plan_landing(
            _block(["a", "b", "c"], [10, 20, 30]), table, verifier_rank=0
        )

        self.assertEqual(plan.dropped_rids, ["a", "c"])
        self.assertEqual(
            plan.writes,
            [PlannedWrite(row_index=1, pool_idx=4, base_committed_len=20)],
        )

    def test_row_index_tracks_original_position(self):
        # A dropped leading row must not shift the surviving row's row_index:
        # row_index must always point back into the block's own rows.
        table = DecoupledSlotTable()
        table.assign("b", pool_idx=4)
        plan = plan_landing(_block(["a", "b"], [10, 20]), table, verifier_rank=0)
        self.assertEqual(plan.writes[0].row_index, 1)

    def test_base_committed_len_from_block(self):
        # base_committed_len comes from the block; pool_idx from the table.
        table = DecoupledSlotTable()
        table.assign("a", pool_idx=3)
        plan = plan_landing(_block(["a"], [42]), table, verifier_rank=0)
        self.assertEqual(
            plan.writes[0],
            PlannedWrite(row_index=0, pool_idx=3, base_committed_len=42),
        )

    def test_empty_block(self):
        table = DecoupledSlotTable()
        plan = plan_landing(_block([], []), table, verifier_rank=0)
        self.assertEqual(plan, LandingPlan(writes=[], dropped_rids=[]))

    def test_rank_mismatch_raises(self):
        # A block routed to a different verifier must be rejected before any
        # routing: rids are only unique within the owning verifier, so a foreign
        # rid colliding with a live local rid would land into the local seat.
        table = DecoupledSlotTable()
        table.assign("a", pool_idx=3)
        with self.assertRaises(RuntimeError):
            plan_landing(
                _block(["a"], [10], dst_verifier_rank=1), table, verifier_rank=0
            )

    def test_malformed_block_raises(self):
        # Malformed wire input (parallel arrays out of sync) must be caught on
        # the ingest path via block.validate() -- an uncaught raise here kills
        # the recv daemon loop, stalling landing for ALL requests.
        table = DecoupledSlotTable()
        table.assign("a", pool_idx=3)
        row_stride = (2 + 1) * 2 * 2
        malformed = DraftEnumerationBufferBatch(
            src_drafter_rank=0,
            dst_verifier_rank=0,
            num_steps=2,
            fanout=2,
            rids=["a", "b"],
            base_committed_lens=[10],  # length mismatch vs rids
            tokens=tuple(range(2 * row_stride)),
        )
        with self.assertRaises(ValueError):
            plan_landing(malformed, table, verifier_rank=0)

    def test_duplicate_rid_raises(self):
        # Duplicate rids resolve to the same seat, making the scatter's winning
        # row/stamp nondeterministic. Rejected by block.validate() (ValueError).
        table = DecoupledSlotTable()
        table.assign("a", pool_idx=3)
        with self.assertRaises(ValueError):
            plan_landing(_block(["a", "a"], [10, 20]), table, verifier_rank=0)


if __name__ == "__main__":
    unittest.main()
