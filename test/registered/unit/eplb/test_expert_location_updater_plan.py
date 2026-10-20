"""Unit tests for the expert location updater's planner and executors; no server, no process group."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

import unittest

import torch
import torch.distributed

from sglang.srt.eplb import expert_location_updater as updater
from sglang.test.test_utils import CustomTestCase

_NUM_TENSORS = 2
_NUM_LOCAL = 2
_NUM_PHYSICAL = 4  # two ranks


def _weights(physical_to_logical_map, rank):
    """One row per local expert, filled with its logical id."""
    lo = rank * _NUM_LOCAL
    return [
        torch.tensor(
            [[float(physical_to_logical_map[lo + i])] * 3 for i in range(_NUM_LOCAL)]
        )
        for _ in range(_NUM_TENSORS)
    ]


def _p2p_entries(plan):
    """The planner lists every local logical expert; keep those with ops."""
    return [(eid, ops) for eid, ops in plan.p2p_op_infos if ops]


def _plan(old, new, rank, weights, buffers):
    return updater.plan_expert_weights_single_layer(
        routed_experts_weights=weights,
        temp_buffers=buffers,
        old_physical_to_logical_map=old,
        new_physical_to_logical_map=new,
        num_local_physical_experts=_NUM_LOCAL,
        num_gpu_per_node=2,
        rank=rank,
        debug=True,
    )


class TestExpertLocationUpdaterPlan(CustomTestCase):
    def test_plan_moves_nothing_when_map_is_unchanged(self):
        old = [0, 1, 2, 0]
        weights = _weights(old, rank=0)
        plan = _plan(old, old, 0, weights, updater.create_temp_buffers(weights))
        self.assertEqual(plan.weight2buffer_copy_infos, [])
        self.assertEqual(_p2p_entries(plan), [])
        self.assertEqual(plan.buffer2weight_copy_infos, [])

    def test_plan_touches_no_memory_and_lists_each_move(self):
        old = [0, 1, 2, 0]
        new = [1, 0, 2, 1]

        weights0 = _weights(old, rank=0)
        buffers0 = updater.create_temp_buffers(weights0)
        snapshot = [w.clone() for w in weights0]
        buffers_snapshot = [b.clone() for b in buffers0]
        plan0 = _plan(old, new, 0, weights0, buffers0)
        for before, after in zip(snapshot, weights0, strict=True):
            self.assertTrue(torch.equal(before, after))
        for before, after in zip(buffers_snapshot, buffers0, strict=True):
            self.assertTrue(torch.equal(before, after))

        # rank 0 swaps its two experts on the GPU and sends logical 1 to rank 1
        self.assertEqual(plan0.weight2buffer_copy_infos, [(1, 0), (0, 1)])
        self.assertEqual(plan0.buffer2weight_copy_infos, [(0, 0), (1, 1)])
        self.assertEqual([eid for eid, _ in _p2p_entries(plan0)], [1])
        sends = _p2p_entries(plan0)[0][1]
        self.assertEqual(len(sends), _NUM_TENSORS)
        self.assertTrue(all(op.op is torch.distributed.isend for op in sends))
        self.assertTrue(all(op.peer == 1 for op in sends))

        # rank 1 keeps logical 2 and receives logical 1 from rank 0
        weights1 = _weights(old, rank=1)
        plan1 = _plan(old, new, 1, weights1, updater.create_temp_buffers(weights1))
        self.assertEqual(plan1.weight2buffer_copy_infos, [])
        self.assertEqual(plan1.buffer2weight_copy_infos, [(3, 3)])
        self.assertEqual([eid for eid, _ in _p2p_entries(plan1)], [1])
        recvs = _p2p_entries(plan1)[0][1]
        self.assertTrue(all(op.op is torch.distributed.irecv for op in recvs))
        self.assertTrue(all(op.peer == 0 for op in recvs))

    def test_plan_needs_no_process_group(self):
        old = [0, 1, 2, 0]
        new = [1, 0, 2, 1]
        weights = _weights(old, rank=1)
        self.assertFalse(torch.distributed.is_initialized())
        plan = _plan(old, new, 1, weights, updater.create_temp_buffers(weights))
        descs = [d for _, ds in plan.p2p_op_infos for d in ds]
        self.assertTrue(all(isinstance(d, updater.P2PDesc) for d in descs))
        self.assertEqual({d.peer for d in descs}, {0})

    def test_executors_apply_local_moves(self):
        old = [0, 1, 2, 0]
        new = [1, 0, 2, 1]
        weights = _weights(old, rank=0)
        buffers = updater.create_temp_buffers(weights)
        plan = _plan(old, new, 0, weights, buffers)

        updater.execute_weight2buffer_copies(plan, weights, buffers)
        for before, after in zip(_weights(old, rank=0), weights, strict=True):
            self.assertTrue(torch.equal(before, after))

        updater.execute_buffer2weight_copies(plan, weights, buffers)
        for expected, actual in zip(_weights(new, rank=0), weights, strict=True):
            self.assertTrue(torch.equal(expected, actual))

    def test_single_rank_update_matches_target_map(self):
        old = [0, 1]
        new = [1, 0]
        weights = _weights(old, rank=0)
        logs = updater.update_expert_weights_single_layer(
            routed_experts_weights=weights,
            temp_buffers=updater.create_temp_buffers(weights),
            old_physical_to_logical_map=old,
            new_physical_to_logical_map=new,
            num_local_physical_experts=_NUM_LOCAL,
            num_gpu_per_node=1,
            rank=0,
            debug=True,
        )
        for expected, actual in zip(_weights(new, rank=0), weights, strict=True):
            self.assertTrue(torch.equal(expected, actual))
        self.assertTrue(any("same-gpu" in line for line in logs))


if __name__ == "__main__":
    unittest.main()
