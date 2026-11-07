from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

import os
import time
import traceback
import unittest

import torch
import torch.distributed
import torch.multiprocessing as mp

from sglang.srt.eplb.eplb_async_migrator import plan_layer_pull
from sglang.test.test_utils import CustomTestCase, find_available_port

_WORLD_SIZE = 2
_NUM_LOCAL = 2
_NUM_LOGICAL = 3
_NUM_LAYERS = 2
_OLD_MAPS = [[0, 1, 2, 0], [2, 1, 0, 1]]
_NEW_MAPS = [[1, 0, 2, 1], [0, 2, 1, 1]]
_ROW = 4


def _fill(tensor, physical_to_logical_map, rank):
    lo = rank * _NUM_LOCAL
    for i in range(_NUM_LOCAL):
        tensor[i] = float(physical_to_logical_map[lo + i])


def _expected(physical_to_logical_map, rank):
    t = torch.empty(_NUM_LOCAL, _ROW)
    _fill(t, physical_to_logical_map, rank)
    return t


class _SharedMemoryTransport:
    def __init__(self, shared, rank, read_delay):
        self._shared = shared
        self._rank = rank
        self._read_delay = read_delay
        self.reads = []

    def prepare(self, routed_experts_weights_of_layer, staging):
        pass

    def read_experts(self, src_rank, layer_id, slot_pairs, staging):
        time.sleep(self._read_delay)
        for src_slot, dst_slot in slot_pairs:
            for tensor_index, buffer in enumerate(staging):
                buffer[dst_slot].copy_(
                    self._shared[src_rank][layer_id][tensor_index][src_slot]
                )
            self.reads.append((src_rank, layer_id, src_slot, dst_slot))


def _run_rank(rank, master_port, shared, writer):
    try:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(master_port)
        torch.distributed.init_process_group(
            backend="gloo", rank=rank, world_size=_WORLD_SIZE
        )

        from sglang.srt.eplb.eplb_async_migrator import AsyncExpertMigrator
        from sglang.srt.eplb.expert_location import (
            EplbPlacement,
            ExpertLocationMetadata,
            _compute_logical_to_all_physical_map,
        )
        from sglang.srt.runtime_context import get_context, get_parallel

        def metadata_for(maps):
            physical_to_logical_map = torch.tensor(maps, dtype=torch.int64)
            return ExpertLocationMetadata._init_raw(
                ep_size=_WORLD_SIZE,
                physical_to_logical_map=physical_to_logical_map,
                logical_to_all_physical_map=_compute_logical_to_all_physical_map(
                    physical_to_logical_map,
                    num_logical_experts=_NUM_LOGICAL,
                    ep_size=_WORLD_SIZE,
                    moe_ep_rank=None,
                ),
            )

        with (
            get_context().override_server_args(
                device="cpu",
                ep_size=_WORLD_SIZE,
                tp_size=_WORLD_SIZE,
                nnodes=1,
                ep_dispatch_algorithm=None,
            ),
            get_parallel().override(tp_rank=rank, tp_size=_WORLD_SIZE),
        ):
            weights_of_layer = shared[rank]
            old_metadata = metadata_for(_OLD_MAPS)
            new_physical_to_logical_map = torch.tensor(_NEW_MAPS, dtype=torch.int64)
            placement = EplbPlacement(
                physical_to_logical_map=new_physical_to_logical_map,
                logical_to_all_physical_map=_compute_logical_to_all_physical_map(
                    new_physical_to_logical_map,
                    num_logical_experts=_NUM_LOGICAL,
                    ep_size=_WORLD_SIZE,
                    moe_ep_rank=None,
                ),
                logical_to_rank_dispatch_physical_map=None,
            )

            transport = _SharedMemoryTransport(
                shared, rank, read_delay=0.5 if rank == 1 else 0.0
            )
            migrator = AsyncExpertMigrator(transport)
            migrator.prepare(weights_of_layer, cpu_group=torch.distributed.group.WORLD)
            job = migrator.start(
                logical_count=torch.zeros(
                    (_NUM_LAYERS, _NUM_LOGICAL), dtype=torch.int32
                ),
                old_metadata=old_metadata,
                routed_experts_weights_of_layer=weights_of_layer,
                compute_placement=lambda count: placement,
            )
            passes = 0
            while not migrator.placement_ready(job):
                passes += 1
                assert passes < 2000, "placement never agreed ready"
                time.sleep(0.005)
            migrator.build_metadata(job)

            polls_per_layer = []
            for layer_id in job.layer_ids:
                migrator.issue_layer(job, layer_id)
                polls = 0
                while not migrator.layer_ready(job, layer_id):
                    polls += 1
                    assert polls < 2000, "layer never agreed ready"
                    for expected, actual in zip(
                        [_expected(_OLD_MAPS[layer_id], rank)] * 2,
                        weights_of_layer[layer_id],
                        strict=True,
                    ):
                        assert torch.equal(expected, actual), "install happened early"
                    time.sleep(0.005)
                polls_per_layer.append(polls)
                migrator.install_layer(job, layer_id)
                for expected, actual in zip(
                    [_expected(_NEW_MAPS[layer_id], rank)] * 2,
                    weights_of_layer[layer_id],
                    strict=True,
                ):
                    assert torch.equal(expected, actual), (
                        f"rank {rank} layer {layer_id}: {actual} != {expected}"
                    )
                assert (
                    old_metadata.physical_to_logical_map_cpu[layer_id].tolist()
                    == _NEW_MAPS[layer_id]
                )
                for later in job.layer_ids[job.layers_installed :]:
                    assert (
                        old_metadata.physical_to_logical_map_cpu[later].tolist()
                        == _OLD_MAPS[later]
                    ), "a later layer's map changed before its install"

            assert job.layers_installed == _NUM_LAYERS
            assert all(p >= 1 for p in polls_per_layer), polls_per_layer
            if rank == 1:
                assert transport.reads, "rank 1 pulls logical 1 from rank 0"
                assert all(src == 0 for src, *_ in transport.reads), transport.reads
            migrator.cancel(job)
        torch.distributed.barrier()
        writer.send(True)
    except Exception:
        traceback.print_exc()
        writer.send(False)


class TestPlanLayerPull(CustomTestCase):
    def test_same_gpu_then_free_rider_then_nearest_holder(self):
        old = [0, 1, 2, 3, 4, 5, 6, 7]
        new = [1, 0, 4, 4, 0, 0, 2, 7]
        plan0 = plan_layer_pull(old, new, 2, 2, rank=0)
        self.assertEqual(plan0.weight2buffer_copies, [(1, 0), (0, 1)])
        self.assertEqual(plan0.reads, {})
        plan1 = plan_layer_pull(old, new, 2, 2, rank=1)
        self.assertEqual(plan1.reads, {2: [(0, 0)]})
        self.assertEqual(plan1.buffer2weight_copies, [(0, 0), (0, 1)])
        plan3 = plan_layer_pull(old, new, 2, 2, rank=3)
        self.assertEqual(plan3.reads, {1: [(0, 0)]})
        plan2 = plan_layer_pull(old, new, 2, 2, rank=2)
        self.assertEqual(plan2.reads, {0: [(0, 0)]})
        self.assertEqual(plan2.buffer2weight_copies, [(0, 0), (0, 1)])


class TestEplbAsyncMigrator(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        mp.set_start_method("spawn", force=True)

    def test_two_ranks_pull_and_install_layer_by_layer(self):
        shared = {}
        for rank in range(_WORLD_SIZE):
            shared[rank] = {}
            for layer_id in range(_NUM_LAYERS):
                tensors = []
                for _ in range(2):
                    t = torch.empty(_NUM_LOCAL, _ROW)
                    _fill(t, _OLD_MAPS[layer_id], rank)
                    tensors.append(t.share_memory_())
                shared[rank][layer_id] = tensors

        master_port = find_available_port(23456)
        reader, writer = mp.Pipe(duplex=False)
        processes = [
            mp.Process(target=_run_rank, args=(rank, master_port, shared, writer))
            for rank in range(_WORLD_SIZE)
        ]
        for p in processes:
            p.start()
        for _ in range(_WORLD_SIZE):
            self.assertTrue(reader.recv(), "a rank failed; see its traceback above")
        for p in processes:
            p.join()


if __name__ == "__main__":
    unittest.main()
