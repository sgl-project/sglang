from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

import os
import time
import traceback
import unittest

import torch
import torch.distributed
import torch.multiprocessing as mp

from sglang.srt.eplb import expert_location_updater as updater
from sglang.test.test_utils import CustomTestCase, find_available_port

_WORLD_SIZE = 2
_NUM_LOCAL = 2
_NUM_LOGICAL = 3
_NUM_LAYERS = 2
_OLD_MAPS = [[0, 1, 2, 0], [2, 1, 0, 1]]
_NEW_MAPS = [[1, 0, 2, 1], [0, 2, 1, 1]]


def _weights_for(physical_to_logical_map, rank):
    lo = rank * _NUM_LOCAL
    return [
        torch.tensor(
            [[float(physical_to_logical_map[lo + i])] * 4 for i in range(_NUM_LOCAL)]
        )
        for _ in range(2)
    ]


def _run_rank(rank, master_port, writer):
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
            weights_of_layer = {
                layer_id: _weights_for(_OLD_MAPS[layer_id], rank)
                for layer_id in range(_NUM_LAYERS)
            }

            migrator = AsyncExpertMigrator()
            job = migrator.start(
                logical_count=torch.zeros(
                    (_NUM_LAYERS, _NUM_LOGICAL), dtype=torch.int32
                ),
                old_metadata=old_metadata,
                routed_experts_weights_of_layer=weights_of_layer,
                compute_placement=lambda count: placement,
            )
            deadline = time.time() + 60
            while not migrator.poll_placement(job):
                assert time.time() < deadline, "placement never became ready"
                time.sleep(0.01)
            migrator.build_metadata(job)

            for layer_id in job.layer_ids:
                migrator.issue_layer(job, layer_id)
                for expected, actual in zip(
                    _weights_for(_OLD_MAPS[layer_id], rank),
                    weights_of_layer[layer_id],
                    strict=True,
                ):
                    assert torch.equal(expected, actual), "install happened early"
                migrator.install_layer(job, layer_id)
                torch.cpu.synchronize()
                for expected, actual in zip(
                    _weights_for(_NEW_MAPS[layer_id], rank),
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
            migrator.cancel(job)
        torch.distributed.barrier()
        writer.send(True)
    except Exception:
        traceback.print_exc()
        writer.send(False)


def _single_rank_weights(physical_to_logical_map):
    return [
        torch.tensor([[float(e)] * 3 for e in physical_to_logical_map[:_NUM_LOCAL]])
        for _ in range(2)
    ]


class TestExpertLocationUpdaterPlan(CustomTestCase):
    def test_plan_issues_no_device_op(self):
        old = [0, 1]
        new = [1, 0]
        weights = _single_rank_weights(old)
        buffers = updater.create_temp_buffers(weights)
        weights_before = [w.clone() for w in weights]
        buffers_before = [b.clone() for b in buffers]
        plan = updater.plan_expert_weights_single_layer(
            routed_experts_weights=weights,
            temp_buffers=buffers,
            old_physical_to_logical_map=old,
            new_physical_to_logical_map=new,
            num_local_physical_experts=_NUM_LOCAL,
            num_gpu_per_node=1,
            rank=0,
        )
        self.assertTrue(plan.weight2buffer_copy_infos)
        for before, after in zip(weights_before, weights, strict=True):
            self.assertTrue(torch.equal(before, after))
        for before, after in zip(buffers_before, buffers, strict=True):
            self.assertTrue(torch.equal(before, after))

    def test_synchronous_update_reaches_target_map(self):
        old = [0, 1]
        new = [1, 0]
        weights = _single_rank_weights(old)
        updater.update_expert_weights_single_layer(
            routed_experts_weights=weights,
            temp_buffers=updater.create_temp_buffers(weights),
            old_physical_to_logical_map=old,
            new_physical_to_logical_map=new,
            num_local_physical_experts=_NUM_LOCAL,
            num_gpu_per_node=1,
            rank=0,
        )
        for expected, actual in zip(_single_rank_weights(new), weights, strict=True):
            self.assertTrue(torch.equal(expected, actual))


class TestEplbAsyncMigrator(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        mp.set_start_method("spawn", force=True)

    def test_two_ranks_migrate_layer_by_layer(self):
        master_port = find_available_port(23456)
        reader, writer = mp.Pipe(duplex=False)
        processes = [
            mp.Process(target=_run_rank, args=(rank, master_port, writer))
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
