"""Real Gloo PP transfers, including TP shard reassembly across four ranks."""

import tempfile
import traceback
import unittest
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.srt.distributed.parallel_state import GroupCoordinator
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=35, suite="base-a-test-cpu")


def _coordinator(ranks, rank, group):
    coordinator = GroupCoordinator.__new__(GroupCoordinator)
    coordinator.ranks = ranks
    coordinator.world_size = len(ranks)
    coordinator.rank_in_group = ranks.index(rank)
    coordinator.rank = rank
    coordinator.cpu_group = group
    coordinator.device_group = group
    coordinator.device = torch.device("cpu")
    coordinator.pp_cpu_staging = True
    coordinator.pynccl_comm = None
    coordinator.mq_broadcaster = None
    return coordinator


class _TPGather:
    def __init__(self, ranks, rank, group):
        self.world_size = len(ranks)
        self.rank_in_group = ranks.index(rank)
        self.group = group
        self.calls = 0

    def all_gather(self, tensor, dim):
        self.calls += 1
        shards = [torch.empty_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(shards, tensor, group=self.group)
        return torch.cat(shards, dim=dim)


def _payload(offset=0):
    return {
        "noncontiguous": torch.arange(24).reshape(4, 6).t() + offset,
        "odd": torch.arange(5, dtype=torch.int32) + offset,
        "empty": torch.empty((2, 0, 3), dtype=torch.bfloat16),
        "scalar": torch.tensor(7 + offset),
        "after_empty": torch.arange(8, dtype=torch.bfloat16).reshape(2, 4) + offset,
        "control": "preserved",
    }


def _assert_payload(actual, expected):
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(actual[key], value)
        else:
            assert actual[key] == value


def _exercise_gloo(rank, init_method):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=init_method,
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=30),
    )
    try:
        for ranks in ([0, 2], [1, 3]):
            group = dist.new_group(ranks, backend="gloo", timeout=timedelta(seconds=30))
            if rank in ranks:
                pp = _coordinator(ranks, rank, group)
        for ranks in ([0, 1], [2, 3]):
            group = dist.new_group(ranks, backend="gloo", timeout=timedelta(seconds=30))
            if rank in ranks:
                tp = _TPGather(ranks, rank, group)
        if rank < 2:
            payload = _payload()
            works = pp.send_tensor_dict(
                payload, dst=1, all_gather_group=tp, async_send=True
            )
            # CPU input is deliberately mutated before waiting: send owns a snapshot.
            for value in payload.values():
                if isinstance(value, torch.Tensor):
                    value.fill_(-1)
            for work in works:
                work.work.wait()
        else:
            _assert_payload(pp.recv_tensor_dict(src=0, all_gather_group=tp), _payload())
            assert tp.calls == 2
        dist.barrier()
        # Bidirectional exchange must post both directions without a send/send deadlock.
        received = pp.send_recv_tensor_dict(
            _payload(rank), send_dst=1 - pp.rank_in_group, recv_src=1 - pp.rank_in_group
        )
        _assert_payload(received, _payload((rank + 2) % 4))
        dist.barrier()
        # Broadcast from a nonzero PP rank, including object-list routing.
        objects = ["gloo", 7] if rank >= 2 else [None, None]
        pp.broadcast_object_list(objects, src=1)
        assert objects == ["gloo", 7]
        tensor = torch.tensor(42 if rank >= 2 else -1)
        pp.broadcast(tensor, src=1)
        assert tensor.item() == 42
        if rank < 2:
            pp.send(torch.tensor([13], dtype=torch.bfloat16), dst=1)
            a = pp.send_object("first", dst=1, tag=17, async_send=True)
            b = pp.send_object("second", dst=1, tag=23, async_send=True)
            for work in a + b:
                work.work.wait()
        else:
            torch.testing.assert_close(
                pp.recv((1,), torch.bfloat16, src=0),
                torch.tensor([13], dtype=torch.bfloat16),
            )
            assert pp.recv_object(src=0, tag=23) == "second"
            assert pp.recv_object(src=0, tag=17) == "first"
    except BaseException:
        # Preserve the original rank failure before peers report connection errors.
        traceback.print_exc()
        raise
    finally:
        dist.destroy_process_group()


class TestPPGlooTransport(CustomTestCase):
    def test_real_gloo_transfers_and_tp_reassembly(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(
                _exercise_gloo,
                args=(Path(directory, "store").as_uri(),),
                nprocs=4,
                join=True,
            )


if __name__ == "__main__":
    unittest.main()
