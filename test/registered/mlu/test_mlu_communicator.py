import os
import socket
import unittest

import torch
import torch_mlu  # noqa: F401
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.test.ci.ci_register import register_mlu_ci
from sglang.test.test_utils import CustomTestCase

register_mlu_ci(est_time=60, suite="pr-test-mlu")


def _get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _run_mlu_communicator_worker(rank: int, world_size: int, port: int):
    import torch_mlu  # noqa: F401

    from sglang.srt.distributed.device_communicators.mlu_communicator import (
        MluCommunicator,
    )

    torch.mlu.set_device(rank)
    dist.init_process_group(
        backend="cncl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    try:
        group = dist.group.WORLD
        comm = MluCommunicator(group=group)
        assert not comm.disabled

        x = torch.full((2, 3), rank + 1, dtype=torch.float32, device="mlu")
        reduced = comm.all_reduce(x.clone())
        expected_sum = torch.full((2, 3), 3.0, dtype=torch.float32, device="mlu")
        torch.testing.assert_close(reduced.cpu(), expected_sum.cpu())

        gathered = comm.all_gather(x, dim=-1)
        expected_gather = torch.tensor(
            [[1, 1, 1, 2, 2, 2], [1, 1, 1, 2, 2, 2]],
            dtype=torch.float32,
            device="mlu",
        )
        torch.testing.assert_close(gathered.cpu(), expected_gather.cpu())
    finally:
        dist.destroy_process_group()


class TestMLUCommunicator(CustomTestCase):
    def test_all_reduce_and_all_gather_on_cncl(self):
        world_size = 2
        port = _get_free_port()
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(port))
        mp.spawn(
            _run_mlu_communicator_worker,
            args=(world_size, port),
            nprocs=world_size,
            join=True,
        )


if __name__ == "__main__":
    unittest.main()
