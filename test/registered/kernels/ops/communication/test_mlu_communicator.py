import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_mlu  # noqa: F401

from sglang.test.ci.ci_register import register_mlu_ci
from sglang.test.test_utils import CustomTestCase

register_mlu_ci(est_time=60, suite="pr-test-2-mlu")


def _run_mlu_communicator_worker(rank: int, world_size: int, init_method: str):

    from sglang.srt.distributed.parallel_state import init_model_parallel_group

    torch.mlu.set_device(rank)
    dist.init_process_group(
        backend="cncl",
        init_method=init_method,
        timeout=timedelta(seconds=60),
        rank=rank,
        world_size=world_size,
    )
    comm = None
    try:
        comm = init_model_parallel_group(
            group_ranks=[list(range(world_size))],
            local_rank=rank,
            backend="cncl",
            use_custom_allreduce=False,
            use_mscclpp=False,
            use_torch_symm_mem_allreduce=False,
        )
        assert comm.mlu_communicator is not None
        assert not comm.mlu_communicator.disabled

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
        if comm is not None:
            comm.destroy()
        dist.destroy_process_group()


class TestMLUCommunicator(CustomTestCase):
    def test_all_reduce_and_all_gather_on_cncl(self):
        world_size = 2
        self.assertGreaterEqual(torch.mlu.device_count(), world_size)
        with tempfile.TemporaryDirectory() as directory:
            init_method = (Path(directory) / "rendezvous").as_uri()
            mp.spawn(
                _run_mlu_communicator_worker,
                args=(world_size, init_method),
                nprocs=world_size,
                join=True,
            )


if __name__ == "__main__":
    unittest.main()
