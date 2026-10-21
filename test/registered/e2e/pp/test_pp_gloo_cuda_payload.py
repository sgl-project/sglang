"""CUDA payloads must traverse a CPU-only Gloo PP boundary."""

import tempfile
import traceback
import unittest
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.srt.distributed.parallel_state import GroupCoordinator
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=45, stage="base-b", runner_config="2-gpu-large")


def _run_host_staged_pp(rank, init_method):
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=init_method,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=20),
    )
    try:
        # Real Gloo groups; only unrelated coordinator initialization is omitted.
        pp = GroupCoordinator.__new__(GroupCoordinator)
        pp.ranks = [0, 1]
        pp.rank = rank
        pp.rank_in_group = rank
        pp.world_size = 2
        pp.cpu_group = dist.group.WORLD
        pp.device_group = dist.group.WORLD
        pp.device = torch.device("cuda", rank)
        pp.pp_cpu_staging = True
        pp.pynccl_comm = None
        pp.mq_broadcaster = None
        x = torch.arange(24, device=pp.device, dtype=torch.float32).reshape(4, 6)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                y = x * 2 + 1
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            y = x * 2 + 1
        for step in range(3):
            x.fill_(step)
            graph.replay()
            payload = {
                "activation": y,
                "empty": torch.empty((2, 0), device=pp.device),
                "scalar": torch.tensor(step, device=pp.device),
                "step": step,
            }
            if rank == 0:
                works = pp.send_tensor_dict(payload, dst=1, async_send=True)
                # Reuse the graph buffer immediately: in-flight sends own a snapshot.
                y.fill_(-1)
                for work in works:
                    work.work.wait()
            else:
                received = pp.recv_tensor_dict(src=0)
                assert received["activation"].device == pp.device
                assert received["empty"].device == pp.device
                assert received["scalar"].device == pp.device
                torch.testing.assert_close(
                    received["activation"],
                    torch.full(
                        (4, 6), 2 * step + 1, device=pp.device, dtype=torch.float32
                    ),
                )
                assert received["scalar"].item() == step
                assert received["step"] == step
                assert received["empty"].shape == (2, 0)
            dist.barrier()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        dist.destroy_process_group()


class TestPPGlooCudaPayload(CustomTestCase):
    def test_graph_output_crosses_host_staged_pp(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(
                _run_host_staged_pp,
                args=(Path(directory, "store").as_uri(),),
                nprocs=2,
                join=True,
            )


if __name__ == "__main__":
    unittest.main()
