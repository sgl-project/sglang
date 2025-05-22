"""
if [[ $RANK -eq 0 ]]; then
    ray start --block --head --port=6379 &
    python3 test_mscclpp.py;
else
    ray start --block --address=${MASTER_ADDR}:6379;
fi
"""

import ctypes
import multiprocessing as mp
import os
import socket
import unittest
from enum import IntEnum
from typing import Any, List, Optional

import ray
import sgl_kernel.allreduce as custom_ops
import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from sglang.srt.distributed.device_communicators.cuda_wrapper import CudaRTLibrary


class MscclContextSelection(IntEnum):
    MSCCL1SHOT1NODELL = 1
    MSCCL1SHOT2NODELL = 2


def get_open_port() -> int:
    # try ipv4
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]
    except OSError:
        # try ipv6
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]


def multi_process_parallel(
    world_size: int,
    cls: Any,
    test_target: Any,
) -> None:

    # Using ray helps debugging the error when it failed
    # as compared to multiprocessing.
    # NOTE: We need to set working_dir for distributed tests,
    # otherwise we may get import errors on ray workers
    ray.init(log_to_driver=True)

    TEST_MASTER_ADDR = os.getenv("SGL_MSCCLPP_TEST_MASTER_ADDR", "localhost")
    distributed_init_port = get_open_port()
    refs = []
    for rank in range(world_size):
        refs.append(
            test_target.remote(
                cls, world_size, TEST_MASTER_ADDR, rank, distributed_init_port
            )
        )
    ray.get(refs)

    ray.shutdown()


class TestCustomAllReduce(unittest.TestCase):
    test_sizes = [
        512,
        2560,
        4096,
        5120,
        7680,
        32768,
        262144,
        524288,
    ]
    world_sizes = [8]

    @ray.remote(num_gpus=1, max_calls=1)
    def _run_correctness_worker(
        self, world_size, master_addr, rank, distributed_init_port
    ):
        del os.environ["CUDA_VISIBLE_DEVICES"]
        print(
            f"run with world_size={world_size}, master_addr={master_addr}, rank={rank}, port={distributed_init_port}"
        )
        device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
        torch.cuda.set_device(device)
        distributed_init_method = f"tcp://{master_addr}:{distributed_init_port}"
        dist.init_process_group(
            backend="nccl",
            init_method=distributed_init_method,
            rank=rank,
            world_size=world_size,
        )
        group = dist.group.WORLD
        cpu_group = torch.distributed.new_group(list(range(world_size)), backend="gloo")
        if rank == 0:
            unique_id = [custom_ops.mscclpp_generate_unique_id()]
        else:
            unique_id = [None]
        dist.broadcast_object_list(
            unique_id, src=0, device=torch.device("cpu"), group=cpu_group
        )
        unique_id = unique_id[0]
        rank_to_node, rank_to_ib = list(range(world_size)), list(range(world_size))
        for r in range(world_size):
            rank_to_node[r] = r // 8
            rank_to_ib[r] = r % 8
        MAX_BYTES = 2**20 // 2
        scratch = torch.empty(
            MAX_BYTES * 8, dtype=torch.bfloat16, device=torch.cuda.current_device()
        )
        put_buffer = torch.empty(
            MAX_BYTES, dtype=torch.bfloat16, device=torch.cuda.current_device()
        )
        print(f"[{rank}] start mscclpp_context init")
        if world_size == 8:
            selection = int(MscclContextSelection.MSCCL1SHOT1NODELL)
        elif world_size == 16:
            selection = int(MscclContextSelection.MSCCL1SHOT2NODELL)
        mscclpp_context = custom_ops.mscclpp_init_context(
            unique_id,
            rank,
            world_size,
            scratch,
            put_buffer,
            rank_to_node,
            rank_to_ib,
            selection,
        )
        try:
            test_loop = 10
            for sz in self.test_sizes:
                for dtype in [torch.float32, torch.float16, torch.bfloat16]:
                    for _ in range(test_loop):
                        inp1 = torch.randint(1, 16, (sz,), dtype=dtype, device=device)
                        inp1_ref = inp1.clone()
                        out1 = torch.empty_like(inp1)

                        custom_ops.mscclpp_allreduce(mscclpp_context, inp1, out1)

                        dist.all_reduce(inp1_ref, group=group)

                        torch.testing.assert_close(out1, inp1_ref)

        finally:
            dist.barrier(group=group)
            dist.destroy_process_group(group=group)

    def test_correctness(self):
        TEST_TP16 = int(os.getenv("SGL_MSCCLPP_TEST_TP16", "0"))
        if TEST_TP16:
            self.world_sizes.append(16)
        for world_size in self.world_sizes:
            print(f"Running test for world_size={world_size}")
            multi_process_parallel(world_size, self, self._run_correctness_worker)
            print(f"mscclpp allreduce tp = {world_size}: OK")


if __name__ == "__main__":
    unittest.main()
