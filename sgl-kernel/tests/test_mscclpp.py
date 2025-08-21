import multiprocessing as mp
import os
import socket
import unittest
from enum import IntEnum
from typing import Any

import sgl_kernel.allreduce as custom_ops
import torch
import torch.distributed as dist


class MscclContextSelection(IntEnum):
    MSCCL1SHOT1NODELL = 1
    MSCCL1SHOT2NODELL = 2


def _run_correctness_worker(world_size, rank, distributed_init_port, test_sizes):
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    torch.cuda.set_device(device)
    distributed_init_method = f"tcp://localhost:{distributed_init_port}"
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
        rank_to_ib[r] = rank % 8
    MAX_BYTES = 2**20
    scratch = torch.empty(
        MAX_BYTES * 8, dtype=torch.bfloat16, device=torch.cuda.current_device()
    )
    put_buffer = torch.empty(
        MAX_BYTES, dtype=torch.bfloat16, device=torch.cuda.current_device()
    )
    print(f"[{rank}] start mscclpp_context init")
    nranks_per_node = torch.cuda.device_count()
    selection = int(MscclContextSelection.MSCCL1SHOT1NODELL)
    mscclpp_context = custom_ops.mscclpp_init_context(
        unique_id,
        rank,
        world_size,
        scratch,
        put_buffer,
        nranks_per_node,
        rank_to_node,
        rank_to_ib,
        selection,
    )
    try:
        test_loop = 10
        for sz in test_sizes:
            for dtype in [torch.float32, torch.float16, torch.bfloat16]:
                if sz * dtype.itemsize > MAX_BYTES:
                    continue
                if rank == 0:
                    print(f"mscclpp allreduce test sz {sz}, dtype {dtype}")
                for _ in range(test_loop):
                    inp1 = torch.randint(1, 16, (sz,), dtype=dtype, device=device)
                    inp1_ref = inp1.clone()
                    out1 = torch.empty_like(inp1)
                    custom_ops.mscclpp_allreduce(
                        mscclpp_context, inp1, out1, nthreads=512, nblocks=21
                    )
                    dist.all_reduce(inp1_ref, group=group)
                    torch.testing.assert_close(out1, inp1_ref)
    finally:
        dist.barrier(group=group)
        dist.destroy_process_group(group=group)


def get_open_port() -> int:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]
    except OSError:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
            s.bind(("::1", 0))
            return s.getsockname()[1]


def multi_process_parallel(
    world_size: int, test_target: Any, target_args: tuple = ()
) -> None:
    mp.set_start_method("spawn", force=True)

    procs = []
    distributed_init_port = get_open_port()
    for i in range(world_size):
        proc_args = (world_size, i, distributed_init_port) + target_args
        proc = mp.Process(target=test_target, args=proc_args, name=f"Worker-{i}")
        proc.start()
        procs.append(proc)

    for i in range(world_size):
        procs[i].join()
        assert (
            procs[i].exitcode == 0
        ), f"Process {i} failed with exit code {procs[i].exitcode}"


class TestMSCCLAllReduce(unittest.TestCase):
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

    def test_correctness(self):
        for world_size in self.world_sizes:
            available_gpus = torch.cuda.device_count()
            if world_size > available_gpus:
                print(
                    f"Skipping world_size={world_size}, found {available_gpus} and now ray is not supported here"
                )
                continue

            print(f"Running test for world_size={world_size}")
            multi_process_parallel(
                world_size, _run_correctness_worker, target_args=(self.test_sizes,)
            )
            print(f"custom allreduce tp = {world_size}: OK")


if __name__ == "__main__":
    unittest.main()
