"""
export WORLD_SIZE=1
export RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=12345

torchrun --nproc_per_node gpu \
--nnodes $WORLD_SIZE \
--node_rank $RANK \
--master_addr $MASTER_ADDR \
--master_port $MASTER_PORT \
test_mscclpp.py

if [[ $RANK -eq 0  ]]; then
    ray start --block --head --port=6379 &
    python3 test_mscclpp.py;
else
    ray start --block --address=${MASTER_ADDR}:6379;
fi
"""

import ctypes
import multiprocessing as mp
import random
import socket
import unittest
from typing import Any, List, Optional

import sgl_kernel.allreduce as custom_ops
import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from sglang.srt.distributed.device_communicators.cuda_wrapper import CudaRTLibrary
from enum import IntEnum
class MscclContextSelection(IntEnum):
  MSCCL1SHOT1NODELL = 1
  MSCCL1SHOT2NODELL = 2

def _run_correctness_worker(world_size, rank, distributed_init_port, test_sizes):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    distributed_init_method = f"tcp://localhost:{distributed_init_port}"
    dist.init_process_group(
        backend="nccl",
        init_method=distributed_init_method,
        rank=rank,
        world_size=world_size,
    )
    group = dist.group.WORLD
    if rank == 0:
        unique_id = [custom_ops.mscclpp_generate_unique_id()]
    else:
        unique_id = [None]
    dist.broadcast_object_list(unique_id, src=0, device=torch.device("cpu"), group=cpu_group)
    unique_id = unique_id[0]
    print(type(unique_id))
    rank_to_node, rank_to_ib = list(range(world_size)), list(range(world_size))
    for r in range(world_size):
        rank_to_node[r] = r // 8
        rank_to_ib[r] = r % 8
    MAX_BYTES = 2**20 // 2
    scratch = torch.empty(MAX_BYTES * 8, dtype=torch.bfloat16, device=torch.cuda.current_device())
    put_buffer = torch.empty(MAX_BYTES, dtype=torch.bfloat16, device=torch.cuda.current_device())
    print(f"[{rank}] start mscclpp_context init")
    if world_size == 8:
        selection = int(MscclContextSelection.MSCCL1SHOT1NODELL)
    elif world_size == 16:
        selection = int(MscclContextSelection.MSCCL1SHOT2NODELL)
    mscclpp_context = custom_ops.mscclpp_init_context(
        unique_id, rank, world_size, scratch, put_buffer, 
        rank_to_node, rank_to_ib, selection
    )
    try:
        test_loop = 10
        for sz in test_sizes:
            for dtype in [torch.float32, torch.float16, torch.bfloat16]:
                for _ in range(test_loop):
                    inp1 = torch.randint(1, 16, (sz,), dtype=dtype, device=device)
                    inp1_ref = inp1.clone()
                    out1 = torch.empty_like(inp1)

                    # custom_ops.all_reduce(
                    #     custom_ptr, inp1, out1, buffer_ptrs[rank], max_size
                    # )
                    custom_ops.mscclpp_allreduce(mscclpp_context, inp1, out1)

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
        1048576,
        2097152,
    ]
    world_sizes = [2, 4, 8]

    @staticmethod
    def create_shared_buffer(
        size_in_bytes: int, group: Optional[ProcessGroup] = None
    ) -> List[int]:
        lib = CudaRTLibrary()
        pointer = lib.cudaMalloc(size_in_bytes)
        handle = lib.cudaIpcGetMemHandle(pointer)
        if group is None:
            group = dist.group.WORLD
        world_size = dist.get_world_size(group=group)
        rank = dist.get_rank(group=group)

        handle_bytes = ctypes.string_at(ctypes.addressof(handle), ctypes.sizeof(handle))
        input_tensor = torch.ByteTensor(list(handle_bytes)).to(f"cuda:{rank}")
        gathered_tensors = [torch.empty_like(input_tensor) for _ in range(world_size)]
        dist.all_gather(gathered_tensors, input_tensor, group=group)

        handles = []
        handle_type = type(handle)
        for tensor in gathered_tensors:
            bytes_list = tensor.cpu().tolist()
            bytes_data = bytes(bytes_list)
            handle_obj = handle_type()
            ctypes.memmove(ctypes.addressof(handle_obj), bytes_data, len(bytes_data))
            handles.append(handle_obj)

        pointers: List[int] = []
        for i, h in enumerate(handles):
            if i == rank:
                pointers.append(pointer.value)
            else:
                try:
                    opened_ptr = lib.cudaIpcOpenMemHandle(h)
                    pointers.append(opened_ptr.value)
                except Exception as e:
                    print(f"Rank {rank}: Failed to open IPC handle from rank {i}: {e}")
                    raise

        dist.barrier(group=group)
        return pointers

    @staticmethod
    def free_shared_buffer(
        pointers: List[int], group: Optional[ProcessGroup] = None
    ) -> None:
        if group is None:
            group = dist.group.WORLD
        rank = dist.get_rank(group=group)
        lib = CudaRTLibrary()
        if pointers and len(pointers) > rank and pointers[rank] is not None:
            lib.cudaFree(ctypes.c_void_p(pointers[rank]))
        dist.barrier(group=group)

    def test_correctness(self):
        for world_size in self.world_sizes:
            available_gpus = torch.cuda.device_count()
            if world_size > available_gpus:
                print(
                    f"Skipping world_size={world_size}, requires {world_size} GPUs, found {available_gpus}"
                )
                continue

            print(f"Running test for world_size={world_size}")
            multi_process_parallel(
                world_size, _run_correctness_worker, target_args=(self.test_sizes,)
            )
            print(f"custom allreduce tp = {world_size}: OK")


if __name__ == "__main__":
    # unittest.main()
    # exit(0)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    world, world_size = dist.group.WORLD, dist.get_world_size()
    rank = dist.get_rank()
    torch.cuda.set_device(rank % 8)
    device = torch.cuda.current_device()
    group = world
    cpu_group = torch.distributed.new_group(list(range(world_size)), backend="gloo")
    dtype = torch.bfloat16
    if rank == 0:
        unique_id = [custom_ops.mscclpp_generate_unique_id()]
    else:
        unique_id = [None]
    dist.broadcast_object_list(unique_id, src=0, device=torch.device("cpu"), group=cpu_group)
    unique_id = unique_id[0]
    print(type(unique_id))
    rank_to_node, rank_to_ib = list(range(world_size)), list(range(world_size))
    for r in range(world_size):
        rank_to_node[r] = r // 8
        rank_to_ib[r] = r % 8
    MAX_BYTES = 2**20 // 2
    scratch = torch.empty(MAX_BYTES * 8, dtype=torch.bfloat16, device=torch.cuda.current_device())
    put_buffer = torch.empty(MAX_BYTES, dtype=torch.bfloat16, device=torch.cuda.current_device())
    print(f"[{rank}] start mscclpp_context init")
    if world_size == 8:
        selection = int(MscclContextSelection.MSCCL1SHOT1NODELL)
    elif world_size == 16:
        selection = int(MscclContextSelection.MSCCL1SHOT2NODELL)
    mscclpp_context = custom_ops.mscclpp_init_context(
        unique_id, rank, world_size, scratch, put_buffer, 
        rank_to_node, rank_to_ib, selection
    )
    print(f"[{rank}] finish mscclpp_context init")
    for sz in [4096]:
        for i in range(10):
            print(f"[{rank}] size: {sz}, loop: {i}/10")
            inp1 = torch.randint(1, 16, (sz,), dtype=dtype, device=torch.cuda.current_device())
            # inp1 = torch.randn((sz, ), dtype=torch.bfloat16, device=torch.cuda.current_device())
            out1 = torch.empty_like(inp1)
            custom_ops.mscclpp_allreduce(mscclpp_context, inp1, out1)
            dist.all_reduce(inp1, group=group)
            print(f"out1: {out1}, inp1: {inp1}", flush=True)
            torch.testing.assert_close(out1, inp1)
    print(f"[{rank}] PASS!")