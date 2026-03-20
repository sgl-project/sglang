"""
Correctness tests for the custom_all_reduce JIT kernel.

Tests the JIT-compiled custom allreduce against torch.distributed allreduce.
Requires multiple GPUs. Single-GPU tests cover meta_size and module loading.
"""

import ctypes
import multiprocessing as mp
import socket
import unittest
from typing import List, Optional

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from sglang.jit_kernel.custom_all_reduce import (
    all_reduce,
    dispose,
    init_custom_ar,
    meta_size,
    register_buffer,
)
from sglang.srt.distributed.device_communicators.cuda_wrapper import CudaRTLibrary


def get_open_port() -> int:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]
    except OSError:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
            s.bind(("::1", 0))
            return s.getsockname()[1]


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
            opened_ptr = lib.cudaIpcOpenMemHandle(h)
            pointers.append(opened_ptr.value)

    dist.barrier(group=group)
    return pointers


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

    custom_ptr = None
    buffer_ptrs = None
    meta_ptrs = None
    try:
        max_size = 8192 * 1024
        meta_ptrs = create_shared_buffer(meta_size() + max_size, group=group)
        rank_data = torch.empty(8 * 1024 * 1024, dtype=torch.uint8, device=device)
        buffer_ptrs = create_shared_buffer(max_size, group=group)

        custom_ptr = init_custom_ar(meta_ptrs, rank_data, rank, True)
        register_buffer(custom_ptr, buffer_ptrs)

        for sz in test_sizes:
            for dtype in [torch.float32, torch.float16, torch.bfloat16]:
                inp = torch.randint(1, 16, (sz,), dtype=dtype, device=device)
                inp_ref = inp.clone()
                out = torch.empty_like(inp)

                all_reduce(custom_ptr, inp, out, buffer_ptrs[rank], max_size)
                dist.all_reduce(inp_ref, group=group)

                torch.testing.assert_close(out, inp_ref)

    finally:
        dist.barrier(group=group)
        if custom_ptr is not None:
            dispose(custom_ptr)
        if buffer_ptrs:
            free_shared_buffer(buffer_ptrs, group)
        if meta_ptrs:
            free_shared_buffer(meta_ptrs, group)
        dist.destroy_process_group(group=group)


def multi_process_parallel(
    world_size: int, test_target, target_args: tuple = ()
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


class TestCustomAllReduceJIT(unittest.TestCase):
    test_sizes = [512, 2560, 4096, 32768, 262144, 524288, 1048576]
    world_sizes = [2, 4, 8]

    def test_meta_size(self):
        """meta_size() should return a positive integer."""
        size = meta_size()
        assert isinstance(size, int) and size > 0, f"Expected positive int, got {size}"

    def test_correctness(self):
        for world_size in self.world_sizes:
            available_gpus = torch.cuda.device_count()
            if world_size > available_gpus:
                print(
                    f"Skipping world_size={world_size}, requires {world_size} GPUs, "
                    f"found {available_gpus}"
                )
                continue

            multi_process_parallel(
                world_size, _run_correctness_worker, target_args=(self.test_sizes,)
            )
            print(f"custom allreduce JIT tp={world_size}: OK")


if __name__ == "__main__":
    unittest.main()
