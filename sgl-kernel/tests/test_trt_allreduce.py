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


def get_open_port() -> int:
    # try ipv4
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]
    except OSError:
        # try ipv6
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]


def multi_process_parallel(
    world_size: int,
    test_target: Any,
) -> None:
    procs = []
    distributed_init_port = get_open_port()
    for i in range(world_size):
        proc = mp.Process(
            target=test_target,
            args=(world_size, i, distributed_init_port),
        )
        proc.start()
        procs.append(proc)

    for i in range(world_size):
        procs[i].join()
        assert procs[i].exitcode == 0


class TestCustomAllReduce(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        random.seed(42)
        cls.test_sizes = [512, 4096, 32768, 262144, 524288, 1048576, 2097152]
        cls.world_sizes = [2, 4, 8]

    @staticmethod
    def create_shared_buffer(
        size_in_bytes: int, group: Optional[ProcessGroup] = None
    ) -> List[int]:
        """
        Creates a shared buffer and returns a list of pointers
        representing the buffer on all processes in the group.
        """
        lib = CudaRTLibrary()
        pointer = lib.cudaMalloc(size_in_bytes)
        handle = lib.cudaIpcGetMemHandle(pointer)
        world_size = dist.get_world_size(group=group)
        rank = dist.get_rank(group=group)
        handles = [None] * world_size
        dist.all_gather_object(handles, handle, group=group)

        pointers: List[int] = []
        for i, h in enumerate(handles):
            if i == rank:
                pointers.append(pointer.value)  # type: ignore
            else:
                pointers.append(lib.cudaIpcOpenMemHandle(h).value)  # type: ignore

        return pointers

    @staticmethod
    def free_shared_buffer(
        pointers: List[int], group: Optional[ProcessGroup] = None
    ) -> None:
        rank = dist.get_rank(group=group)
        lib = CudaRTLibrary()
        lib.cudaFree(ctypes.c_void_p(pointers[rank]))

    def test_correctness(self):
        for world_size in self.world_sizes:
            if world_size > torch.cuda.device_count():
                continue
            multi_process_parallel(world_size, self.correctness)
            print(f"custom allreduce tp = {world_size}: OK")

    def init_custom_allreduce(self, rank, world_size, group):
        buffer_max_size = 8 * 1024 * 1024
        barrier_max_size = 8 * (24 + 2) * 8

        self.buffer_ptrs = self.create_shared_buffer(buffer_max_size, group=group)
        self.tmp_result_buffer_ptrs = self.create_shared_buffer(
            buffer_max_size, group=group
        )
        self.barrier_in_ptrs = self.create_shared_buffer(barrier_max_size, group=group)
        self.barrier_out_ptrs = self.create_shared_buffer(barrier_max_size, group=group)
        self.rank_data = torch.empty(
            8 * 1024 * 1024, dtype=torch.uint8, device=torch.device("cuda:0")
        )

        self.custom_ptr = custom_ops.init_custom_reduce(
            rank,
            world_size,
            self.rank_data,
            self.buffer_ptrs,
            self.tmp_result_buffer_ptrs,
            self.barrier_in_ptrs,
            self.barrier_out_ptrs,
        )

    def custom_allreduce(self, inp, out):
        custom_ops.custom_reduce(self.custom_ptr, inp, out)

    def free_custom_allreduce(self, group):
        self.free_shared_buffer(self.buffer_ptrs, group)
        self.free_shared_buffer(self.tmp_result_buffer_ptrs, group)
        self.free_shared_buffer(self.barrier_in_ptrs, group)
        self.free_shared_buffer(self.barrier_out_ptrs, group)
        custom_ops.custom_dispose(self.custom_ptr)

    @staticmethod
    def init_distributed_env(world_size, rank, distributed_init_port):
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        ranks = [i for i in range(world_size)]
        distributed_init_method = f"tcp://localhost:{distributed_init_port}"
        dist.init_process_group(
            backend="nccl",
            init_method=distributed_init_method,
            rank=rank,
            world_size=world_size,
        )
        group = torch.distributed.new_group(ranks, backend="gloo")
        return group

    # compare result with torch.distributed
    def correctness(self, world_size, rank, distributed_init_port):
        group = self.init_distributed_env(world_size, rank, distributed_init_port)

        self.init_custom_allreduce(rank=rank, world_size=world_size, group=group)

        test_loop = 10
        for sz in self.test_sizes:
            for dtype in [torch.float32, torch.float16, torch.bfloat16]:
                for _ in range(test_loop):
                    inp1 = torch.randint(
                        1, 16, (sz,), dtype=dtype, device=torch.cuda.current_device()
                    )
                    out1 = torch.empty_like(inp1)
                    self.custom_allreduce(inp1, out1)

                    dist.all_reduce(inp1, group=group)
                    torch.testing.assert_close(out1, inp1)

        self.free_custom_allreduce(group)


if __name__ == "__main__":
    unittest.main()
