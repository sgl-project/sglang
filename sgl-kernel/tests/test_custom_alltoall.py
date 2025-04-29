import ctypes
import multiprocessing as mp
import random
import socket
import time
import unittest
from typing import Any, List, Optional

import sgl_kernel as custom_ops
import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from sglang.srt.distributed.device_communicators.cuda_wrapper import CudaRTLibrary
from sglang.srt.utils import is_cuda

_is_cuda = is_cuda()


def _run_correctness_worker(world_size, rank, distributed_init_port):
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

    test_sizes = [16, 32, 64, 128, 256, 512, 1024, 4096]
    head_sizes = [512, 576]
    pad_lens = [0, 10]

    custom_ptr = None

    peer_output_size = 256  # 8*8*2
    max_size = (max(test_sizes) + 8 * max(pad_lens)) * 128 * max(head_sizes)

    try:
        device = torch.device(f"cuda:{rank}")
        meta_ptrs = TestCustomAllToAll.create_shared_buffer(
            custom_ops.meta_size() + peer_output_size, group=group
        )

        rank_data = torch.empty(8 * 1024 * 1024, dtype=torch.uint8, device=device)
        custom_ptr = custom_ops.init_custom_ar(meta_ptrs, rank_data, rank, True)

        output_buffer = torch.empty(max_size, dtype=torch.bfloat16, device=device)
        buffer_ptrs = TestCustomAllToAll.register_output_buffers(
            custom_ptr, output_buffer, group
        )

        test_loop = 10
        for sz in test_sizes:
            for dtype in [torch.bfloat16]:
                for head_size in head_sizes:
                    for pad in pad_lens:
                        for _ in range(test_loop):
                            TestCustomAllToAll.run_acc_tp2dp(
                                custom_ptr,
                                buffer_ptrs[rank],
                                world_size,
                                head_size,
                                sz,
                                pad,
                                dtype,
                                group,
                                output_buffer,
                            )
                        for _ in range(test_loop):
                            TestCustomAllToAll.run_acc_dp2tp(
                                custom_ptr,
                                buffer_ptrs[rank],
                                world_size,
                                head_size,
                                sz,
                                pad,
                                dtype,
                                group,
                                output_buffer,
                            )

    finally:
        dist.barrier(group=group)
        if custom_ptr is not None:
            custom_ops.dispose(custom_ptr)
        if meta_ptrs:
            TestCustomAllToAll.free_shared_buffer(meta_ptrs, group)

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


class TestCustomAllToAll(unittest.TestCase):
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
    def register_output_buffers(
        custom_ptr, output_buffer, group: Optional[ProcessGroup] = None
    ):
        lib = CudaRTLibrary()
        pointer = output_buffer.untyped_storage().data_ptr()
        handle = lib.cudaIpcGetMemHandle(pointer)
        world_size = dist.get_world_size(group=group)
        rank = dist.get_rank(group=group)
        handles = [None] * world_size
        dist.all_gather_object(handles, handle, group=group)

        pointers: List[int] = []
        for i, h in enumerate(handles):
            if i == rank:
                pointers.append(pointer)  # type: ignore
            else:
                pointers.append(lib.cudaIpcOpenMemHandle(h).value)  # type: ignore
        # self.output_ptrs = pointers
        custom_ops.register_buffer(custom_ptr, pointers)
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
        if not _is_cuda:
            return
        for world_size in self.world_sizes:
            available_gpus = torch.cuda.device_count()
            if world_size > available_gpus:
                print(
                    f"Skipping world_size={world_size}, requires {world_size} GPUs, found {available_gpus}"
                )
                continue

            print(f"Running test for world_size={world_size}")
            multi_process_parallel(
                world_size, _run_correctness_worker, target_args=tuple()
            )
            print(f"custom allreduce tp = {world_size}: OK")

    @staticmethod
    def custom_alltoall(
        custom_ptr,
        buffer_ptr,
        output,
        input,
        output_split_sizes,
        input_split_sizes,
        chunk_size,
        output_split_offsets,
        input_split_offsets,
    ):
        output_split_sizes = torch.tensor(
            output_split_sizes,
            dtype=torch.int64,
            device=torch.cuda.current_device(),
        )
        input_split_sizes = torch.tensor(
            input_split_sizes,
            dtype=torch.int64,
            device=torch.cuda.current_device(),
        )
        if output_split_offsets is not None:
            output_split_offsets = torch.tensor(
                output_split_offsets,
                dtype=torch.int64,
                device=torch.cuda.current_device(),
            )
        if input_split_offsets is not None:
            input_split_offsets = torch.tensor(
                input_split_offsets,
                dtype=torch.int64,
                device=torch.cuda.current_device(),
            )
        meta_extra = (input.numel() * input.dtype.itemsize // 512 + 7) // 8
        plan_meta = torch.zeros(
            128 + meta_extra, dtype=torch.int64, device=input.device
        )
        custom_ops.all_to_all_plan(
            custom_ptr,
            output,
            input,
            output_split_sizes,
            input_split_sizes,
            chunk_size,
            output_split_offsets,
            input_split_offsets,
            plan_meta,
        )
        custom_ops.all_to_all(
            custom_ptr,
            output,
            input,
            plan_meta,
            buffer_ptr,
        )

    @staticmethod
    def run_acc_tp2dp(
        custom_ptr,
        buffer_ptr,
        world_size,
        head_size,
        sz,
        pad,
        dtype,
        group,
        output_buffer,
    ):
        global_head = 128
        local_head = global_head // world_size
        chunk_size = head_size * local_head
        out1 = output_buffer[: (sz // world_size + pad) * global_head * head_size].view(
            sz // world_size + pad, world_size, local_head * head_size
        )

        inp1 = torch.randn(
            sz,
            1,
            local_head * head_size,
            dtype=dtype,
            device=torch.cuda.current_device(),
        )
        out_split_size = [sz // world_size] * world_size
        inp_split_size = [sz // world_size] * world_size
        out_split_offset = [0] * world_size
        inp_split_offset = None
        for i in range(1, world_size):
            out_split_offset[i] = out_split_size[i - 1] + pad + out_split_offset[i - 1]
        TestCustomAllToAll.custom_alltoall(
            custom_ptr,
            buffer_ptr,
            out1.transpose(0, 1),
            inp1,
            out_split_size,
            inp_split_size,
            chunk_size,
            out_split_offset,
            inp_split_offset,
        )
        cmp_out1 = out1[: sz // world_size].view(
            sz // world_size, global_head, head_size
        )
        out2 = torch.randn(
            world_size,
            sz // world_size,
            chunk_size,
            dtype=dtype,
            device=torch.cuda.current_device(),
        )
        inp2 = inp1[:sz]
        dist.all_to_all_single(
            out2.view(-1, chunk_size),
            inp2.view(-1, chunk_size),
            out_split_size,
            inp_split_size,
            group=group,
        )
        out2 = (
            out2.transpose(0, 1)
            .contiguous()
            .view(sz // world_size, global_head, head_size)
        )
        torch.testing.assert_close(cmp_out1, out2)

    @staticmethod
    def run_acc_dp2tp(
        custom_ptr,
        buffer_ptr,
        world_size,
        head_size,
        sz,
        pad,
        dtype,
        group,
        output_buffer,
    ):
        global_head = 128
        local_head = global_head // world_size
        chunk_size = head_size * local_head
        out1 = output_buffer[: sz * local_head * head_size].view(
            sz, 1, local_head * head_size
        )

        inp1 = torch.randn(
            sz // world_size + pad,
            world_size,
            local_head * head_size,
            dtype=dtype,
            device=torch.cuda.current_device(),
        )
        out_split_size = [sz // world_size] * world_size
        inp_split_size = [sz // world_size] * world_size
        out_split_offset = None
        inp_split_offset = [0] * world_size
        for i in range(1, world_size):
            inp_split_offset[i] = inp_split_size[i - 1] + pad + inp_split_offset[i - 1]
        TestCustomAllToAll.custom_alltoall(
            custom_ptr,
            buffer_ptr,
            out1,
            inp1.transpose(0, 1),
            out_split_size,
            inp_split_size,
            chunk_size,
            out_split_offset,
            inp_split_offset,
        )
        cmp_out1 = out1.view(sz, local_head, head_size)
        out2 = torch.randn(
            sz,
            local_head,
            head_size,
            dtype=dtype,
            device=torch.cuda.current_device(),
        )
        # world_size, sz//world_size, local_head*head_size
        inp2 = inp1[: sz // world_size].transpose(0, 1).contiguous()
        dist.all_to_all_single(
            out2.view(-1, chunk_size),
            inp2.view(-1, chunk_size),
            out_split_size,
            inp_split_size,
            group=group,
        )
        torch.testing.assert_close(cmp_out1, out2)


if __name__ == "__main__":
    unittest.main()
