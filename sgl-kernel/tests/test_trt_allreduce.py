import ctypes
import logging
import random
import socket
import time
import unittest
from typing import Any, Dict, List, Optional, Tuple

import ray
import sgl_kernel.allreduce as custom_ops
import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
from vllm import _custom_ops as vllm_ops

from sglang.srt.distributed.device_communicators.cuda_wrapper import CudaRTLibrary

logger = logging.getLogger(__name__)


def get_open_port() -> int:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]
    except OSError:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
            s.bind(("", 0, 0, 0))
            return s.getsockname()[1]


def _create_shared_buffer(
    size_in_bytes: int, group: Optional[ProcessGroup] = None
) -> List[int]:
    rank = dist.get_rank(group=group)
    torch.cuda.set_device(rank)
    lib = CudaRTLibrary()
    pointer = lib.cudaMalloc(size_in_bytes)
    handle = lib.cudaIpcGetMemHandle(pointer)
    world_size = dist.get_world_size(group=group)
    object_list = [None] * world_size
    dist.all_gather_object(object_list, handle, group=group)
    handles = object_list
    pointers: List[int] = []
    for i, h in enumerate(handles):
        if i == rank:
            pointers.append(pointer.value)
        else:
            if h is None:
                raise RuntimeError(f"Rank {rank} received None handle from rank {i}")
            try:
                opened_ptr = lib.cudaIpcOpenMemHandle(h)
                pointers.append(opened_ptr.value)
            except Exception as e:
                raise RuntimeError(
                    f"Rank {rank} failed cudaIpcOpenMemHandle from rank {i}"
                ) from e
    dist.barrier(group=group)
    return pointers


def _free_shared_buffer(
    pointers: List[int], group: Optional[ProcessGroup] = None
) -> None:
    rank = dist.get_rank(group=group)
    torch.cuda.set_device(rank)
    lib = CudaRTLibrary()
    if pointers and rank < len(pointers):
        try:
            lib.cudaFree(ctypes.c_void_p(pointers[rank]))
        except Exception as e:
            logger.error(f"Rank {rank} failed to free shared buffer: {e}")
    dist.barrier(group=group)


def _init_distributed_env(world_size, rank, distributed_init_port):
    torch.cuda.set_device(rank)
    distributed_init_method = f"tcp://127.0.0.1:{distributed_init_port}"
    dist.init_process_group(
        backend="nccl",
        init_method=distributed_init_method,
        rank=rank,
        world_size=world_size,
    )
    dist.barrier()
    return dist.group.WORLD  # Use default group


@ray.remote(num_gpus=1, max_calls=1)
def run_correctness_task(world_size, rank, distributed_init_port, test_sizes):
    group = _init_distributed_env(world_size, rank, distributed_init_port)
    state = {}
    try:
        buffer_max_size = 8 * 1024 * 1024
        barrier_max_size = 8 * (24 + 2) * 8

        state["buffer_ptrs"] = _create_shared_buffer(buffer_max_size, group=group)
        state["tmp_result_buffer_ptrs"] = _create_shared_buffer(
            buffer_max_size, group=group
        )
        state["barrier_in_ptrs"] = _create_shared_buffer(barrier_max_size, group=group)
        state["barrier_out_ptrs"] = _create_shared_buffer(barrier_max_size, group=group)
        state["rank_data"] = torch.empty(
            buffer_max_size, dtype=torch.uint8, device=f"cuda:{rank}"
        )

        state["custom_ptr"] = custom_ops.init_custom_reduce(
            rank,
            world_size,
            state["rank_data"],
            state["buffer_ptrs"],
            state["tmp_result_buffer_ptrs"],
            state["barrier_in_ptrs"],
            state["barrier_out_ptrs"],
        )
        dist.barrier(group=group)

        test_loop = 10
        for sz in test_sizes:
            for dtype in [torch.float32, torch.float16, torch.bfloat16]:
                for _ in range(test_loop):
                    inp_torch = torch.randint(
                        1, 16, (sz,), dtype=dtype, device=f"cuda:{rank}"
                    )
                    inp_custom = inp_torch.clone()
                    out_custom = torch.empty_like(inp_custom)

                    custom_ops.custom_reduce(
                        state["custom_ptr"], inp_custom, out_custom
                    )
                    dist.barrier(group=group)

                    dist.all_reduce(inp_torch, group=group)
                    dist.barrier(group=group)

                    torch.testing.assert_close(
                        out_custom, inp_torch, rtol=1e-3, atol=1e-3
                    )

    finally:
        if "custom_ptr" in state and state["custom_ptr"] is not None:
            custom_ops.custom_dispose(state["custom_ptr"])
        if "buffer_ptrs" in state:
            _free_shared_buffer(state["buffer_ptrs"], group)
        if "tmp_result_buffer_ptrs" in state:
            _free_shared_buffer(state["tmp_result_buffer_ptrs"], group)
        if "barrier_in_ptrs" in state:
            _free_shared_buffer(state["barrier_in_ptrs"], group)
        if "barrier_out_ptrs" in state:
            _free_shared_buffer(state["barrier_out_ptrs"], group)
        dist.barrier(group=group)
        if dist.is_initialized():
            dist.destroy_process_group(group=group)


@ray.remote(num_gpus=1, max_calls=1)
def run_performance_task(world_size, rank, distributed_init_port, test_sizes):
    group = _init_distributed_env(world_size, rank, distributed_init_port)
    custom_state = {}
    vllm_state = {}
    try:
        buffer_max_size = 8 * 1024 * 1024
        barrier_max_size = 8 * (24 + 2) * 8

        custom_state["buffer_ptrs"] = _create_shared_buffer(
            buffer_max_size, group=group
        )
        custom_state["tmp_result_buffer_ptrs"] = _create_shared_buffer(
            buffer_max_size, group=group
        )
        custom_state["barrier_in_ptrs"] = _create_shared_buffer(
            barrier_max_size, group=group
        )
        custom_state["barrier_out_ptrs"] = _create_shared_buffer(
            barrier_max_size, group=group
        )
        custom_state["rank_data"] = torch.empty(
            buffer_max_size, dtype=torch.uint8, device=f"cuda:{rank}"
        )
        custom_state["custom_ptr"] = custom_ops.init_custom_reduce(
            rank,
            world_size,
            custom_state["rank_data"],
            custom_state["buffer_ptrs"],
            custom_state["tmp_result_buffer_ptrs"],
            custom_state["barrier_in_ptrs"],
            custom_state["barrier_out_ptrs"],
        )
        dist.barrier(group=group)

        vllm_state["vllm_max_size"] = buffer_max_size
        vllm_meta_buffer_size = vllm_ops.meta_size() + vllm_state["vllm_max_size"]
        vllm_state["vllm_meta_ptrs"] = _create_shared_buffer(
            vllm_meta_buffer_size, group=group
        )
        vllm_state["vllm_buffer_ptrs"] = _create_shared_buffer(
            vllm_state["vllm_max_size"], group=group
        )
        vllm_state["vllm_rank_data"] = torch.empty(
            buffer_max_size, dtype=torch.uint8, device=f"cuda:{rank}"
        )
        vllm_state["vllm_ptr"] = vllm_ops.init_custom_ar(
            vllm_state["vllm_meta_ptrs"], vllm_state["vllm_rank_data"], rank, True
        )
        vllm_ops.register_buffer(vllm_state["vllm_ptr"], vllm_state["vllm_buffer_ptrs"])
        dist.barrier(group=group)

        for sz in test_sizes:
            inp = torch.randint(
                1, 16, (sz,), dtype=torch.float32, device=f"cuda:{rank}"
            )
            out = torch.empty_like(inp)
            warmup_loop = 10
            test_loop = 50  # Reduced for CI speed

            for _ in range(warmup_loop):
                custom_ops.custom_reduce(custom_state["custom_ptr"], inp, out)
            dist.barrier(group=group)
            torch.cuda.synchronize(device=f"cuda:{rank}")
            start_custom = time.time()
            for _ in range(test_loop):
                custom_ops.custom_reduce(custom_state["custom_ptr"], inp, out)
            dist.barrier(group=group)
            torch.cuda.synchronize(device=f"cuda:{rank}")
            elapse_custom = time.time() - start_custom

            for _ in range(warmup_loop):
                vllm_ops.all_reduce(
                    vllm_state["vllm_ptr"],
                    inp,
                    out,
                    vllm_state["vllm_buffer_ptrs"][rank],
                    vllm_state["vllm_max_size"],
                )
            dist.barrier(group=group)
            torch.cuda.synchronize(device=f"cuda:{rank}")
            start_vllm = time.time()
            for _ in range(test_loop):
                vllm_ops.all_reduce(
                    vllm_state["vllm_ptr"],
                    inp,
                    out,
                    vllm_state["vllm_buffer_ptrs"][rank],
                    vllm_state["vllm_max_size"],
                )
            dist.barrier(group=group)
            torch.cuda.synchronize(device=f"cuda:{rank}")
            elapse_vllm = time.time() - start_vllm

            if rank == 0:
                logger.warning(
                    f"Perf Test: size={sz}, world={world_size}, "
                    f"vLLM avg={(elapse_vllm * 1000 / test_loop):.4f}ms, "
                    f"Custom avg={(elapse_custom * 1000 / test_loop):.4f}ms"
                )
            dist.barrier(group=group)

    finally:
        if "custom_ptr" in custom_state and custom_state["custom_ptr"] is not None:
            custom_ops.custom_dispose(custom_state["custom_ptr"])
        if "buffer_ptrs" in custom_state:
            _free_shared_buffer(custom_state["buffer_ptrs"], group)
        if "tmp_result_buffer_ptrs" in custom_state:
            _free_shared_buffer(custom_state["tmp_result_buffer_ptrs"], group)
        if "barrier_in_ptrs" in custom_state:
            _free_shared_buffer(custom_state["barrier_in_ptrs"], group)
        if "barrier_out_ptrs" in custom_state:
            _free_shared_buffer(custom_state["barrier_out_ptrs"], group)

        if "vllm_ptr" in vllm_state and vllm_state["vllm_ptr"] is not None:
            vllm_ops.dispose(vllm_state["vllm_ptr"])
        if "vllm_meta_ptrs" in vllm_state:
            _free_shared_buffer(vllm_state["vllm_meta_ptrs"], group)
        if "vllm_buffer_ptrs" in vllm_state:
            _free_shared_buffer(vllm_state["vllm_buffer_ptrs"], group)

        dist.barrier(group=group)
        if dist.is_initialized():
            dist.destroy_process_group(group=group)


def _multi_process_parallel(world_size: int, target_func: Any, args: Tuple):
    distributed_init_port = get_open_port()
    refs = []
    for rank in range(world_size):
        task_args = (world_size, rank, distributed_init_port) + args
        refs.append(target_func.remote(*task_args))
    try:
        results = ray.get(refs)
        return results
    except ray.exceptions.RayTaskError as e:
        logger.error(f"Ray task failed: {e}")
        raise e
    except Exception as e:
        logger.error(f"An unexpected error occurred during ray.get: {e}")
        raise e


class TestCustomAllReduce(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        random.seed(42)
        cls.test_sizes = [512, 4096, 32768, 262144, 524288, 1048576, 2097152]
        cls.world_sizes = [2, 4, 8]
        ray.init(log_to_driver=True, num_cpus=1, include_dashboard=False)

    @classmethod
    def tearDownClass(cls):
        ray.shutdown()

    def test_correctness(self):
        for world_size in self.world_sizes:
            if world_size > torch.cuda.device_count():
                logger.warning(
                    f"Skipping correctness test for world_size={world_size} due to insufficient GPUs ({torch.cuda.device_count()})."
                )
                continue
            _multi_process_parallel(
                world_size, run_correctness_task, (self.test_sizes,)
            )

    def test_performance(self):
        for world_size in self.world_sizes:
            if world_size > torch.cuda.device_count():
                logger.warning(
                    f"Skipping performance test for world_size={world_size} due to insufficient GPUs ({torch.cuda.device_count()})."
                )
                continue
            _multi_process_parallel(
                world_size, run_performance_task, (self.test_sizes,)
            )


if __name__ == "__main__":
    unittest.main()
