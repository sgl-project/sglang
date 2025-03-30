import ctypes
import logging
import random
import socket
import time
from typing import Any, Callable, List, Optional

import pytest
import ray
import sgl_kernel.allreduce as custom_ops
import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
from vllm import _custom_ops as vllm_ops

from sglang.srt.distributed.device_communicators.cuda_wrapper import CudaRTLibrary

logger = logging.getLogger(__name__)


TEST_SIZES = [512, 4096, 32768, 262144, 524288, 1048576, 2097152]
WORLD_SIZES = [2, 4]
BUFFER_MAX_SIZE = 8 * 1024 * 1024
BARRIER_MAX_SIZE = 8 * (24 + 2) * 8
VLLM_MAX_SIZE = 8 * 1024 * 1024


def get_open_port() -> int:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]
    except OSError:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
            s.bind(("::1", 0))
            return s.getsockname()[1]


def init_distributed_env(world_size, rank, distributed_init_port):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    ranks = [i for i in range(world_size)]
    distributed_init_method = f"tcp://127.0.0.1:{distributed_init_port}"
    dist.init_process_group(
        backend="nccl",
        init_method=distributed_init_method,
        rank=rank,
        world_size=world_size,
    )
    return None


def create_shared_buffer(
    size_in_bytes: int, group: Optional[ProcessGroup] = None
) -> List[int]:
    lib = CudaRTLibrary()
    if not torch.cuda.is_initialized():
        torch.cuda.init()

    pointer = lib.cudaMalloc(size_in_bytes)
    handle = lib.cudaIpcGetMemHandle(pointer)
    world_size = dist.get_world_size(group=group)
    rank = dist.get_rank(group=group)

    gathered_handles = [None] * world_size
    dist.all_gather_object(gathered_handles, handle, group=group)

    pointers: List[int] = [0] * world_size
    for i, h in enumerate(gathered_handles):
        if h is None:
            raise RuntimeError(
                f"Rank {i} did not receive a valid handle from rank {rank}."
            )
        if i == rank:
            pointers[i] = pointer.value
        else:
            opened_ptr = lib.cudaIpcOpenMemHandle(h)
            pointers[i] = opened_ptr.value

    dist.barrier(group=group)
    return pointers


def free_shared_buffer(
    pointers: List[int], group: Optional[ProcessGroup] = None
) -> None:
    if not pointers:
        return
    rank = dist.get_rank(group=group)
    lib = CudaRTLibrary()
    if rank < len(pointers) and pointers[rank] != 0:
        try:
            lib.cudaFree(ctypes.c_void_p(pointers[rank]))
        except Exception as e:
            logger.error(
                f"Rank {rank}: Error freeing shared buffer pointer {pointers[rank]}: {e}"
            )
    dist.barrier(group=group)


@ray.remote(num_gpus=1, max_calls=1)
def correctness_worker(rank, world_size, distributed_init_port):
    group = init_distributed_env(world_size, rank, distributed_init_port)
    worker_state = {}

    try:
        worker_state["buffer_ptrs"] = create_shared_buffer(BUFFER_MAX_SIZE, group=group)
        worker_state["tmp_result_buffer_ptrs"] = create_shared_buffer(
            BUFFER_MAX_SIZE, group=group
        )
        worker_state["barrier_in_ptrs"] = create_shared_buffer(
            BARRIER_MAX_SIZE, group=group
        )
        worker_state["barrier_out_ptrs"] = create_shared_buffer(
            BARRIER_MAX_SIZE, group=group
        )
        worker_state["rank_data"] = torch.empty(
            BUFFER_MAX_SIZE, dtype=torch.uint8, device=torch.cuda.current_device()
        )
        worker_state["custom_ptr"] = custom_ops.init_custom_reduce(
            rank,
            world_size,
            worker_state["rank_data"],
            worker_state["buffer_ptrs"],
            worker_state["tmp_result_buffer_ptrs"],
            worker_state["barrier_in_ptrs"],
            worker_state["barrier_out_ptrs"],
        )

        test_loop = 10
        for sz in TEST_SIZES:
            for dtype in [torch.float32, torch.float16, torch.bfloat16]:
                for i in range(test_loop):
                    dist.barrier(group=group)
                    inp1 = torch.randint(
                        1, 16, (sz,), dtype=dtype, device=torch.cuda.current_device()
                    )
                    inp_clone = inp1.clone()
                    out1 = torch.empty_like(inp1)

                    custom_ops.custom_reduce(worker_state["custom_ptr"], inp1, out1)

                    dist.barrier(group=group)

                    dist.all_reduce(inp_clone, group=group)

                    dist.barrier(group=group)
                    torch.testing.assert_close(out1, inp_clone, rtol=1e-3, atol=1e-3)

    finally:
        if "custom_ptr" in worker_state and worker_state["custom_ptr"]:
            custom_ops.custom_dispose(worker_state["custom_ptr"])
        free_shared_buffer(worker_state.get("buffer_ptrs", []), group)
        free_shared_buffer(worker_state.get("tmp_result_buffer_ptrs", []), group)
        free_shared_buffer(worker_state.get("barrier_in_ptrs", []), group)
        free_shared_buffer(worker_state.get("barrier_out_ptrs", []), group)
        if dist.is_initialized():
            dist.destroy_process_group()


@ray.remote(num_gpus=1, max_calls=1)
def performance_worker(rank, world_size, distributed_init_port):
    group = init_distributed_env(world_size, rank, distributed_init_port)
    worker_state = {}

    try:
        worker_state["vllm_meta_ptrs"] = create_shared_buffer(
            vllm_ops.meta_size() + VLLM_MAX_SIZE, group=group
        )
        worker_state["vllm_buffer_ptrs"] = create_shared_buffer(
            VLLM_MAX_SIZE, group=group
        )
        worker_state["vllm_rank_data"] = torch.empty(
            VLLM_MAX_SIZE, dtype=torch.uint8, device=torch.cuda.current_device()
        )
        worker_state["vllm_ptr"] = vllm_ops.init_custom_ar(
            worker_state["vllm_meta_ptrs"], worker_state["vllm_rank_data"], rank, True
        )
        vllm_ops.register_buffer(
            worker_state["vllm_ptr"], worker_state["vllm_buffer_ptrs"]
        )

        worker_state["buffer_ptrs"] = create_shared_buffer(BUFFER_MAX_SIZE, group=group)
        worker_state["tmp_result_buffer_ptrs"] = create_shared_buffer(
            BUFFER_MAX_SIZE, group=group
        )
        worker_state["barrier_in_ptrs"] = create_shared_buffer(
            BARRIER_MAX_SIZE, group=group
        )
        worker_state["barrier_out_ptrs"] = create_shared_buffer(
            BARRIER_MAX_SIZE, group=group
        )
        worker_state["rank_data"] = torch.empty(
            BUFFER_MAX_SIZE, dtype=torch.uint8, device=torch.cuda.current_device()
        )
        worker_state["custom_ptr"] = custom_ops.init_custom_reduce(
            rank,
            world_size,
            worker_state["rank_data"],
            worker_state["buffer_ptrs"],
            worker_state["tmp_result_buffer_ptrs"],
            worker_state["barrier_in_ptrs"],
            worker_state["barrier_out_ptrs"],
        )

        dist.barrier(group=group)

        for sz in TEST_SIZES:
            inp1 = torch.randint(
                1, 16, (sz,), dtype=torch.float32, device=torch.cuda.current_device()
            )
            out1 = torch.empty_like(inp1)
            test_loop = 100

            torch.cuda.synchronize()
            dist.barrier(group=group)
            start_custom = time.time()
            for _ in range(test_loop):
                custom_ops.custom_reduce(worker_state["custom_ptr"], inp1, out1)
            torch.cuda.synchronize()
            dist.barrier(group=group)
            elapse_custom = time.time() - start_custom

            torch.cuda.synchronize()
            dist.barrier(group=group)
            start_vllm = time.time()
            for _ in range(test_loop):
                vllm_ops.all_reduce(
                    worker_state["vllm_ptr"],
                    inp1,
                    out1,
                    worker_state["vllm_buffer_ptrs"][rank],
                    VLLM_MAX_SIZE,
                )
            torch.cuda.synchronize()
            dist.barrier(group=group)
            elapse_vllm = time.time() - start_vllm

            if rank == 0:
                logger.warning(
                    f"PERF: sz={sz}, world={world_size}, "
                    f"vllm={elapse_vllm * 1000 / test_loop:.4f}ms, "
                    f"custom={elapse_custom * 1000 / test_loop:.4f}ms"
                )

    finally:
        if "vllm_ptr" in worker_state and worker_state["vllm_ptr"]:
            vllm_ops.dispose(worker_state["vllm_ptr"])
        free_shared_buffer(worker_state.get("vllm_meta_ptrs", []), group)
        free_shared_buffer(worker_state.get("vllm_buffer_ptrs", []), group)

        if "custom_ptr" in worker_state and worker_state["custom_ptr"]:
            custom_ops.custom_dispose(worker_state["custom_ptr"])
        free_shared_buffer(worker_state.get("buffer_ptrs", []), group)
        free_shared_buffer(worker_state.get("tmp_result_buffer_ptrs", []), group)
        free_shared_buffer(worker_state.get("barrier_in_ptrs", []), group)
        free_shared_buffer(worker_state.get("barrier_out_ptrs", []), group)

        if dist.is_initialized():
            dist.destroy_process_group()


class TestCustomAllReduce:

    @pytest.fixture(scope="class", autouse=True)
    def ray_controller(self):
        if not ray.is_initialized():
            ray.init(log_to_driver=False, ignore_reinit_error=True)
        yield
        ray.shutdown()

    def test_correctness(self, request):
        node_world_sizes = WORLD_SIZES
        num_gpus = torch.cuda.device_count()
        logger.info(f"Detected {num_gpus} GPUs for correctness test.")

        for world_size in node_world_sizes:
            if world_size > num_gpus:
                pytest.skip(
                    f"Skipping world_size={world_size}, requires {world_size} GPUs, found {num_gpus}"
                )
                continue

            logger.info(f"Running correctness test with world_size={world_size}")
            distributed_init_port = get_open_port()
            refs = [
                correctness_worker.remote(rank, world_size, distributed_init_port)
                for rank in range(world_size)
            ]
            try:
                ray.get(refs)
                logger.info(f"Correctness test PASSED for world_size={world_size}")
            except Exception as e:
                logger.error(
                    f"Correctness test FAILED for world_size={world_size}: {e}"
                )
                pytest.fail(f"Correctness test failed for world_size={world_size}: {e}")

    def test_performance(self, request):
        node_world_sizes = WORLD_SIZES
        num_gpus = torch.cuda.device_count()
        logger.info(f"Detected {num_gpus} GPUs for performance test.")

        for world_size in node_world_sizes:
            if world_size > num_gpus:
                pytest.skip(
                    f"Skipping world_size={world_size}, requires {world_size} GPUs, found {num_gpus}"
                )
                continue

            logger.info(f"Running performance test with world_size={world_size}")
            distributed_init_port = get_open_port()
            refs = [
                performance_worker.remote(rank, world_size, distributed_init_port)
                for rank in range(world_size)
            ]
            try:
                ray.get(refs)
                logger.info(f"Performance test COMPLETED for world_size={world_size}")
            except Exception as e:
                logger.error(
                    f"Performance test FAILED for world_size={world_size}: {e}"
                )
                pytest.fail(f"Performance test failed for world_size={world_size}: {e}")


if __name__ == "__main__":
    pytest.main([__file__])
