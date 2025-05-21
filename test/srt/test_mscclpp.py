"""For Now, MSCCL is only supported on TP16 and TP8 case

export WORLD_SIZE=1
export RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=12345

nsys profile \
-o msccl \
-s none \
--force-overwrite=true \
--capture-range=cudaProfilerApi \
--capture-range-end=stop \
--trace=cuda,nvtx \
torchrun --nproc_per_node gpu \
--nnodes $WORLD_SIZE \
--node_rank $RANK \
--master_addr $MASTER_ADDR \
--master_port $MASTER_PORT test_msccl.py

if [[ $RANK -eq 0  ]]; then
    ray start --block --head --port=6379 &
    python3 test_msccl.py;
else
    ray start --block --address=${MASTER_ADDR}:6379;
fi
"""

import itertools
import os
import random
import socket
import unittest
from contextlib import contextmanager, nullcontext
from typing import Any, List, Optional, Union

import cupy as cp
import ray
import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup, ReduceOp

from sglang.srt.distributed import init_distributed_environment
from sglang.srt.distributed.device_communicators.custom_all_reduce import (
    CustomAllreduce,
)
from sglang.srt.distributed.device_communicators.pymsccl import PyMscclCommunicator
from sglang.srt.distributed.device_communicators.pynccl import PyNcclCommunicator
from sglang.srt.distributed.parallel_state import (
    get_tensor_model_parallel_group,
    graph_capture,
    initialize_model_parallel,
)
from sglang.srt.distributed.utils import StatelessProcessGroup


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

    distributed_init_port = get_open_port()
    refs = []
    for rank in range(world_size):
        print(f"{cls}, {world_size}, {rank}, {distributed_init_port}", flush=True)
        refs.append(test_target.remote(cls, world_size, rank, distributed_init_port))
    ray.get(refs)

    ray.shutdown()


def torch_allreduce(torch_input: torch.Tensor, group: ProcessGroup) -> torch.Tensor:
    dist.all_reduce(torch_input, group=group)
    return torch_input


def msccl_allreduce(
    msccl_input: torch.Tensor, msccl_comm: PyMscclCommunicator
) -> torch.Tensor:
    return msccl_comm.all_reduce(msccl_input)


def pynccl_allreduce(
    msccl_input: torch.Tensor, pynccl_comm: PyNcclCommunicator
) -> torch.Tensor:
    pynccl_comm.all_reduce(msccl_input)
    return msccl_input


class TestMSCCLAllReduce(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        random.seed(42)
        # 512B to 32MB
        # cls.test_sizes = [512, 4096, 32768, 262144, 2097152, 16777216, 33554432]
        # cls.world_sizes = [2, 4, 6, 8]
        cls.test_sizes = [4096, 524288]
        cls.test_dtypes = [torch.bfloat16]
        cls.world_sizes = [8]
        cls.test_loop = 100
        cls.warmup_loop = 2
        cls.graph_loop = 10

    def _init_dist(self, world_size, rank, distributed_init_port):
        del os.environ["CUDA_VISIBLE_DEVICES"]
        device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
        torch.cuda.set_device(device)
        master_addr = os.getenv("MASTER_ADDR", None)
        assert (
            master_addr is not None
        ), "MASTER_ADDR must be set to enable multi node init"
        distributed_init_method = f"tcp://{master_addr}:{distributed_init_port}"
        init_distributed_environment(
            world_size=world_size,
            rank=rank,
            distributed_init_method=distributed_init_method,
            local_rank=rank % 8,
        )
        initialize_model_parallel(tensor_model_parallel_size=world_size)
        return device

    def _bench_graph_time(self, func, inp_randn):
        graph_input = inp_randn.clone()
        with graph_capture() as graph_capture_context:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=graph_capture_context.stream):
                for _ in range(self.graph_loop):
                    graph_out = func(graph_input)

        graph.replay()
        func_output = graph_out.clone()

        for _ in range(5):
            graph.replay()
        torch.cuda.synchronize()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        latencies: List[float] = []
        for _ in range(self.test_loop):
            torch.cuda.synchronize()
            dist.barrier()
            start_event.record()
            graph.replay()
            end_event.record()
            end_event.synchronize()
            latencies.append(start_event.elapsed_time(end_event))
        func_cost_us = sum(latencies) / len(latencies) / self.graph_loop * 1000
        graph.reset()
        return func_output, func_cost_us

    def _bench_eager_time(self, func, inp_randn):
        eager_input = inp_randn.clone()
        eager_output = func(eager_input)
        func_output = eager_output.clone()

        for _ in range(self.warmup_loop):
            func(eager_input)
        torch.cuda.synchronize()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record()
        for _ in range(self.test_loop):
            func(eager_input)
        end_event.record()
        torch.cuda.synchronize()
        func_cost_us = start_event.elapsed_time(end_event) / self.test_loop * 1000

        return func_output, func_cost_us

    def test_allreduce(self):
        for world_size in self.world_sizes:
            multi_process_parallel(world_size, self, self._allreduce)

    @ray.remote(num_gpus=1, max_calls=1)
    def _allreduce(self, world_size, rank, distributed_init_port):
        device = self._init_dist(world_size, rank, distributed_init_port)
        print(f"[{rank}] _init_dist finished")
        group = get_tensor_model_parallel_group().device_group
        cpu_group = get_tensor_model_parallel_group().cpu_group
        msccl_ar = PyMscclCommunicatorCudaAPI(
            group,
            cpu_group,
            device,
            dtype=torch.bfloat16,
        )
        print(f"[{rank}] msccl ar finished")

        for sz, dtype in itertools.product(self.test_sizes, self.test_dtypes):
            print(f"[{rank}] sz={sz}, dtype={dtype} start")
            inp_randn = torch.randint(1, 16, (sz,), dtype=dtype, device=device)
            # torch_eager_output, torch_eager_time = self._bench_eager_time(
            #     lambda inp: torch_allreduce(inp, group), inp_randn
            # )
            # print(f"[{rank}] sz={sz}, dtype={dtype} torch_eager cost {torch_eager_time}")
            msccl_eager_output, msccl_eager_time = self._bench_eager_time(
                lambda inp: msccl_allreduce(inp, msccl_ar), inp_randn
            )
            print(
                f"[{rank}] sz={sz}, dtype={dtype} msccl_eager cost {msccl_eager_time}: {msccl_eager_output}"
            )
            # graph_capture_context = graph_capture()
            # torch_graph_output, torch_graph_time = self._bench_graph_time(
            #     lambda inp: torch_allreduce(inp, group), inp_randn, graph_capture_context, msccl_ar.stream
            # )
            # print(f"[{rank}] sz={sz}, dtype={dtype} torch_graph cost {torch_graph_time}: {torch_graph_output}")

            msccl_graph_capture_context = msccl_ar.change_state(
                enable=True, stream=msccl_ar.stream
            )
            msccl_graph_output, msccl_graph_time = self._bench_graph_time(
                lambda inp: msccl_allreduce(inp, msccl_ar),
                inp_randn,
                msccl_graph_capture_context,
                msccl_ar.stream,
            )
            print(
                f"[{rank}] sz={sz}, dtype={dtype} msccl_graph cost {msccl_graph_time}: {msccl_graph_output}"
            )
            # torch.testing.assert_close(torch_graph_output, msccl_graph_output)
            # torch.testing.assert_close(torch_eager_output, msccl_eager_output)
            # print(f"[{rank}] sz={sz}, dtype={dtype}:", end="\t")
            # print(f"torch_graph_time={torch_graph_time}", end="\t")
            # print(f"msccl_graph_time={msccl_graph_time}", end="\t")
            # print(f"torch_eager_time={torch_eager_time}", end="\t")
            # print(f"msccl_eager_time={msccl_eager_time}")
        print(f"✅ rank{rank} world_size={world_size} all cases passed", flush=True)


def get_torch_prof_ctx(do_prof: bool):
    ctx = (
        torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            with_stack=True,
        )
        if do_prof
        else nullcontext()
    )
    return ctx


if __name__ == "__main__":
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    # unittest.main()
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    world, world_size = dist.group.WORLD, dist.get_world_size()
    rank = dist.get_rank()
    torch.cuda.set_device(rank % 8)
    cp.cuda.Device(rank % 8).use()
    device = torch.cuda.current_device()
    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=rank % 8,
    )
    print(f"[{rank}] init_distributed_environment finished")
    initialize_model_parallel(tensor_model_parallel_size=world_size)
    print(f"[{rank}] initialize_model_parallel finished")
    group = get_tensor_model_parallel_group().device_group
    cpu_group = get_tensor_model_parallel_group().cpu_group
    pynccl_comm = get_tensor_model_parallel_group().pynccl_comm
    pymsccl_comm = get_tensor_model_parallel_group().pymsccl_comm
    dist.barrier()
    tester = TestMSCCLAllReduce()
    tester.setUpClass()
    profile = True
    ctx = get_torch_prof_ctx(profile)
    with ctx:
        dtype = torch.bfloat16
        torch.cuda.cudart().cudaProfilerStart()
        for i in range(10, 20):
            sz = 2**i
            inp_randn = torch.randint(1, 16, (sz,), dtype=dtype, device=device)
            torch.cuda.nvtx.range_push("torch eager")
            torch_eager_output, torch_eager_time = tester._bench_eager_time(
                lambda inp: torch_allreduce(inp, group), inp_randn
            )
            torch.cuda.nvtx.range_pop()
            if rank == 0:
                print(
                    f"[{rank}] sz={sz}, dtype={dtype} torch_eager cost {torch_eager_time} {torch_eager_output}"
                )

            torch.cuda.nvtx.range_push("pynccl graph")
            # since pynccl is inplace op, this return result is not correct if graph loop > 1
            _, pynccl_graph_time = tester._bench_graph_time(
                lambda inp: pynccl_allreduce(inp, pynccl_comm), inp_randn
            )
            torch.cuda.nvtx.range_pop()
            if rank == 0:
                print(
                    f"[{rank}] sz={sz}, dtype={dtype} pynccl_graph cost {pynccl_graph_time}"
                )

            torch.cuda.nvtx.range_push("msccl graph")
            msccl_graph_output, msccl_graph_time = tester._bench_graph_time(
                lambda inp: msccl_allreduce(inp, pymsccl_comm), inp_randn
            )
            torch.cuda.nvtx.range_pop()
            if rank == 0:
                print(
                    f"[{rank}] sz={sz}, dtype={dtype} msccl_graph cost {msccl_graph_time}: {msccl_graph_output}"
                )
            torch.cuda.cudart().cudaProfilerStop()

            torch.testing.assert_close(torch_eager_output, msccl_graph_output)
            # torch.testing.assert_close(torch_eager_output, msccl_eager_output)

            if rank == 0:
                print(f"[{rank}] sz={sz}, dtype={dtype}: PASS!")

    if profile:
        prof_dir = f"prof/msccl"
        os.makedirs(prof_dir, exist_ok=True)
        ctx.export_chrome_trace(f"{prof_dir}/trace_rank{dist.get_rank()}.json.gz")
