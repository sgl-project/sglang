"""For Now, MSCCL is only supported on TP16 and TP8 case

export WORLD_SIZE=1
export RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=12345

torchrun --nproc_per_node gpu \
--nnodes $WORLD_SIZE \
--node_rank $RANK \
--master_addr $MASTER_ADDR \
--master_port $MASTER_PORT benchmark_mscclpp.py
"""
import os
from contextlib import contextmanager, nullcontext
from typing import Any, List, Optional, Union

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup, ReduceOp

from sglang.srt.distributed import init_distributed_environment
from sglang.srt.distributed.device_communicators.custom_all_reduce import (
    CustomAllreduce,
)
from sglang.srt.distributed.device_communicators.pymscclpp import PyMscclppCommunicator
from sglang.srt.distributed.device_communicators.pynccl import PyNcclCommunicator
from sglang.srt.distributed.parallel_state import (
    get_tensor_model_parallel_group,
    graph_capture,
    initialize_model_parallel,
    set_mscclpp_all_reduce
)

def torch_allreduce(torch_input: torch.Tensor, group: ProcessGroup) -> torch.Tensor:
    dist.all_reduce(torch_input, group=group)
    return torch_input


def msccl_allreduce(
    msccl_input: torch.Tensor, msccl_comm: PyMscclppCommunicator
) -> torch.Tensor:
    return msccl_comm.all_reduce(msccl_input)


def pynccl_allreduce(
    msccl_input: torch.Tensor, pynccl_comm: PyNcclCommunicator
) -> torch.Tensor:
    pynccl_comm.all_reduce(msccl_input)
    return msccl_input

def _bench_graph_time(func, inp_randn, warmup_loop=2, graph_loop=10, test_loop=10):
    graph_input = inp_randn.clone()
    with graph_capture() as graph_capture_context:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=graph_capture_context.stream):
            for _ in range(graph_loop):
                graph_out = func(graph_input)

    graph.replay()
    func_output = graph_out.clone()

    for _ in range(warmup_loop):
        graph.replay()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    latencies: List[float] = []
    for _ in range(test_loop):
        torch.cuda.synchronize()
        dist.barrier()
        start_event.record()
        graph.replay()
        end_event.record()
        end_event.synchronize()
        latencies.append(start_event.elapsed_time(end_event))
    func_cost_us = sum(latencies) / len(latencies) / graph_loop * 1000
    graph.reset()
    return func_output, func_cost_us

def _bench_eager_time(func, inp_randn, warmup_loop=2, test_loop=10):
    eager_input = inp_randn.clone()
    eager_output = func(eager_input)
    func_output = eager_output.clone()

    for _ in range(warmup_loop):
        func(eager_input)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start_event.record()
    for _ in range(test_loop):
        func(eager_input)
    end_event.record()
    torch.cuda.synchronize()
    func_cost_us = start_event.elapsed_time(end_event) / test_loop * 1000

    return func_output, func_cost_us


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
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    world, world_size = dist.group.WORLD, dist.get_world_size()
    rank = dist.get_rank()
    torch.cuda.set_device(rank % 8)
    device = torch.cuda.current_device()
    set_mscclpp_all_reduce(True)
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
    pymscclpp_comm = get_tensor_model_parallel_group().pymscclpp_comm
    dist.barrier()
    profile = False
    ctx = get_torch_prof_ctx(profile)
    with ctx:
        dtype = torch.bfloat16
        torch.cuda.cudart().cudaProfilerStart()
        for i in range(10, 20):
            sz = 2**i
            inp_randn = torch.randint(1, 16, (sz,), dtype=dtype, device=device)
            torch_eager_output, torch_eager_time = _bench_eager_time(
                lambda inp: torch_allreduce(inp, group), inp_randn
            )
            if rank == 0:
                print(
                    f"[{rank}] sz={sz}, dtype={dtype} torch_eager cost {torch_eager_time} {torch_eager_output}"
                )
            
            msccl_eager_output, msccl_eager_time = _bench_eager_time(
                lambda inp: msccl_allreduce(inp, pymscclpp_comm), inp_randn
            )
            if rank == 0:
                print(
                    f"[{rank}] sz={sz}, dtype={dtype} msccl_eager cost {msccl_eager_time}: {msccl_eager_output}"
                )

            # since pynccl is inplace op, this return result is not correct if graph loop > 1
            _, pynccl_graph_time = _bench_graph_time(
                lambda inp: pynccl_allreduce(inp, pynccl_comm), inp_randn
            )
            if rank == 0:
                print(
                    f"[{rank}] sz={sz}, dtype={dtype} pynccl_graph cost {pynccl_graph_time}"
                )

            msccl_graph_output, msccl_graph_time = _bench_graph_time(
                lambda inp: msccl_allreduce(inp, pymscclpp_comm), inp_randn
            )
            if rank == 0:
                print(
                    f"[{rank}] sz={sz}, dtype={dtype} msccl_graph cost {msccl_graph_time}: {msccl_graph_output}"
                )
            torch.cuda.cudart().cudaProfilerStop()

            torch.testing.assert_close(torch_eager_output, msccl_graph_output)
            torch.testing.assert_close(torch_eager_output, msccl_eager_output)

            if rank == 0:
                print(f"[{rank}] sz={sz}, dtype={dtype}: PASS!")

    if profile:
        prof_dir = f"prof/msccl"
        os.makedirs(prof_dir, exist_ok=True)
        ctx.export_chrome_trace(f"{prof_dir}/trace_rank{dist.get_rank()}.json.gz")
