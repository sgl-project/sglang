"""For Now, TORCH_SYMM_MEM is only supported on following limited tp case

SM90: {
    2: 64 * MiB,  # 64 MB
    4: 64 * MiB,  # 64 MB
    6: 128 * MiB,  # 128 MB
    8: 128 * MiB,  # 128 MB
},
SM100: {
    2: 64 * MiB,  # 64 MB
    4: 64 * MiB,  # 64 MB
    6: 128 * MiB,  # 128 MB
    8: 128 * MiB,  # 128 MB
}

export WORLD_SIZE=8
export RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=12345

torchrun --nproc_per_node gpu \
--nnodes $WORLD_SIZE \
--node_rank $RANK \
--master_addr $MASTER_ADDR \
--master_port $MASTER_PORT ./benchmark/kernels/all_reduce/benchmark_torch_symm_mem.py
"""

import os
from contextlib import nullcontext
from typing import List

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from sglang.srt.distributed import init_distributed_environment
from sglang.srt.distributed.device_communicators.pynccl import PyNcclCommunicator
from sglang.srt.distributed.device_communicators.torch_symm_mem import (
    TorchSymmMemCommunicator,
)
from sglang.srt.distributed.parallel_state import (
    get_tensor_model_parallel_group,
    graph_capture,
    initialize_model_parallel,
    set_torch_symm_mem_all_reduce,
)

# CI environment detection
IS_CI = (
    os.getenv("CI", "false").lower() == "true"
    or os.getenv("GITHUB_ACTIONS", "false").lower() == "true"
)


def torch_allreduce(torch_input: torch.Tensor, group: ProcessGroup) -> torch.Tensor:
    dist.all_reduce(torch_input, group=group)
    return torch_input


def torch_symm_mem_allreduce(
    torch_symm_mem_input: torch.Tensor, torch_symm_mem_comm: TorchSymmMemCommunicator
) -> torch.Tensor:
    return torch_symm_mem_comm.all_reduce(torch_symm_mem_input)


def pynccl_allreduce(
    pynccl_input: torch.Tensor, pynccl_comm: PyNcclCommunicator
) -> torch.Tensor:
    pynccl_comm.all_reduce(pynccl_input)
    return pynccl_input


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


def human_readable_size(size, decimal_places=1):
    for unit in ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]:
        if size < 1024.0 or unit == "PiB":
            break
        size /= 1024.0
    return f"{size:.{decimal_places}f} {unit}"


try:
    from tabulate import tabulate
except ImportError:
    print("tabulate not installed, skipping table printing")
    tabulate = None


def print_markdown_table(data):
    if tabulate is not None:
        print(tabulate(data, headers="keys", tablefmt="github"))
        return
    headers = data[0].keys()
    header_row = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"
    rows = []
    for item in data:
        row = "| " + " | ".join(str(item[key]) for key in headers) + " |"
        rows.append(row)
    markdown_table = "\n".join([header_row, separator] + rows)
    print(markdown_table)


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
    set_torch_symm_mem_all_reduce(True)
    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=rank % 8,
    )
    initialize_model_parallel(tensor_model_parallel_size=world_size)
    group = get_tensor_model_parallel_group().device_group
    cpu_group = get_tensor_model_parallel_group().cpu_group
    pynccl_comm = get_tensor_model_parallel_group().pynccl_comm
    torch_symm_mem_comm = get_tensor_model_parallel_group().torch_symm_mem_comm
    dist.barrier()
    profile = False
    dtype = torch.bfloat16
    ctx = get_torch_prof_ctx(profile)
    result = []

    with ctx:
        if IS_CI:
            i_range = range(10, 11)
        else:
            i_range = range(10, 20)
        for i in i_range:
            sz = 2**i
            if sz * dtype.itemsize > 2**24:
                break
            inp_randn = torch.randint(1, 16, (sz,), dtype=dtype, device=device)

            memory = torch.empty_like(inp_randn)
            memory_out = torch.empty_like(memory)
            torch_eager_output, torch_eager_time = _bench_eager_time(
                lambda inp: torch_allreduce(inp, group), inp_randn
            )
            symm_mem_eager_output, symm_mem_eager_time = _bench_eager_time(
                lambda inp: torch_symm_mem_allreduce(inp, torch_symm_mem_comm),
                inp_randn,
            )
            symm_mem_graph_output, symm_mem_graph_time = _bench_graph_time(
                lambda inp: torch_symm_mem_allreduce(inp, torch_symm_mem_comm),
                inp_randn,
            )
            # since pynccl is inplace op, this return result is not correct if graph loop > 1
            _, pynccl_graph_time = _bench_graph_time(
                lambda inp: pynccl_allreduce(inp, pynccl_comm), inp_randn
            )
            torch.testing.assert_close(torch_eager_output, symm_mem_graph_output)
            torch.testing.assert_close(torch_eager_output, symm_mem_eager_output)
            result.append(
                {
                    "msg_size": human_readable_size(inp_randn.nbytes),
                    "torch eager time": torch_eager_time,
                    "symm mem eager time": symm_mem_eager_time,
                    "symm mem graph time": symm_mem_graph_time,
                    "pynccl graph time": pynccl_graph_time,
                }
            )
            if rank == 0:
                print(f"sz={sz}, dtype={dtype}: correctness check PASS!")
    if rank == 0:
        print_markdown_table(result)
    if profile:
        prof_dir = f"prof/torch_symm_mem"
        os.makedirs(prof_dir, exist_ok=True)
        ctx.export_chrome_trace(f"{prof_dir}/trace_rank{dist.get_rank()}.json.gz")
