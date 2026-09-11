"""For Now, MSCCL is only supported on TP16 and TP8 case

export WORLD_SIZE=1
export RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=12345

torchrun --nproc_per_node gpu \
--nnodes $WORLD_SIZE \
--node_rank $RANK \
--master_addr $MASTER_ADDR \
--master_port $MASTER_PORT benchmark/kernels/all_reduce/benchmark_mscclpp.py
"""

import argparse
import os
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from sglang.srt.runtime_context import publish
from sglang.srt.server_args import ServerArgs
from sglang.srt.distributed import init_distributed_environment
from sglang.srt.distributed.device_communicators.pymscclpp import PyMscclppCommunicator
from sglang.srt.distributed.device_communicators.pynccl import PyNcclCommunicator
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.distributed.parallel_state import (
    cleanup_dist_env_and_memory,
    get_tensor_model_parallel_group,
    graph_capture,
    initialize_model_parallel,
    set_mscclpp_all_reduce,
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


def _bench_graph_time(
    func,
    inp_randn,
    tp_group,
    warmup_loop=2,
    graph_loop=100,
    test_loop=10,
):
    with use_symmetric_memory(tp_group):
        graph_input = inp_randn.clone()
    with graph_capture() as graph_capture_context:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=graph_capture_context.stream):
            for _ in range(graph_loop):
                graph_input.copy_(inp_randn)
                graph_out = func(graph_input)

    graph.replay()
    func_output = graph_out.clone()

    for _ in range(warmup_loop):
        graph.replay()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    latencies = []
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


def _bench_eager_time(func, inp_randn, tp_group, warmup_loop=2, test_loop=10):
    with use_symmetric_memory(tp_group):
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

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--enable-symm-mem",
        action="store_true",
        help="Allocate collective inputs from NCCL symmetric memory.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl", device_id=torch.device("cuda", local_rank)
        )
    world, world_size = dist.group.WORLD, dist.get_world_size()
    rank = dist.get_rank()
    set_mscclpp_all_reduce(True)
    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
    )
    publish(
        ServerArgs(model_path="dummy", enable_symm_mem=args.enable_symm_mem),
        role="scheduler",
    )
    initialize_model_parallel(
        tensor_model_parallel_size=world_size,
        enable_symm_mem=args.enable_symm_mem,
    )
    tp_group = get_tensor_model_parallel_group()
    group = tp_group.device_group
    pynccl_comm = tp_group.pynccl_comm
    pymscclpp_comm = tp_group.pymscclpp_comm
    dist.barrier()
    profile = False
    dtype = torch.bfloat16
    ctx = get_torch_prof_ctx(profile)
    result = []
    max_message_bytes = 1 << 20

    with ctx:
        for i in range(10, 20):
            sz = 2**i
            if sz * dtype.itemsize > max_message_bytes:
                break
            inp_randn = torch.randint(0, 2, (sz,), dtype=dtype, device=device)

            torch_eager_output, torch_eager_time = _bench_eager_time(
                lambda inp: torch_allreduce(inp, group), inp_randn, tp_group
            )
            msccl_eager_output, msccl_eager_time = _bench_eager_time(
                lambda inp: msccl_allreduce(inp, pymscclpp_comm),
                inp_randn,
                tp_group,
            )
            msccl_graph_output, msccl_graph_time = _bench_graph_time(
                lambda inp: msccl_allreduce(inp, pymscclpp_comm),
                inp_randn,
                tp_group,
            )
            # since pynccl is inplace op, this return result is not correct if graph loop > 1
            _, pynccl_graph_time = _bench_graph_time(
                lambda inp: pynccl_allreduce(inp, pynccl_comm),
                inp_randn,
                tp_group,
            )
            torch.testing.assert_close(torch_eager_output, msccl_graph_output)
            torch.testing.assert_close(torch_eager_output, msccl_eager_output)
            result.append(
                {
                    "msg_size": human_readable_size(inp_randn.nbytes),
                    "torch eager time": torch_eager_time,
                    "msccl eager time": msccl_eager_time,
                    "msccl graph time": msccl_graph_time,
                    "pynccl graph time": pynccl_graph_time,
                }
            )
            if rank == 0:
                print(f"sz={sz}, dtype={dtype}: correctness check PASS!")
    if rank == 0:
        print_markdown_table(result)
    if profile:
        prof_dir = f"prof/msccl"
        os.makedirs(prof_dir, exist_ok=True)
        ctx.export_chrome_trace(f"{prof_dir}/trace_rank{dist.get_rank()}.json.gz")

    pymscclpp_comm.destroy()
    dist.barrier()
    cleanup_dist_env_and_memory()
