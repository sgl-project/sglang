"""Benchmark multi-node MSCCL++ AllGather against NCCL."""

import argparse
import os

import torch
import torch.distributed as dist

from sglang.srt.distributed import init_distributed_environment
from sglang.srt.distributed.parallel_state import (
    cleanup_dist_env_and_memory,
    get_tensor_model_parallel_group,
    initialize_model_parallel,
    set_mscclpp_all_reduce,
)


def _max_across_ranks(value: float, group) -> float:
    tensor = torch.tensor([value], dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX, group=group)
    return tensor.item()


def _time_eager(func, group, warmup: int = 2, repeat: int = 10) -> float:
    for _ in range(warmup):
        func()
    torch.cuda.synchronize()
    dist.barrier(group=group)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        func()
    end.record()
    end.synchronize()
    return _max_across_ranks(start.elapsed_time(end) * 1000 / repeat, group)


def _time_cuda_graph(
    func,
    group,
    warmup: int = 2,
    graph_loop: int = 10,
    repeat: int = 10,
) -> float:
    current_stream = torch.cuda.current_stream()
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(current_stream)
    with torch.cuda.stream(capture_stream):
        func(capture_stream)
    capture_stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        for _ in range(graph_loop):
            func(capture_stream)

    with torch.cuda.stream(capture_stream):
        for _ in range(warmup):
            graph.replay()
    capture_stream.synchronize()

    dist.barrier(group=group)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record(capture_stream)
    with torch.cuda.stream(capture_stream):
        for _ in range(repeat):
            graph.replay()
    end.record(capture_stream)
    end.synchronize()
    torch.cuda.current_stream().wait_stream(capture_stream)
    return _max_across_ranks(
        start.elapsed_time(end) * 1000 / (repeat * graph_loop),
        group,
    )


def _validate_cuda_graph_updates(
    func,
    reference_func,
    input_tensor: torch.Tensor,
    actual: torch.Tensor,
    expected: torch.Tensor,
    group,
) -> None:
    current_stream = torch.cuda.current_stream()
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(current_stream)
    with torch.cuda.stream(capture_stream):
        func(capture_stream)
    capture_stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        func(capture_stream)

    input_tensor.neg_()
    reference_func()
    current_stream.synchronize()
    capture_stream.wait_stream(current_stream)
    with torch.cuda.stream(capture_stream):
        graph.replay()
    capture_stream.synchronize()

    if not torch.equal(actual, expected):
        raise AssertionError("MSCCL++ CUDA graph replay used stale AllGather data")
    dist.barrier(group=group)


def _message_sizes(min_bytes: int, max_bytes: int):
    size = 1 << (min_bytes - 1).bit_length()
    while size <= max_bytes:
        yield size
        size <<= 1


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-bytes", type=int, default=1 << 10)
    parser.add_argument("--max-bytes", type=int, default=8 << 20)
    args = parser.parse_args()
    if not (1 << 10) <= args.min_bytes <= args.max_bytes <= (8 << 20):
        parser.error("Require 1 KiB <= min-bytes <= max-bytes <= 8 MiB")

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()
    set_mscclpp_all_reduce(True)
    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
    )
    initialize_model_parallel(tensor_model_parallel_size=world_size)
    tp_group = get_tensor_model_parallel_group()
    gpu_group = tp_group.device_group
    cpu_group = tp_group.cpu_group
    pynccl_comm = tp_group.pynccl_comm
    mscclpp_comm = tp_group.pymscclpp_comm
    assert pynccl_comm is not None
    assert mscclpp_comm is not None
    if args.min_bytes % world_size != 0 or args.max_bytes % world_size != 0:
        parser.error("Total message sizes must be divisible by WORLD_SIZE")

    if not mscclpp_comm.allgather_best_configs:
        raise RuntimeError("No tuned MSCCL++ AllGather configuration is available")

    result = []

    for total_bytes in _message_sizes(args.min_bytes, args.max_bytes):
        input_bytes = total_bytes // world_size
        if input_bytes % torch.bfloat16.itemsize != 0:
            continue
        elements = input_bytes // torch.bfloat16.itemsize
        input_tensor = (
            torch.arange(
                elements,
                dtype=torch.long,
                device=device,
            )
            % 193
            + rank
        ).to(torch.bfloat16)
        nccl_output = torch.empty(
            elements * world_size,
            dtype=input_tensor.dtype,
            device=device,
        )
        mscclpp_output = torch.empty_like(nccl_output)

        def torch_nccl():
            dist.all_gather_into_tensor(
                nccl_output,
                input_tensor,
                group=gpu_group,
            )

        def py_nccl():
            pynccl_comm.all_gather(nccl_output, input_tensor)

        def mscclpp():
            mscclpp_comm.all_gather(mscclpp_output, input_tensor)

        def mscclpp_on_stream(stream):
            mscclpp_comm.all_gather(
                mscclpp_output,
                input_tensor,
                stream=stream,
            )

        def py_nccl_on_stream(stream):
            pynccl_comm.cp_all_gather_into_tensor(
                nccl_output,
                input_tensor,
                stream=stream,
            )

        torch_nccl()
        mscclpp()
        py_nccl()
        torch.cuda.synchronize()
        if not torch.equal(mscclpp_output, nccl_output):
            raise AssertionError(
                f"MSCCL++ AllGather mismatch at {input_bytes} input bytes"
            )
        _validate_cuda_graph_updates(
            mscclpp_on_stream,
            torch_nccl,
            input_tensor,
            mscclpp_output,
            nccl_output,
            cpu_group,
        )

        nccl_eager_us = _time_eager(torch_nccl, cpu_group)
        mscclpp_eager_us = _time_eager(mscclpp, cpu_group)
        mscclpp_graph_us = _time_cuda_graph(mscclpp_on_stream, cpu_group)
        pynccl_graph_us = _time_cuda_graph(py_nccl_on_stream, cpu_group)

        result.append(
            {
                "msg_size": human_readable_size(total_bytes),
                "torch eager time": nccl_eager_us,
                "msccl eager time": mscclpp_eager_us,
                "msccl graph time": mscclpp_graph_us,
                "pynccl graph time": pynccl_graph_us,
            }
        )
        if rank == 0:
            print(
                f"sz={total_bytes // input_tensor.element_size()}, "
                f"dtype={input_tensor.dtype}: correctness check PASS!"
            )

        torch.cuda.empty_cache()

    if rank == 0:
        print_markdown_table(result)

    mscclpp_comm.destroy()
    dist.barrier()
    cleanup_dist_env_and_memory()


if __name__ == "__main__":
    main()
