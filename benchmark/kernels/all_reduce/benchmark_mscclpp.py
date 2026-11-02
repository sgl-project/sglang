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
from contextlib import ExitStack, contextmanager, nullcontext
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from sglang.srt.distributed import init_distributed_environment
from sglang.srt.distributed.device_communicators.pymscclpp import PyMscclppCommunicator
from sglang.srt.distributed.device_communicators.pynccl import PyNcclCommunicator
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.distributed.parallel_state import (
    cleanup_dist_env_and_memory,
    graph_capture,
    initialize_model_parallel,
    set_mscclpp,
)
from sglang.srt.runtime_context import get_parallel
from sglang.test.test_utils import publish_build_topology


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
    graph_loop=10,
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


def _check_graph_replay_uses_new_input(func, inp_randn, tp_group, group, replays=3):
    # The timing loop replays on unchanged input; replaying one captured graph
    # over new input exposes bugs that stay silent when every replay gets the
    # same input and so expects the same output.
    with use_symmetric_memory(tp_group):
        graph_input = torch.empty_like(inp_randn)
    source = inp_randn.clone()
    with graph_capture() as graph_capture_context:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=graph_capture_context.stream):
            graph_input.copy_(source)
            graph_out = func(graph_input)

    for _ in range(replays):
        source.copy_(torch.randint_like(source, 0, 2))
        expected = torch_allreduce(source.clone(), group)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(graph_out, expected, atol=0, rtol=0)
    graph.reset()


BACKENDS = ("torch-eager", "mscclpp-eager", "mscclpp-graph", "pynccl-graph")
# Backends whose result depends on the MSCCL++ algorithm under test.
MSCCLPP_BACKENDS = ("mscclpp-eager", "mscclpp-graph")


def _dispatched_backend(tp_group, run):
    # A fallback to another backend keeps results correct, so only a dispatch
    # check catches a GroupCoordinator change that bypasses MSCCL++.
    entry_points = (
        (tp_group.ca_comm, "custom_all_reduce", "custom-ar"),
        (tp_group.pcie_ipc_comm, "pcie_ipc_all_reduce", "pcie-ipc"),
        (tp_group.qr_comm, "quick_all_reduce", "quick-ar"),
        (tp_group.pymscclpp_comm, "all_reduce", "mscclpp"),
        (tp_group.torch_symm_mem_comm, "all_reduce", "torch-symm-mem"),
        (tp_group.pynccl_comm, "all_reduce", "pynccl"),
        (tp_group.pynccl_comm, "outplace_all_reduce", "pynccl"),
        (dist, "all_reduce", "torch"),
    )
    with ExitStack() as stack:
        spies = []
        for obj, method, name in entry_points:
            if obj is not None:
                spy = patch.object(obj, method, wraps=getattr(obj, method))
                spies.append((name, stack.enter_context(spy)))
        output = run()
    called = sorted({name for name, spy in spies if spy.called})
    return "+".join(called) or "other", output


def _check_sglang_eager_dispatch(inp_randn, tp_group, expected):
    backend, output = _dispatched_backend(
        tp_group, lambda: tp_group.all_reduce(inp_randn.clone())
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(output, expected, atol=0, rtol=0)
    return backend


def _check_sglang_graph_dispatch(inp_randn, tp_group, expected):
    with use_symmetric_memory(tp_group):
        graph_input = inp_randn.clone()
    graph = torch.cuda.CUDAGraph()
    with graph_capture() as graph_capture_context:

        def capture():
            with torch.cuda.graph(graph, stream=graph_capture_context.stream):
                return tp_group.all_reduce(graph_input)

        backend, graph_out = _dispatched_backend(tp_group, capture)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(graph_out, expected, atol=0, rtol=0)
    graph.reset()
    return backend


@contextmanager
def _record_mscclpp_candidates():
    # PyMscclppCommunicator keeps only the tuned winner per message size; keep
    # every bound candidate so --mscclpp-algorithms can force the others.
    recorded = {}
    original = PyMscclppCommunicator._create_algorithm_candidates

    def record(comm, *args, **kwargs):
        candidates = original(comm, *args, **kwargs)
        recorded[comm] = candidates
        return candidates

    with patch.object(PyMscclppCommunicator, "_create_algorithm_candidates", record):
        yield recorded


def _allreduce_configs_to_check(comm, candidates, selection, nbytes, dtype):
    tuned = comm._get_allreduce_tuned_config(nbytes)
    if tuned is not None and not tuned.supports_dtype(dtype):
        tuned = None
    if "tuned" in selection and tuned is None:
        raise RuntimeError(
            f"No tuned MSCCL++ all-reduce configuration for {nbytes} bytes and {dtype}"
        )
    configs = []
    if tuned is not None and ("tuned" in selection or "all" in selection):
        configs.append(tuned)
    for candidate in candidates:
        if "all" not in selection and candidate.name not in selection:
            continue
        if not candidate.supports_message_size(nbytes):
            continue
        if not candidate.supports_dtype(dtype):
            continue
        if any(config.algorithm is candidate.algorithm for config in configs):
            continue
        configs.append(candidate.select(*candidate.tuning_launches()[0]))
    return tuned, configs


def _config_label(config, tuned):
    label = config.algorithm.name
    launch = config.selected_launch()
    if launch != (0, 0):
        label += f" {launch[0]}x{launch[1]}"
    if config is tuned:
        label += " (tuned)"
    return label


@contextmanager
def _use_allreduce_config(comm, nbytes, config):
    configs = comm._best_configs["allreduce"]
    previous = configs.get(nbytes)
    configs[nbytes] = config
    try:
        yield
    finally:
        if previous is None:
            del configs[nbytes]
        else:
            configs[nbytes] = previous


def _message_sizes(min_bytes: int, max_bytes: int):
    size = 1 << (min_bytes - 1).bit_length()
    while size <= max_bytes:
        yield size
        size <<= 1


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
    parser.add_argument(
        "--dtypes",
        nargs="+",
        choices=["float16", "bfloat16", "float32"],
        default=["bfloat16"],
        help="Data types to benchmark and check.",
    )
    parser.add_argument(
        "--min-bytes",
        type=int,
        default=2 << 10,
        help="Smallest message size in bytes (rounded up to a power of two).",
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=1 << 20,
        help="Largest message size in bytes.",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=BACKENDS,
        default=list(BACKENDS),
        help="All-reduce paths to time and check against NCCL (default: all).",
    )
    parser.add_argument(
        "--mscclpp-algorithms",
        nargs="+",
        default=["tuned"],
        help="MSCCL++ all-reduce algorithms to force: 'tuned' (the tuning "
        "winner per size), 'all' (every algorithm that supports the size), or "
        "algorithm names such as default_allreduce_packet.",
    )
    parser.add_argument(
        "--expect-graph-backend",
        # Only MSCCL++ dispatch has been verified; other backends depend on the
        # host (e.g. quick-ar is ROCm-only) and are reported but not asserted.
        choices=["mscclpp"],
        help="Fail unless GroupCoordinator.all_reduce dispatches to this "
        "backend under CUDA graph capture.",
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
    set_mscclpp(True)
    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
    )
    publish_build_topology(
        world_rank=rank,
        tp_size=world_size,
        enable_symm_mem=args.enable_symm_mem,
    )
    with _record_mscclpp_candidates() as mscclpp_candidates:
        initialize_model_parallel(enable_symm_mem=args.enable_symm_mem)
    tp_group = get_parallel().tp_group
    group = tp_group.device_group
    pynccl_comm = tp_group.pynccl_comm
    pymscclpp_comm = tp_group.pymscclpp_comm
    # Without this check a missing or broken mscclpp install fails later with an
    # AttributeError or "no tuned configuration" error that hides the cause.
    if pymscclpp_comm is None or not pymscclpp_comm.available:
        raise RuntimeError(
            "MSCCL++ communicator unavailable: check that `mscclpp` imports "
            "and that tuning produced at least one all-reduce configuration"
        )
    allreduce_candidates = [
        candidate
        for candidate in mscclpp_candidates[pymscclpp_comm]
        if candidate.collective == "allreduce"
    ]
    unknown_algorithms = set(args.mscclpp_algorithms) - {"tuned", "all"}
    unknown_algorithms -= {candidate.name for candidate in allreduce_candidates}
    if unknown_algorithms:
        raise ValueError(
            f"Unknown MSCCL++ all-reduce algorithms {sorted(unknown_algorithms)}; "
            "available here: "
            f"{sorted({candidate.name for candidate in allreduce_candidates})}"
        )
    backend_benches = {
        "torch-eager": lambda inp: _bench_eager_time(
            lambda x: torch_allreduce(x, group), inp, tp_group
        ),
        "mscclpp-eager": lambda inp: _bench_eager_time(
            lambda x: msccl_allreduce(x, pymscclpp_comm), inp, tp_group
        ),
        "mscclpp-graph": lambda inp: _bench_graph_time(
            lambda x: msccl_allreduce(x, pymscclpp_comm), inp, tp_group
        ),
        # Each captured iteration re-copies the input, so the in-place result is
        # a single all-reduce.
        "pynccl-graph": lambda inp: _bench_graph_time(
            lambda x: pynccl_allreduce(x, pynccl_comm), inp, tp_group
        ),
    }
    dist.barrier()
    profile = False
    ctx = get_torch_prof_ctx(profile)
    result = []

    with ctx:
        for dtype_name in args.dtypes:
            dtype = getattr(torch, dtype_name)
            for nbytes in _message_sizes(args.min_bytes, args.max_bytes):
                sz = nbytes // dtype.itemsize
                # Values in {0, 1} keep every sum exact in fp16/bf16/fp32, so
                # every backend must agree with NCCL bit-for-bit in any order.
                inp_randn = torch.randint(0, 2, (sz,), dtype=dtype, device=device)
                expected = torch_allreduce(inp_randn.clone(), group)
                tuned_config, configs = _allreduce_configs_to_check(
                    pymscclpp_comm,
                    allreduce_candidates,
                    args.mscclpp_algorithms,
                    nbytes,
                    dtype,
                )
                if not configs:
                    continue

                def run_backend(backend):
                    output, cost_us = backend_benches[backend](inp_randn)
                    torch.testing.assert_close(output, expected, atol=0, rtol=0)
                    return cost_us

                baseline_times = {
                    backend: run_backend(backend)
                    for backend in args.backends
                    if backend not in MSCCLPP_BACKENDS
                }
                eager_dispatch = _check_sglang_eager_dispatch(
                    inp_randn, tp_group, expected
                )
                for config in configs:
                    with _use_allreduce_config(pymscclpp_comm, nbytes, config):
                        times = dict(baseline_times)
                        for backend in args.backends:
                            if backend in MSCCLPP_BACKENDS:
                                times[backend] = run_backend(backend)
                        _check_graph_replay_uses_new_input(
                            lambda inp: msccl_allreduce(inp, pymscclpp_comm),
                            inp_randn,
                            tp_group,
                            group,
                        )
                        graph_dispatch = _check_sglang_graph_dispatch(
                            inp_randn, tp_group, expected
                        )
                    label = _config_label(config, tuned_config)
                    if (
                        args.expect_graph_backend is not None
                        and graph_dispatch != args.expect_graph_backend
                    ):
                        raise RuntimeError(
                            f"GroupCoordinator.all_reduce dispatched {nbytes} "
                            f"bytes of {dtype_name} to {graph_dispatch} under "
                            f"graph capture with MSCCL++ {label}; expected "
                            f"{args.expect_graph_backend}"
                        )
                    row = {
                        "msg_size": human_readable_size(nbytes),
                        "dtype": dtype_name,
                        "mscclpp algorithm": label,
                    }
                    row.update(
                        (f"{backend} (us)", times[backend]) for backend in args.backends
                    )
                    row["sglang eager dispatch"] = eager_dispatch
                    row["sglang graph dispatch"] = graph_dispatch
                    result.append(row)
                    if rank == 0:
                        print(
                            f"sz={sz}, dtype={dtype}, mscclpp={label}: "
                            "correctness check PASS!"
                        )
    if not result:
        raise RuntimeError(
            "No MSCCL++ all-reduce algorithm supports the requested message "
            "sizes and dtypes"
        )
    if rank == 0:
        print_markdown_table(result)
    if profile:
        prof_dir = f"prof/msccl"
        os.makedirs(prof_dir, exist_ok=True)
        ctx.export_chrome_trace(f"{prof_dir}/trace_rank{dist.get_rank()}.json.gz")

    # The recorded candidates hold MSCCL++ algorithms that must be freed before
    # the communicator, as they are after tuning, or the process aborts on exit.
    del mscclpp_candidates, allreduce_candidates, tuned_config, configs, config
    pymscclpp_comm.destroy()
    dist.barrier()
    cleanup_dist_env_and_memory()
