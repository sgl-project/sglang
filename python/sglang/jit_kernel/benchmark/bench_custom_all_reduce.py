"""
Benchmark latency comparison for custom all-reduce implementations.

Compares:
- JIT custom allreduce (sglang.jit_kernel.custom_all_reduce)
- NCCL allreduce (torch.distributed.all_reduce)
- AOT custom allreduce (sgl_kernel.allreduce), if available

Requires multiple GPUs.

Usage:
    # Default: benchmark with 2 GPUs
    python bench_custom_all_reduce.py

    # Specify world size
    python bench_custom_all_reduce.py --world-size 4

    # Specify dtypes and sizes
    python bench_custom_all_reduce.py --world-size 2 --dtype float16
"""

import argparse
import ctypes
import multiprocessing as mp
import socket
import statistics
from typing import List, Optional

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from sglang.srt.distributed.device_communicators.cuda_wrapper import CudaRTLibrary


def get_open_port() -> int:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]
    except OSError:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
            s.bind(("::1", 0))
            return s.getsockname()[1]


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
            opened_ptr = lib.cudaIpcOpenMemHandle(h)
            pointers.append(opened_ptr.value)

    dist.barrier(group=group)
    return pointers


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


def _bench_worker(
    world_size: int,
    rank: int,
    distributed_init_port: int,
    test_sizes: List[int],
    dtypes: List[torch.dtype],
    warmup: int,
    repeat: int,
    results_queue,
):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    distributed_init_method = f"tcp://localhost:{distributed_init_port}"
    dist.init_process_group(
        backend="nccl",
        init_method=distributed_init_method,
        rank=rank,
        world_size=world_size,
        device_id=device,
    )
    group = dist.group.WORLD

    # Try to import JIT custom allreduce
    try:
        from sglang.jit_kernel.custom_all_reduce import all_reduce as jit_all_reduce
        from sglang.jit_kernel.custom_all_reduce import (
            dispose,
            init_custom_ar,
            meta_size,
            register_buffer,
        )

        jit_available = True
    except Exception as e:
        if rank == 0:
            print(f"  JIT custom_all_reduce not available: {e}")
        jit_available = False

    # Try to import AOT custom allreduce
    try:
        import sgl_kernel.allreduce as _aot_ar

        aot_available = True
    except ImportError:
        aot_available = False

    custom_ptr = None
    buffer_ptrs = None
    meta_ptrs = None

    results = {}

    try:
        max_size = 8192 * 1024  # 8 MiB buffer

        if jit_available:
            meta_ptrs = create_shared_buffer(meta_size() + max_size, group=group)
            rank_data = torch.empty(8 * 1024 * 1024, dtype=torch.uint8, device=device)
            buffer_ptrs = create_shared_buffer(max_size, group=group)
            custom_ptr = init_custom_ar(meta_ptrs, rank_data, rank, True)
            register_buffer(custom_ptr, buffer_ptrs)

        for dtype in dtypes:
            dtype_results = {}
            for sz in test_sizes:
                inp = torch.ones(sz, dtype=dtype, device=device)
                out_jit = torch.empty_like(inp)
                out_nccl = torch.empty_like(inp)

                sz_bytes = inp.nbytes

                # ---- NCCL benchmark ----
                # Warmup
                for _ in range(warmup):
                    inp_ref = inp.clone()
                    dist.all_reduce(inp_ref, group=group)
                dist.barrier()
                torch.cuda.synchronize()

                # Timed runs using CUDA events
                nccl_times = []
                for _ in range(repeat):
                    inp_ref = inp.clone()
                    start_ev = torch.cuda.Event(enable_timing=True)
                    end_ev = torch.cuda.Event(enable_timing=True)
                    start_ev.record()
                    dist.all_reduce(inp_ref, group=group)
                    end_ev.record()
                    torch.cuda.synchronize()
                    nccl_times.append(start_ev.elapsed_time(end_ev) * 1000)  # us
                out_nccl.copy_(inp_ref)

                # ---- JIT custom allreduce benchmark ----
                jit_times = []
                if jit_available and sz_bytes <= max_size:
                    # Warmup
                    for _ in range(warmup):
                        jit_all_reduce(
                            custom_ptr, inp, out_jit, buffer_ptrs[rank], max_size
                        )
                    dist.barrier()
                    torch.cuda.synchronize()

                    for _ in range(repeat):
                        start_ev = torch.cuda.Event(enable_timing=True)
                        end_ev = torch.cuda.Event(enable_timing=True)
                        start_ev.record()
                        jit_all_reduce(
                            custom_ptr, inp, out_jit, buffer_ptrs[rank], max_size
                        )
                        end_ev.record()
                        torch.cuda.synchronize()
                        jit_times.append(start_ev.elapsed_time(end_ev) * 1000)  # us

                # ---- AOT allreduce benchmark (rank 0 only checks availability) ----
                aot_times = []
                if aot_available and sz_bytes <= max_size:
                    # Reuse same buffers if possible (AOT has same interface)
                    try:
                        # Warmup
                        for _ in range(warmup):
                            _aot_ar.all_reduce(
                                custom_ptr, inp, out_jit, buffer_ptrs[rank], max_size
                            )
                        dist.barrier()
                        torch.cuda.synchronize()

                        for _ in range(repeat):
                            start_ev = torch.cuda.Event(enable_timing=True)
                            end_ev = torch.cuda.Event(enable_timing=True)
                            start_ev.record()
                            _aot_ar.all_reduce(
                                custom_ptr, inp, out_jit, buffer_ptrs[rank], max_size
                            )
                            end_ev.record()
                            torch.cuda.synchronize()
                            aot_times.append(start_ev.elapsed_time(end_ev) * 1000)
                    except Exception:
                        aot_times = []

                dist.barrier()

                if rank == 0:
                    dtype_results[sz] = {
                        "nccl": nccl_times,
                        "jit": jit_times,
                        "aot": aot_times,
                        "skipped": sz_bytes > max_size,
                    }

            if rank == 0:
                results[dtype] = dtype_results

    finally:
        dist.barrier(group=group)
        if custom_ptr is not None:
            dispose(custom_ptr)
        if buffer_ptrs:
            free_shared_buffer(buffer_ptrs, group)
        if meta_ptrs:
            free_shared_buffer(meta_ptrs, group)
        dist.destroy_process_group(group=group)

    if rank == 0:
        results_queue.put(
            {
                "results": results,
                "jit_available": jit_available,
                "aot_available": aot_available,
            }
        )


def _dtype_str(dtype: torch.dtype) -> str:
    return {
        torch.float32: "float32",
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
    }.get(dtype, str(dtype))


def _print_results(
    results: dict,
    test_sizes: List[int],
    dtypes: List[torch.dtype],
    world_size: int,
    jit_available: bool,
    aot_available: bool,
):
    col_nccl = "NCCL (us)"
    col_jit = "JIT (us)"
    col_aot = "AOT (us)"
    col_speedup_nccl = "Speedup (JIT/NCCL)"
    col_speedup_aot = "Speedup (AOT/JIT)"

    for dtype in dtypes:
        print(f"\n--- dtype={_dtype_str(dtype)}, world_size={world_size} ---")
        header = f"{'Elements':>12}  {'Bytes':>10}  {col_nccl:>12}"
        if jit_available:
            header += f"  {col_jit:>12}  {col_speedup_nccl:>20}"
        if aot_available:
            header += f"  {col_aot:>12}"
        if jit_available and aot_available:
            header += f"  {col_speedup_aot:>18}"
        print(header)
        print("-" * len(header))

        dtype_results = results.get(dtype, {})
        for sz in test_sizes:
            r = dtype_results.get(sz)
            if r is None:
                continue

            elem_bytes = torch.empty(1, dtype=dtype).element_size()
            nbytes = sz * elem_bytes

            nccl_med = statistics.median(r["nccl"]) if r["nccl"] else float("nan")
            line = f"{sz:>12,}  {nbytes:>10,}  {nccl_med:>12.2f}"

            jit_med = None
            if jit_available:
                if r["skipped"]:
                    line += f"  {'SKIP':>12}  {'N/A':>20}"
                elif r["jit"]:
                    jit_med = statistics.median(r["jit"])
                    speedup = nccl_med / jit_med if jit_med > 0 else float("nan")
                    line += f"  {jit_med:>12.2f}  {speedup:>20.2f}x"
                else:
                    line += f"  {'N/A':>12}  {'N/A':>20}"

            aot_med = None
            if aot_available:
                if r["aot"]:
                    aot_med = statistics.median(r["aot"])
                    line += f"  {aot_med:>12.2f}"
                else:
                    line += f"  {'N/A':>12}"

            if jit_available and aot_available:
                if jit_med is not None and aot_med is not None and jit_med > 0:
                    speedup_aot = aot_med / jit_med
                    line += f"  {speedup_aot:>18.2f}x"
                else:
                    line += f"  {'N/A':>18}"

            print(line)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark custom allreduce JIT kernel"
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=None,
        help="Number of GPUs to use (default: all available, up to 8)",
    )
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32", "all"],
        default="all",
        help="Dtype to benchmark (default: all)",
    )
    parser.add_argument(
        "--warmup", type=int, default=20, help="Warmup iterations (default: 20)"
    )
    parser.add_argument(
        "--repeat", type=int, default=100, help="Measurement iterations (default: 100)"
    )
    args = parser.parse_args()

    dtype_map = {
        "float16": [torch.float16],
        "bfloat16": [torch.bfloat16],
        "float32": [torch.float32],
        "all": [torch.float16, torch.bfloat16, torch.float32],
    }
    dtypes = dtype_map[args.dtype]

    # Element counts to benchmark
    test_sizes = [512, 2048, 8192, 32768, 131072, 524288, 1048576, 2097152]

    available_gpus = torch.cuda.device_count()
    if args.world_size is not None:
        world_size = args.world_size
    else:
        world_size = min(available_gpus, 2)

    print("=" * 70)
    print("Custom All-Reduce JIT Kernel Benchmark")
    print("=" * 70)
    print(f"GPUs available:  {available_gpus}")
    print(f"World size:      {world_size}")
    print(f"Dtypes:          {[_dtype_str(d) for d in dtypes]}")
    print(f"Element counts:  {test_sizes}")
    print(f"Warmup:          {args.warmup}")
    print(f"Repeat:          {args.repeat}")
    print("=" * 70)

    if world_size < 2:
        print("ERROR: Need at least 2 GPUs for this benchmark.")
        return
    if world_size > available_gpus:
        print(
            f"ERROR: Requested {world_size} GPUs but only {available_gpus} available."
        )
        return

    mp.set_start_method("spawn", force=True)
    port = get_open_port()
    results_queue = mp.Queue()

    procs = []
    for rank in range(world_size):
        p = mp.Process(
            target=_bench_worker,
            args=(
                world_size,
                rank,
                port,
                test_sizes,
                dtypes,
                args.warmup,
                args.repeat,
                results_queue,
            ),
            name=f"Worker-{rank}",
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()
        if p.exitcode != 0:
            print(f"Process {p.name} exited with code {p.exitcode}")

    if not results_queue.empty():
        data = results_queue.get()
        _print_results(
            data["results"],
            test_sizes,
            dtypes,
            world_size,
            data["jit_available"],
            data["aot_available"],
        )
    else:
        print("No results collected (benchmark may have failed).")


if __name__ == "__main__":
    main()
