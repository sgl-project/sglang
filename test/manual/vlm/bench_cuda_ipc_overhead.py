"""Benchmark cudaIpcGetMemHandle / cudaIpcOpenMemHandle overhead.

Usage:
    python bench_cuda_ipc_overhead.py

Measures:
  1. _share_cuda_()       — wraps cudaIpcGetMemHandle (producer side)
  2. _new_shared_cuda()   — wraps cudaIpcOpenMemHandle (consumer side, separate process)
  3. GPU-to-GPU copy      — baseline comparison (same device)

Tests multiple tensor sizes typical for VLM features.
"""

import multiprocessing as mp
import pickle
import time

import torch

WARMUP = 10
REPEAT = 100

# Fine-grained sizes: 64KB to 512MB
SIZES = [
    ("64 KB", 64 * 1024),
    ("256 KB", 256 * 1024),
    ("512 KB", 512 * 1024),
    ("1 MB", 1 * 1024 * 1024),
    ("2 MB", 2 * 1024 * 1024),
    ("4 MB", 4 * 1024 * 1024),
    ("8 MB", 8 * 1024 * 1024),
    ("16 MB", 16 * 1024 * 1024),
    ("32 MB", 32 * 1024 * 1024),
    ("64 MB", 64 * 1024 * 1024),
    ("128 MB", 128 * 1024 * 1024),
    ("256 MB", 256 * 1024 * 1024),
    ("512 MB", 512 * 1024 * 1024),
]


def bench_share_cuda(tensor, warmup=WARMUP, repeat=REPEAT):
    """Measure _share_cuda_() = cudaIpcGetMemHandle."""
    storage = tensor.untyped_storage()
    for _ in range(warmup):
        h = storage._share_cuda_()
    torch.cuda.synchronize()

    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        h = storage._share_cuda_()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return h, times


def _consumer_worker(handles_bytes, device_idx, warmup, repeat, result_pipe):
    """Child process: open IPC handles and measure latency."""
    handles = pickle.loads(handles_bytes)
    target = torch.device(f"cuda:{device_idx}")

    results = {}
    for label, handle in handles:
        redirected = (device_idx,) + tuple(handle)[1:]

        # warmup
        for _ in range(warmup):
            with torch.cuda.device(target):
                s = torch.UntypedStorage._new_shared_cuda(*redirected)
            del s

        torch.cuda.synchronize()
        times = []
        for _ in range(repeat):
            t0 = time.perf_counter()
            with torch.cuda.device(target):
                s = torch.UntypedStorage._new_shared_cuda(*redirected)
            t1 = time.perf_counter()
            times.append(t1 - t0)
            del s

        results[label] = times

    result_pipe.send(results)
    result_pipe.close()


def bench_open_handle_batch(
    handles_with_labels, device_idx, warmup=WARMUP, repeat=REPEAT
):
    """Measure _new_shared_cuda() for all sizes in one child process."""
    parent_conn, child_conn = mp.Pipe()
    handles_bytes = pickle.dumps(handles_with_labels)

    p = mp.Process(
        target=_consumer_worker,
        args=(handles_bytes, device_idx, warmup, repeat, child_conn),
    )
    p.start()
    results = parent_conn.recv()
    p.join()
    return results


def bench_gpu_copy(tensor, warmup=WARMUP, repeat=REPEAT):
    """Measure GPU-to-GPU copy (same device) as baseline."""
    dst = torch.empty_like(tensor)
    for _ in range(warmup):
        dst.copy_(tensor)
    torch.cuda.synchronize()

    times = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        dst.copy_(tensor)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return times


def median_us(times):
    s = sorted(times)
    mid = len(s) // 2
    return s[mid] * 1e6


def p95_us(times):
    s = sorted(times)
    idx = int(len(s) * 0.95)
    return s[idx] * 1e6


def main():
    mp.set_start_method("spawn", force=True)
    device = torch.device("cuda:0")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Warmup: {WARMUP}, Repeat: {REPEAT}")
    print()

    # Phase 1: producer-side GetHandle + GPU copy for all sizes
    tensors = {}
    share_results = {}
    copy_results = {}
    handles_with_labels = []

    for label, nbytes in SIZES:
        tensor = torch.empty(nbytes, dtype=torch.int8, device=device)
        tensors[label] = tensor
        handle, share_times = bench_share_cuda(tensor)
        share_results[label] = share_times
        copy_results[label] = bench_gpu_copy(tensor)
        handles_with_labels.append((label, handle))

    # Phase 2: consumer-side OpenHandle (all sizes in one child process)
    open_results = bench_open_handle_batch(handles_with_labels, device.index)

    # Print results
    print(
        f"{'Size':>8}  {'GetHandle':>10} {'(p95)':>7}  {'OpenHandle':>10} {'(p95)':>7}  "
        f"{'GPU copy':>9} {'(p95)':>7}  {'Open/copy':>10}"
    )
    print("-" * 90)

    for label, nbytes in SIZES:
        share_us = median_us(share_results[label])
        share_p95 = p95_us(share_results[label])
        open_us = median_us(open_results[label])
        open_p95 = p95_us(open_results[label])
        copy_us = median_us(copy_results[label])
        copy_p95 = p95_us(copy_results[label])

        ratio = f"{open_us / copy_us:.1f}x" if copy_us > 0 else "N/A"

        print(
            f"{label:>8}  {share_us:>8.1f}us {share_p95:>6.1f}  {open_us:>8.1f}us {open_p95:>6.1f}  "
            f"{copy_us:>7.1f}us {copy_p95:>6.1f}  {ratio:>10}"
        )

    # Phase 3: batch open N handles to measure total time
    print()
    print("=== Batch open N handles (total time) ===")
    print(f"{'N tensors':>10}  {'Total open (ms)':>16}  {'Per tensor (us)':>16}")
    print("-" * 48)

    for n in [1, 10, 50, 100, 300]:
        # Create N tensors of 2MB each
        batch_tensors = []
        batch_handles = []
        for i in range(n):
            t = torch.empty(2 * 1024 * 1024, dtype=torch.int8, device=device)
            batch_tensors.append(t)
            storage = t.untyped_storage()
            h = storage._share_cuda_()
            batch_handles.append((f"t{i}", h))

        open_res = bench_open_handle_batch(
            batch_handles, device.index, warmup=3, repeat=20
        )

        # Sum median per-tensor times
        total_us = sum(median_us(open_res[f"t{i}"]) for i in range(n))
        print(f"{n:>10}  {total_us/1000:>14.2f}ms  {total_us/n:>14.1f}us")

        del batch_tensors
        torch.cuda.empty_cache()

    del tensors
    torch.cuda.empty_cache()

    print()
    print("GetHandle  = cudaIpcGetMemHandle  (producer)")
    print("OpenHandle = cudaIpcOpenMemHandle  (consumer)")
    print("GPU copy   = same-device memcpy")


if __name__ == "__main__":
    main()
