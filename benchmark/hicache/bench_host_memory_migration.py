"""Reproduce ROCm HiCache USERPTR eviction stalls on an otherwise idle NUMA host.

Run each allocator in a separate process on the same GPU and node:
  HSA_USERPTR_FOR_PAGED_MEM=0 python benchmark/hicache/bench_host_memory_migration.py --allocator registered --pool-gb 180
  HSA_USERPTR_FOR_PAGED_MEM=0 python benchmark/hicache/bench_host_memory_migration.py --allocator gtt --pool-gb 180

The registered arm uses the pre-fix mmap + host-register path; the gtt arm
uses alloc_with_host_register from this checkout. Both keep the HIP environment
identical. This is a manual hardware test, not a CI test or an LLM benchmark.
It needs two allowed memory NUMA nodes, sufficient host/GTT memory, and permission
to call move_pages. A container's seccomp policy must permit NUMA migration;
pass its worker's host PID with --kfd-pid if PID namespaces differ.
Only this process's own allocation is migrated; no system policy is changed.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import json
import os
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from sglang.srt.mem_cache.pool_host.common import (
    HostTensorAllocator,
    _cuda_host_register,
    _cuda_host_unregister,
    alloc_with_host_register,
)

COPY_BYTES = 16 * 1024**2
MIGRATE_BYTES = 2 * 1024**2


def memory_nodes():
    allowed = next(
        line.split()[1]
        for line in Path("/proc/self/status").read_text().splitlines()
        if line.startswith("Mems_allowed_list:")
    )
    nodes = []
    for entry in allowed.split(","):
        bounds = [int(value) for value in entry.split("-")]
        for node in range(bounds[0], bounds[-1] + 1):
            info = Path(f"/sys/devices/system/node/node{node}/meminfo").read_text()
            if any(
                "MemTotal:" in line and int(line.split()[-2]) > 0
                for line in info.splitlines()
            ):
                nodes.append(node)
    if len(nodes) < 2:
        raise RuntimeError("Two allowed NUMA nodes with memory are required")
    return nodes


def evicted_ms(pid):
    counters = {}
    for path in Path(f"/sys/class/kfd/kfd/proc/{pid}").glob("stats_*"):
        counters[path.name] = int((path / "evicted_ms").read_text())
    if not counters:
        raise RuntimeError(
            "KFD evicted_ms counters unavailable; --kfd-pid must identify this worker on the host"
        )
    return counters


class PageMover:
    def __init__(self, ptr):
        page_size = os.sysconf("SC_PAGE_SIZE")
        if ptr % page_size or MIGRATE_BYTES % page_size:
            raise RuntimeError("Migration range must be page aligned")
        self.count = MIGRATE_BYTES // page_size
        self.pages = (ctypes.c_void_p * self.count)(
            *(ptr + offset * page_size for offset in range(self.count))
        )
        self.lib = ctypes.CDLL("libnuma.so.1", use_errno=True)
        self.lib.move_pages.argtypes = [
            ctypes.c_int,
            ctypes.c_ulong,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
        ]
        self.lib.move_pages.restype = ctypes.c_long

    def call(self, target=None):
        status = (ctypes.c_int * self.count)(*([-errno.EIO] * self.count))
        nodes = (
            None
            if target is None
            else (ctypes.c_int * self.count)(*([target] * self.count))
        )
        ctypes.set_errno(0)
        # pid=0: own process. MPOL_MF_MOVE=2: move only exclusively owned pages.
        rc = self.lib.move_pages(
            0, self.count, self.pages, nodes, status, 0 if target is None else 2
        )
        return {
            "rc": rc,
            "errno": ctypes.get_errno() if rc < 0 else 0,
            "status": list(status),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allocator", choices=("registered", "gtt"), required=True)
    parser.add_argument("--pool-gb", type=float, default=180)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument(
        "--kfd-pid",
        type=int,
        default=os.getpid(),
        help="This worker's host PID (for containerized runs)",
    )
    args = parser.parse_args()
    size = int(args.pool_gb * 1e9)
    if size < COPY_BYTES or args.trials < 1 or args.iterations < 1:
        parser.error("Pool must fit 16 MiB; trials and iterations must be positive")
    if os.environ.get("HSA_USERPTR_FOR_PAGED_MEM") != "0":
        parser.error("Set HSA_USERPTR_FOR_PAGED_MEM=0 before starting either arm")
    if not torch.version.hip or not torch.cuda.is_available():
        parser.error("A ROCm GPU is required")
    nodes = memory_nodes()
    torch.cuda.set_device(0)
    evicted_ms(args.kfd_pid)
    torch.set_num_threads(1)
    allocator = HostTensorAllocator()
    if args.allocator == "registered":
        host = allocator.allocate((size,), torch.uint8, "cpu")
        _cuda_host_register(host)
    else:
        host = alloc_with_host_register((size,), torch.uint8, "cpu", True, allocator)
    host[:COPY_BYTES].fill_(42)
    mover = PageMover(host.data_ptr())
    source = host[:COPY_BYTES]
    dest = torch.empty(COPY_BYTES, dtype=torch.uint8, device="cuda")
    a = torch.randn((4096, 4096), dtype=torch.bfloat16, device="cuda")
    b = torch.randn_like(a)
    c = torch.empty_like(a)

    def iteration(trigger=None):
        start = time.perf_counter()
        for _ in range(4):
            torch.mm(a, b, out=c)
        if trigger is not None:
            trigger.set()
        dest.copy_(source, non_blocking=True)
        torch.cuda.synchronize()
        return (time.perf_counter() - start) * 1000

    try:
        for _ in range(50):
            iteration()
        control = [iteration() for _ in range(args.iterations)]
        median = statistics.median(control)
        print(
            json.dumps(
                {
                    "allocator": args.allocator,
                    "pool_bytes": size,
                    "gpu": torch.cuda.get_device_name(0),
                    "torch": torch.__version__,
                    "kfd_pid": args.kfd_pid,
                    "numa_nodes": nodes,
                    "control_median_ms": median,
                    "control_max_ms": max(control),
                }
            ),
            flush=True,
        )
        results = []
        with ThreadPoolExecutor(max_workers=1) as executor:
            for trial in range(args.trials):
                current = mover.call()
                if args.allocator == "registered":
                    if current["rc"] != 0 or any(
                        node < 0 for node in current["status"]
                    ):
                        raise RuntimeError(f"Cannot query USERPTR pages: {current}")
                    target = next(
                        node for node in nodes if node != current["status"][0]
                    )
                else:
                    target = nodes[trial % len(nodes)]
                before = evicted_ms(args.kfd_pid)
                trigger = threading.Event()

                def migrate():
                    if not trigger.wait(30):
                        raise RuntimeError("GPU workload did not start")
                    return mover.call(target)

                future = executor.submit(migrate)
                samples = [iteration(trigger)]
                after_migration = 0
                while after_migration < args.iterations:
                    samples.append(iteration())
                    if future.done():
                        after_migration += 1
                migration = future.result()
                after = evicted_ms(args.kfd_pid)
                delta = {key: after[key] - before[key] for key in before}
                moved = sum(
                    old >= 0 and old != target and new == target
                    for old, new in zip(current["status"], migration["status"])
                )
                rejected = (
                    migration["rc"] == -1 and migration["errno"] == errno.EFAULT
                ) or (
                    migration["rc"] >= 0
                    and all(value == -errno.EFAULT for value in migration["status"])
                )
                result = {
                    "trial": trial + 1,
                    "target_node": target,
                    "move_rc": migration["rc"],
                    "move_errno": migration["errno"],
                    "page_status_counts": {
                        str(value): migration["status"].count(value)
                        for value in set(migration["status"])
                    },
                    "moved_pages": moved,
                    "gtt_migration_rejected": rejected,
                    "max_iteration_ms": max(samples),
                    "excess_over_control_ms": sum(
                        max(0, value - median) for value in samples
                    ),
                    "evicted_ms_delta": delta,
                }
                print(json.dumps(result), flush=True)
                if args.allocator == "registered" and moved == 0:
                    raise RuntimeError(
                        "No pages migrated; USERPTR reproduction is inconclusive"
                    )
                if args.allocator == "gtt" and (
                    not rejected or max(delta.values()) != 0
                ):
                    raise RuntimeError(
                        "GTT migration rejection / zero-eviction check failed"
                    )
                if not torch.all(dest == 42).item():
                    raise RuntimeError("Host-to-device copy returned incorrect data")
                results.append(result)
        reproduced = any(
            max(r["evicted_ms_delta"].values()) > 0 and r["max_iteration_ms"] > 100
            for r in results
        )
        print(
            json.dumps(
                {
                    "allocator": args.allocator,
                    "trials": len(results),
                    "userptr_stall_reproduced": reproduced,
                }
            ),
            flush=True,
        )
        if args.allocator == "registered" and not reproduced:
            raise RuntimeError(
                "Pages migrated, but no >100 ms GPU eviction stall was reproduced"
            )
    finally:
        torch.cuda.synchronize()
        _cuda_host_unregister(host)


if __name__ == "__main__":
    main()
