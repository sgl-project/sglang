# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

"""Buffered vs O_DIRECT page I/O of the ``fast_file`` HiCache backend.

Drives ``HiCacheFastFile`` the way the cache controller does (``batch_set_v1``
and ``batch_get_v1`` in 128-page batches) against a page-first host pool
stand-in on page-aligned memory, with the page geometry of a real model. Every
(mode, read_workers, repeat) cell runs in its own process. Before the read
phase the page cache is written back and dropped, so buffered reads come from
the disk rather than from the pages the writes left in memory.

    python benchmark/hicache/bench_fast_file_io.py --storage-dir /mnt/nvme/bench \\
        --preset qwen3-8b --pages 400 --read-workers 1 8 --repeat 3
"""

from __future__ import annotations

import argparse
import json
import mmap
import os
import statistics
import subprocess
import sys
import time
import zlib

import torch

from sglang.srt.mem_cache.hicache_storage import (
    STORAGE_BATCH_SIZE,
    HiCacheStorageConfig,
)
from sglang.srt.mem_cache.storage.fast_file.fast_file_store import HiCacheFastFile

# (layers, kv heads per rank, head_dim); dtype and page size come from the CLI.
PRESETS = {
    "qwen3-0.6b": (28, 8, 128),
    "qwen3-8b": (36, 8, 128),
    "llama3-8b": (32, 8, 128),
    "llama3-70b-tp8": (80, 1, 128),
}


class _PageFirstPool:
    """Page-first host pool stand-in: kv_buffer[k_or_v, token, bytes_per_token]."""

    def __init__(self, *, pages: int, page_size: int, bytes_per_token: int):
        self.layout = "page_first"
        self.page_size = page_size
        self.size = pages * page_size
        self.dtype = torch.uint8
        self.size_per_token = bytes_per_token
        self._mapping = mmap.mmap(-1, 2 * self.size * bytes_per_token)
        self.kv_buffer = torch.frombuffer(self._mapping, dtype=torch.uint8).reshape(
            2, self.size, bytes_per_token
        )

    def get_page_buffer_meta(self, indices):
        ptrs, sizes = [], []
        base = self.kv_buffer.data_ptr()
        value_offset = self.size * self.size_per_token
        page_bytes = self.page_size * self.size_per_token
        for index in indices[:: self.page_size].tolist():
            key_ptr = base + index * self.size_per_token
            ptrs.extend((key_ptr, key_ptr + value_offset))
            sizes.extend((page_bytes, page_bytes))
        return ptrs, sizes

    def page_checksum(self, page: int) -> int:
        lo, hi = page * self.page_size, (page + 1) * self.page_size
        k = self.kv_buffer[0, lo:hi].numpy().tobytes()
        v = self.kv_buffer[1, lo:hi].numpy().tobytes()
        return zlib.crc32(v, zlib.crc32(k))


def _cached_kib() -> int:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("Cached:"):
                return int(line.split()[1])
    return 0


def _drop_page_cache(directory: str) -> None:
    # DONTNEED only drops clean pages; buffered writes leave dirty ones behind,
    # so write them back first or the "cold" read would hit the page cache.
    os.sync()
    for entry in os.scandir(directory):
        if not entry.name.endswith(".bin"):
            continue
        fd = os.open(entry.path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


def _timed_batches(fn, keys, indices, page_size):
    """Run fn over 128-page batches; return (wall_s, cpu_s, cached_delta_mib, per-batch s)."""
    cached0, wall0, cpu0 = _cached_kib(), time.perf_counter(), time.process_time()
    lat = []
    for start in range(0, len(keys), STORAGE_BATCH_SIZE):
        batch_keys = keys[start : start + STORAGE_BATCH_SIZE]
        batch_idx = indices[start * page_size : (start + len(batch_keys)) * page_size]
        t0 = time.perf_counter()
        ok = fn(batch_keys, batch_idx)
        lat.append(time.perf_counter() - t0)
        if not all(ok):
            raise RuntimeError(f"batch starting at {start} failed: {sum(ok)}/{len(ok)}")
    return (
        time.perf_counter() - wall0,
        time.process_time() - cpu0,
        (_cached_kib() - cached0) / 1024,
        lat,
    )


def run_cell(args) -> dict:
    layers, heads, head_dim = PRESETS[args.preset]
    bytes_per_token = layers * heads * head_dim * args.dtype_bytes
    page_bytes = 2 * args.page_size * bytes_per_token
    pool = _PageFirstPool(
        pages=args.pages, page_size=args.page_size, bytes_per_token=bytes_per_token
    )
    # Incompressible, page-distinct contents so a read hitting the wrong file shows.
    rng = torch.Generator().manual_seed(args.seed)
    pool.kv_buffer.view(-1).copy_(
        torch.randint(
            0, 256, (pool.kv_buffer.numel(),), generator=rng, dtype=torch.uint8
        )
    )
    checksums = {
        p: pool.page_checksum(p) for p in range(0, args.pages, max(1, args.pages // 8))
    }

    config = HiCacheStorageConfig(
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=False,
        enable_storage_metrics=False,
        is_page_first_layout=True,
        model_name=f"bench-{args.preset}",
        extra_config={
            "file_storage_path": args.storage_dir,
            "o_direct": args.mode == "o_direct",
            "read_workers": args.read_workers,
        },
    )
    backend = HiCacheFastFile(config, mem_pool_host=pool)
    backend.register_mem_pool_host(pool)
    backend.clear()
    keys = [f"page{i:07d}" for i in range(args.pages)]
    indices = torch.arange(args.pages * args.page_size, dtype=torch.int64)
    result = {
        "mode": args.mode,
        "read_workers": args.read_workers,
        "o_direct_active": bool(backend._pool_o_direct["kv"]),
        "alignment": backend._o_direct_alignment,
        "page_bytes": page_bytes,
        "pages": args.pages,
    }
    try:
        wall, cpu, cached, lat = _timed_batches(
            backend.batch_set_v1, keys, indices, args.page_size
        )
        result["write"] = {
            "gbps": args.pages * page_bytes / 1e9 / wall,
            "cpu_s": cpu,
            "cached_mib": cached,
        }
        _drop_page_cache(backend.file_path)
        pool.kv_buffer.zero_()
        wall, cpu, cached, lat = _timed_batches(
            backend.batch_get_v1, keys, indices, args.page_size
        )
        per_page_ms = sorted(l / STORAGE_BATCH_SIZE * 1e3 for l in lat[:-1] or lat)
        result["read"] = {
            "gbps": args.pages * page_bytes / 1e9 / wall,
            "cpu_s": cpu,
            "cached_mib": cached,
            "per_page_ms_p50": per_page_ms[len(per_page_ms) // 2],
            "per_page_ms_max": per_page_ms[-1],
        }
        result["verified"] = all(
            pool.page_checksum(p) == c for p, c in checksums.items()
        )
    finally:
        backend.clear()
        backend.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--storage-dir", required=True, help="directory on the disk under test"
    )
    parser.add_argument("--preset", choices=sorted(PRESETS), default="qwen3-8b")
    parser.add_argument("--page-size", type=int, default=64, help="tokens per page")
    parser.add_argument(
        "--dtype-bytes", type=int, default=2, help="2 for bf16, 1 for fp8"
    )
    parser.add_argument("--pages", type=int, default=400)
    parser.add_argument("--read-workers", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--modes", nargs="+", default=["buffered", "o_direct"])
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", help="write every cell's raw result here")
    parser.add_argument("--cell", help=argparse.SUPPRESS)
    parser.add_argument("--mode", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.cell:
        # Child process: one (mode, read_workers) measurement.
        args.read_workers = int(args.cell)
        print(json.dumps(run_cell(args)))
        return

    os.makedirs(args.storage_dir, exist_ok=True)
    rows = []
    for mode in args.modes:
        for workers in args.read_workers:
            for _ in range(args.repeat):
                cmd = [
                    sys.executable,
                    __file__,
                    "--storage-dir",
                    args.storage_dir,
                    "--preset",
                    args.preset,
                    "--page-size",
                    str(args.page_size),
                    "--dtype-bytes",
                    str(args.dtype_bytes),
                    "--pages",
                    str(args.pages),
                    "--seed",
                    str(args.seed),
                    "--mode",
                    mode,
                    "--cell",
                    str(workers),
                ]
                out = subprocess.run(cmd, check=True, capture_output=True, text=True)
                rows.append(json.loads(out.stdout.strip().splitlines()[-1]))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=1)

    page_mib = rows[0]["page_bytes"] / 2**20
    alignment = max(r["alignment"] for r in rows)
    print(
        f"preset={args.preset} page={args.page_size} tokens = {page_mib:.2f} MiB, "
        f"{args.pages} pages, median of {args.repeat} runs; probed alignment={alignment}"
    )
    print(
        "write GB/s is what the kernel accepted (buffered) or what landed on the "
        "device (o_direct); read columns are measured with the page cache written "
        "back and dropped first"
    )
    print(
        f"{'mode':<9}{'rd':>3} {'active':>6} | {'write GB/s':>10} {'cpu s':>6} {'cache MiB':>9} | "
        f"{'read GB/s':>9} {'cpu s':>6} {'cache MiB':>9} {'p50 ms':>7} {'max ms':>7} | ok"
    )
    for mode in args.modes:
        for workers in args.read_workers:
            cells = [
                r for r in rows if r["mode"] == mode and r["read_workers"] == workers
            ]
            med = lambda path: statistics.median(c[path[0]][path[1]] for c in cells)
            print(
                f"{mode:<9}{workers:>3} {str(all(c['o_direct_active'] for c in cells)):>6} | "
                f"{med(('write', 'gbps')):>10.2f} {med(('write', 'cpu_s')):>6.2f} {med(('write', 'cached_mib')):>9.0f} | "
                f"{med(('read', 'gbps')):>9.2f} {med(('read', 'cpu_s')):>6.2f} {med(('read', 'cached_mib')):>9.0f} "
                f"{med(('read', 'per_page_ms_p50')):>7.2f} {med(('read', 'per_page_ms_max')):>7.2f} | "
                f"{all(c['verified'] for c in cells)}"
            )


if __name__ == "__main__":
    main()
