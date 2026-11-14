import argparse
import logging
import os
import random
import time
from typing import List

import torch
from tqdm import tqdm

from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.storage.hf3fs.mini_3fs_metadata_server import (
    Hf3fsLocalMetadataClient,
)
from sglang.srt.mem_cache.storage.hf3fs.storage_hf3fs import HiCacheHF3FS


def print_stats(x: List[int]):
    x = sorted(x)
    lenx = len(x)
    print(
        f"mean = {sum(x) / len(x):.2f}, "
        f"min = {min(x):.2f}, "
        f"p25 = {x[int(lenx * 0.25)]:.2f}, "
        f"p50 = {x[int(lenx * 0.5)]:.2f}, "
        f"p75 = {x[int(lenx * 0.75)]:.2f}, "
        f"max = {max(x):.2f}"
    )


def test(args):
    file_size = 8 * args.bytes_per_page
    entries = 2
    dtype = torch.bfloat16

    hicache_hf3fs = HiCacheHF3FS(
        rank=0,
        file_path=args.file_path,
        file_size=file_size,
        numjobs=args.numjobs,
        bytes_per_page=args.bytes_per_page,
        entries=entries,
        client_timeout=5,
        dtype=dtype,
        metadata_client=Hf3fsLocalMetadataClient(),
    )

    numel = args.bytes_per_page // dtype.itemsize

    num_pages = 10
    tensors = {}
    for i in range(num_pages):
        k = f"key_{i}"
        v = torch.randn((numel,)).to(dtype=dtype)
        ok = hicache_hf3fs._batch_set([k], [v])[0]
        assert ok, f"Failed to insert {k}"
        tensors[k] = v
    out = torch.empty(numel, dtype=dtype)
    assert not hicache_hf3fs._batch_get(["key_0"], [out])[0]
    assert not hicache_hf3fs._batch_get(["key_1"], [out])[0]

    start = num_pages - hicache_hf3fs.num_pages
    for i in range(start, start + hicache_hf3fs.num_pages):
        k = f"key_{i}"
        assert hicache_hf3fs.exists(k)
        assert hicache_hf3fs._batch_get([k], [out])[0]
        v = tensors[k]
        assert torch.equal(v, out), f"Tensor mismatch for {k}"

    assert not hicache_hf3fs.exists("not_exists")

    hicache_hf3fs.delete("key_7")
    v2 = torch.randn((numel,)).to(dtype=dtype)
    assert hicache_hf3fs._batch_set(["key_new"], [v2])[0]
    assert hicache_hf3fs._batch_get(["key_new"], [out])[0]
    assert torch.equal(out, v2)

    hicache_hf3fs.clear()
    rank_metadata = hicache_hf3fs.metadata_client._get_metadata(
        hicache_hf3fs.rank, PoolName.KV
    )
    assert len(rank_metadata.free_pages) == rank_metadata.num_pages

    # batch
    num_pages = hicache_hf3fs.num_pages
    del tensors
    keys = []
    values = []
    for i in range(num_pages):
        k = f"key_{i}"
        keys.append(k)
        v = torch.randn((numel,)).to(dtype=dtype)
        values.append(v)

    ok = hicache_hf3fs._batch_set(keys, values)
    assert all(ok)
    assert not hicache_hf3fs.exists("key_8")
    assert not hicache_hf3fs.exists("key_9")

    read_values = [torch.empty_like(value) for value in values]
    results = hicache_hf3fs._batch_get(keys, read_values)
    assert all(results)
    for result, key, value in zip(read_values, keys, values):
        assert torch.equal(value, result), f"Tensor mismatch for {key}"

    hicache_hf3fs.close()

    print("All test cases passed.")


def bench(args):
    dtype = torch.bfloat16
    hicache_hf3fs = HiCacheHF3FS(
        rank=0,
        file_path=args.file_path,
        file_size=args.file_size,
        numjobs=args.numjobs,
        bytes_per_page=args.bytes_per_page,
        entries=args.entries,
        client_timeout=5,
        dtype=dtype,
        metadata_client=Hf3fsLocalMetadataClient(),
    )

    numel = args.bytes_per_page // dtype.itemsize

    num_page = args.batch_pages
    values = [torch.randn((numel,)).to(dtype=dtype) for _ in tqdm(range(num_page))]
    read_values = [torch.empty_like(value) for value in values]
    rank_metadata = hicache_hf3fs.metadata_client._get_metadata(
        hicache_hf3fs.rank, PoolName.KV
    )

    w_bw = []
    w_size = num_page * args.bytes_per_page / (1 << 30)
    for i in tqdm(
        range(args.warmup + args.iterations), desc="Benchmarking write (GiB/s)"
    ):
        keys = [f"{j}" for j in range(i * num_page, (i + 1) * num_page)]
        tik = time.perf_counter()
        ok = hicache_hf3fs._batch_set(keys, values)
        tok = time.perf_counter()
        if i < args.warmup:
            continue
        w_bw.append(w_size / (tok - tik))
        assert all(ok)
    print_stats(w_bw)

    r_bw = []
    r_size = num_page * args.bytes_per_page / (1 << 30)
    for i in tqdm(
        range(args.warmup + args.iterations), desc="Benchmarking read (GiB/s)"
    ):
        keys = random.sample(
            list(rank_metadata.key_to_index.keys()),
            num_page,
        )
        tik = time.perf_counter()
        results = hicache_hf3fs._batch_get(keys, read_values)
        tok = time.perf_counter()
        if i < args.warmup:
            continue
        r_bw.append(r_size / (tok - tik))
        assert all(results)
    print_stats(r_bw)

    hicache_hf3fs.close()


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--file-path", default="/data/test.bin")
    parser.add_argument(
        "--file-size", type=int, default=1 << 40, help="File size in bytes"
    )
    parser.add_argument(
        "--bytes-per-page", type=int, default=16 << 20, help="Bytes per cache slot"
    )
    parser.add_argument("--numjobs", type=int, default=16, help="Concurrent clients")
    parser.add_argument(
        "--entries", type=int, default=8, help="I/O ring entries per client"
    )
    parser.add_argument(
        "--batch-pages", type=int, default=128, help="Cache pages per batch"
    )
    parser.add_argument("--warmup", type=int, default=50, help="Warmup iterations")
    parser.add_argument(
        "--iterations", type=int, default=100, help="Measured iterations"
    )
    args = parser.parse_args()
    for name in (
        "file_size",
        "bytes_per_page",
        "numjobs",
        "entries",
        "batch_pages",
        "iterations",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be nonnegative")
    if args.bytes_per_page % torch.bfloat16.itemsize:
        parser.error("--bytes-per-page must be divisible by the bfloat16 element size")
    if args.file_size // args.bytes_per_page < args.batch_pages:
        parser.error("--file-size must hold at least --batch-pages cache pages")
    return args


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    test(args)
    bench(args)
    os.remove(args.file_path)


if __name__ == "__main__":
    main()
