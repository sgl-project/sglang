import argparse
import concurrent.futures
import logging
import os
import random
import time
from typing import List

import torch
from tqdm import tqdm

from sglang.srt.mem_cache.storage.hf3fs.hf3fs_usrbio_client import Hf3fsUsrBioClient


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
    entries = 32
    file_ops = Hf3fsUsrBioClient(
        args.file_path, args.file_size, args.bytes_per_page, entries, 5
    )

    print("test batch_read / batch_write")
    num_pages = min(128, args.file_size // args.bytes_per_page)
    dtype = torch.bfloat16
    numel = args.bytes_per_page // dtype.itemsize
    offsets = list(range(args.file_size // args.bytes_per_page))
    random.shuffle(offsets)
    offsets = offsets[:num_pages]
    offsets = [i * args.bytes_per_page for i in offsets]
    tensor_writes = [
        torch.randn(numel, dtype=dtype)
        for _ in tqdm(range(num_pages), desc="prepare tensor")
    ]
    for i in tqdm(range(0, num_pages, file_ops.entries), desc="batch_write"):
        results = file_ops.batch_write(
            offsets[i : i + file_ops.entries], tensor_writes[i : i + file_ops.entries]
        )
        assert all([result == numel * dtype.itemsize for result in results])
    tensor_reads = [
        torch.empty(numel, dtype=dtype)
        for _ in tqdm(range(num_pages), desc="prepare tensor")
    ]
    for i in tqdm(range(0, num_pages, file_ops.entries), desc="batch_read"):
        results = file_ops.batch_read(
            offsets[i : i + file_ops.entries], tensor_reads[i : i + file_ops.entries]
        )
        assert all([result == numel * dtype.itemsize for result in results])
    assert all([torch.equal(r, w) for r, w in zip(tensor_reads, tensor_writes)])

    file_ops.close()
    print("test done")


def bench(args):
    dtype = torch.bfloat16
    numel = args.bytes_per_page // dtype.itemsize

    file_ops = [
        Hf3fsUsrBioClient(
            args.file_path, args.file_size, args.bytes_per_page, args.entries, 5
        )
        for _ in range(args.numjobs)
    ]

    num_page = args.entries

    offsets = list(range(args.file_size // args.bytes_per_page))
    tensors_write = [
        [torch.randn(numel, dtype=dtype) for _ in range(num_page)]
        for _ in range(args.numjobs)
    ]
    tensors_read = [
        [torch.empty(numel, dtype=dtype) for _ in range(num_page)]
        for _ in range(args.numjobs)
    ]
    random.shuffle(offsets)

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=args.numjobs)

    w_bw = []
    w_size = num_page * args.numjobs * args.bytes_per_page / (1 << 30)
    for i in tqdm(
        range(args.warmup + args.iterations), desc="Benchmarking write (GiB/s)"
    ):
        _offsets = [
            [
                offset * args.bytes_per_page
                for offset in offsets[
                    (i * args.numjobs + j) * num_page : (i * args.numjobs + j + 1)
                    * num_page
                ]
            ]
            for j in range(args.numjobs)
        ]
        tik = time.perf_counter()
        futures = [
            executor.submit(file_ops[j].batch_write, offset, tensors_write[j])
            for j, offset in enumerate(_offsets)
        ]
        results = [future.result() for future in futures]
        tok = time.perf_counter()
        if i < args.warmup:
            continue
        w_bw.append(w_size / (tok - tik))
        results = [
            _result == args.bytes_per_page for result in results for _result in result
        ]
        assert all(results)
    print_stats(w_bw)

    r_bw = []
    r_size = w_size
    for i in tqdm(
        range(args.warmup + args.iterations), desc="Benchmarking read (GiB/s)"
    ):
        _offsets = [
            [
                offset * args.bytes_per_page
                for offset in offsets[
                    (i * args.numjobs + j) * num_page : (i * args.numjobs + j + 1)
                    * num_page
                ]
            ]
            for j in range(args.numjobs)
        ]
        tik = time.perf_counter()
        futures = [
            executor.submit(file_ops[j].batch_read, offset, tensors_read[j])
            for j, offset in enumerate(_offsets)
        ]
        results = [future.result() for future in futures]
        tok = time.perf_counter()
        if i < args.warmup:
            continue
        r_bw.append(r_size / (tok - tik))
        results = [
            _result == args.bytes_per_page for result in results for _result in result
        ]
        assert all(results)
    print_stats(r_bw)

    executor.shutdown(wait=True)
    for _file_ops in file_ops:
        _file_ops.close()
    print("bench done")


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--file-path", default="/data/bench.bin")
    parser.add_argument(
        "--file-size", type=int, default=1 << 40, help="File size in bytes"
    )
    parser.add_argument(
        "--bytes-per-page", type=int, default=16 << 20, help="Bytes per I/O block"
    )
    parser.add_argument("--numjobs", type=int, default=16, help="Concurrent clients")
    parser.add_argument(
        "--entries", type=int, default=8, help="I/O ring entries per client"
    )
    parser.add_argument("--warmup", type=int, default=50, help="Warmup iterations")
    parser.add_argument(
        "--iterations", type=int, default=100, help="Measured iterations"
    )
    args = parser.parse_args()
    for name in ("file_size", "bytes_per_page", "numjobs", "entries", "iterations"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be nonnegative")
    if args.bytes_per_page % torch.bfloat16.itemsize:
        parser.error("--bytes-per-page must be divisible by the bfloat16 element size")
    required_pages = (args.warmup + args.iterations) * args.numjobs * args.entries
    if args.file_size // args.bytes_per_page < required_pages:
        parser.error("--file-size must hold every warmup and measured batch")
    return args


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    test(args)
    bench(args)
    os.remove(args.file_path)


if __name__ == "__main__":
    main()
