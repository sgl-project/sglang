import concurrent.futures
import logging
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
        f"mean = {sum(x)/len(x):.2f}, "
        f"min = {min(x):.2f}, "
        f"p25 = {x[int(lenx*0.25)]:.2f}, "
        f"p50 = {x[int(lenx*0.5)]:.2f}, "
        f"p75 = {x[int(lenx*0.75)]:.2f}, "
        f"max = {max(x):.2f}"
    )


def test():
    # /path/to/hf3fs
    file_path = "/data/bench.bin"
    file_size = 1 << 40
    bytes_per_page = 16 << 20
    entries = 32
    file_ops = Hf3fsUsrBioClient(file_path, file_size, bytes_per_page, entries)

    print("test batch_read / batch_write")
    num_pages = 128
    dtype = torch.bfloat16
    numel = bytes_per_page // dtype.itemsize
    offsets = list(range(file_size // bytes_per_page))
    random.shuffle(offsets)
    offsets = offsets[:num_pages]
    offsets = [i * bytes_per_page for i in offsets]
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
    assert all([torch.allclose(r, w) for r, w in zip(tensor_reads, tensor_writes)])

    file_ops.close()
    print("test done")


def bench():
    file_path = "/data/bench.bin"
    file_size = 1 << 40
    bytes_per_page = 16 << 20
    entries = 8
    numjobs = 16

    dtype = torch.bfloat16
    numel = bytes_per_page // dtype.itemsize

    file_ops = [
        Hf3fsUsrBioClient(file_path, file_size, bytes_per_page, entries)
        for _ in range(numjobs)
    ]

    num_page = entries

    offsets = list(range(file_size // bytes_per_page))
    tensors_write = [torch.randn(numel, dtype=dtype)] * num_page
    tensors_read = [torch.empty(numel, dtype=dtype)] * num_page
    random.shuffle(offsets)

    warmup = 50
    iteration = 100

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=numjobs)

    w_bw = []
    w_size = num_page * numjobs * bytes_per_page / (1 << 30)
    for i in tqdm(range(warmup + iteration), desc="Benchmarking write (GB/s)"):
        _offsets = [
            [
                offset * bytes_per_page
                for offset in offsets[
                    (i * numjobs + j) * num_page : (i * numjobs + j + 1) * num_page
                ]
            ]
            for j in range(numjobs)
        ]
        tik = time.perf_counter()
        futures = [
            executor.submit(file_ops[j].batch_write, offset, tensors_write)
            for j, offset in enumerate(_offsets)
        ]
        results = [future.result() for future in futures]
        tok = time.perf_counter()
        if i < warmup:
            continue
        w_bw.append(w_size / (tok - tik))
        results = [
            _result == bytes_per_page for result in results for _result in result
        ]
        assert all(results)
    print_stats(w_bw)

    r_bw = []
    r_size = w_size
    for i in tqdm(range(warmup + iteration), desc="Benchmarking read (GB/s)"):
        _offsets = [
            [
                offset * bytes_per_page
                for offset in offsets[
                    (i * numjobs + j) * num_page : (i * numjobs + j + 1) * num_page
                ]
            ]
            for j in range(numjobs)
        ]
        tik = time.perf_counter()
        futures = [
            executor.submit(file_ops[j].batch_read, offset, tensors_read)
            for j, offset in enumerate(_offsets)
        ]
        results = [future.result() for future in futures]
        tok = time.perf_counter()
        if i < warmup:
            continue
        r_bw.append(r_size / (tok - tik))
        results = [
            _result == bytes_per_page for result in results for _result in result
        ]
        assert all(results)
    print_stats(r_bw)

    executor.shutdown(wait=True)
    for _file_ops in file_ops:
        _file_ops.close()
    print("bench done")


def main():
    logging.basicConfig(level=logging.INFO)
    test()
    bench()


if __name__ == "__main__":
    main()
