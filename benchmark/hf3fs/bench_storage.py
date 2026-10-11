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


def test(file_path):
    # Qwen3-32B
    layer_num = 64
    head_num, head_dim = 8, 128
    store_dtype = torch.bfloat16
    tokens_per_page = 64

    file_size = 128 << 20
    numjobs = 16
    bytes_per_page = 16 << 20
    entries = 2
    dtype = store_dtype

    hicache_hf3fs = HiCacheHF3FS(
        rank=0,
        file_path=file_path,
        file_size=file_size,
        numjobs=numjobs,
        bytes_per_page=bytes_per_page,
        entries=entries,
        client_timeout=5,
        dtype=dtype,
        metadata_client=Hf3fsLocalMetadataClient(),
    )

    numel = 2 * tokens_per_page * layer_num * head_num * head_dim
    assert numel * dtype.itemsize == bytes_per_page

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


def bench(file_path):
    # Qwen3-32B
    layer_num = 64
    head_num, head_dim = 8, 128
    store_dtype = torch.bfloat16
    tokens_per_page = 64

    file_size = 1 << 40
    numjobs = 16
    bytes_per_page = 16 << 20
    entries = 8
    dtype = store_dtype
    hicache_hf3fs = HiCacheHF3FS(
        rank=0,
        file_path=file_path,
        file_size=file_size,
        numjobs=numjobs,
        bytes_per_page=bytes_per_page,
        entries=entries,
        client_timeout=5,
        dtype=dtype,
        metadata_client=Hf3fsLocalMetadataClient(),
    )

    numel = 2 * tokens_per_page * layer_num * head_num * head_dim
    assert numel * dtype.itemsize == bytes_per_page

    num_page = 128
    values = [torch.randn((numel,)).to(dtype=dtype) for _ in tqdm(range(num_page))]
    read_values = [torch.empty_like(value) for value in values]
    rank_metadata = hicache_hf3fs.metadata_client._get_metadata(
        hicache_hf3fs.rank, PoolName.KV
    )

    warmup = 50
    iteration = 100

    w_bw = []
    w_size = num_page * bytes_per_page / (1 << 30)
    for i in tqdm(range(warmup + iteration), desc="Benchmarking write (GiB/s)"):
        keys = [f"{j}" for j in range(i * num_page, (i + 1) * num_page)]
        tik = time.perf_counter()
        ok = hicache_hf3fs._batch_set(keys, values)
        tok = time.perf_counter()
        if i < warmup:
            continue
        w_bw.append(w_size / (tok - tik))
        assert all(ok)
    print_stats(w_bw)

    r_bw = []
    r_size = num_page * bytes_per_page / (1 << 30)
    for i in tqdm(range(warmup + iteration), desc="Benchmarking read (GiB/s)"):
        keys = random.sample(
            list(rank_metadata.key_to_index.keys()),
            num_page,
        )
        tik = time.perf_counter()
        results = hicache_hf3fs._batch_get(keys, read_values)
        tok = time.perf_counter()
        if i < warmup:
            continue
        r_bw.append(r_size / (tok - tik))
        assert all(results)
    print_stats(r_bw)

    hicache_hf3fs.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-path", default="/data/test.bin")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    test(args.file_path)
    bench(args.file_path)
    os.remove(args.file_path)


if __name__ == "__main__":
    main()
