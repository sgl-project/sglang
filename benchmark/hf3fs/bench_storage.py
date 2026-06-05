import json
import logging
import os
import random
import time
from typing import List

import torch
from tqdm import tqdm

from sglang.srt.mem_cache.storage.hf3fs.mini_3fs_metadata_server import (
    Hf3fsLocalMetadataClient,
)
from sglang.srt.mem_cache.storage.hf3fs.storage_hf3fs import HiCacheHF3FS


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
    # Qwen3-32B
    layer_num = 64
    head_num, head_dim = 8, 128
    kv_lora_rank, qk_rope_head_dim = 0, 0
    store_dtype = torch.bfloat16
    tokens_per_page = 64

    file_path_prefix = "/data/test"
    file_size = 128 << 20
    numjobs = 16
    bytes_per_page = 16 << 20
    entries = 2
    dtype = store_dtype

    config_path = os.getenv(HiCacheHF3FS.default_env_var)
    assert config_path
    try:
        with open(config_path, "w") as f:
            json.dump(
                {
                    "file_path_prefix": file_path_prefix,
                    "file_size": file_size,
                    "numjobs": numjobs,
                    "entries": entries,
                },
                f,
            )
    except Exception as e:
        raise RuntimeError(f"Failed to dump config to {config_path}: {str(e)}")
    hicache_hf3fs = HiCacheHF3FS.from_env_config(bytes_per_page, dtype)

    numel = 2 * tokens_per_page * layer_num * head_num * head_dim
    assert numel * dtype.itemsize == bytes_per_page

    num_pages = 10
    tensors = {}
    for i in range(num_pages):
        k = f"key_{i}"
        v = torch.randn((numel,)).to(dtype=dtype)
        ok = hicache_hf3fs.set(k, v)
        if i < (file_size // bytes_per_page):
            assert ok, f"Failed to insert {k}"
        else:
            assert not ok
        tensors[k] = v
    assert hicache_hf3fs.get("key_8") is None
    assert hicache_hf3fs.get("key_9") is None

    start = 0
    for i in range(start, start + hicache_hf3fs.num_pages):
        k = f"key_{i}"
        assert hicache_hf3fs.exists(k)
        out = hicache_hf3fs.get(k)
        assert out is not None
        v = tensors[k]
        assert torch.allclose(v, out, atol=1e-3), f"Tensor mismatch for {k}"

    assert not hicache_hf3fs.exists("not_exists")

    hicache_hf3fs.delete("key_7")
    v2 = torch.randn((numel,)).to(dtype=dtype)
    assert hicache_hf3fs.set("key_new", v2)
    assert torch.allclose(hicache_hf3fs.get("key_new"), v2, atol=1e-3)

    hicache_hf3fs.clear()
    assert (
        len(hicache_hf3fs.metadata_client.rank_metadata.free_pages)
        == hicache_hf3fs.metadata_client.rank_metadata.num_pages
    )

    # batch
    num_pages = 10
    tensors = {}
    keys = []
    values = []
    for i in range(num_pages):
        k = f"key_{i}"
        keys.append(k)
        v = torch.randn((numel,)).to(dtype=dtype)
        values.append(v)

    ok = hicache_hf3fs.batch_set(keys, values)
    assert not ok
    assert hicache_hf3fs.get("key_8") is None
    assert hicache_hf3fs.get("key_9") is None

    results = hicache_hf3fs.batch_get(keys[: hicache_hf3fs.num_pages])
    for result, key, value in zip(
        results, keys[: hicache_hf3fs.num_pages], values[: hicache_hf3fs.num_pages]
    ):
        assert torch.allclose(value, result, atol=1e-3), f"Tensor mismatch for {key}"

    hicache_hf3fs.close()
    os.remove(hicache_hf3fs.file_path)

    print("All test cases passed.")


def bench():
    # Qwen3-32B
    layer_num = 64
    head_num, head_dim = 8, 128
    kv_lora_rank, qk_rope_head_dim = 0, 0
    store_dtype = torch.bfloat16
    tokens_per_page = 64

    file_path = "/data/test.bin"
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
        dtype=dtype,
        metadata_client=Hf3fsLocalMetadataClient(),
    )

    numel = 2 * tokens_per_page * layer_num * head_num * head_dim
    assert numel * dtype.itemsize == bytes_per_page

    num_page = 128
    values = [torch.randn((numel,)).to(dtype=dtype) for _ in tqdm(range(num_page))]

    warmup = 50
    iteration = 100

    w_bw = []
    w_size = num_page * bytes_per_page / (1 << 30)
    for i in tqdm(range(warmup + iteration), desc="Benchmarking write (GB/s)"):
        keys = [f"{j}" for j in range(i * num_page, (i + 1) * num_page)]
        tik = time.perf_counter()
        ok = hicache_hf3fs.batch_set(keys, values)
        tok = time.perf_counter()
        if i < warmup:
            continue
        w_bw.append(w_size / (tok - tik))
        assert ok
    print_stats(w_bw)

    r_bw = []
    r_size = num_page * bytes_per_page / (1 << 30)
    for i in tqdm(range(warmup + iteration), desc="Benchmarking read (GB/s)"):
        keys = random.sample(
            list(hicache_hf3fs.metadata_client.rank_metadata.key_to_index.keys()),
            num_page,
        )
        tik = time.perf_counter()
        results = hicache_hf3fs.batch_get(keys)
        tok = time.perf_counter()
        if i < warmup:
            continue
        r_bw.append(r_size / (tok - tik))
        assert all([r is not None for r in results])
    print_stats(r_bw)

    hicache_hf3fs.close()


def allclose():
    # Qwen3-32B
    layer_num = 64
    head_num, head_dim = 8, 128
    kv_lora_rank, qk_rope_head_dim = 0, 0
    store_dtype = torch.bfloat16
    tokens_per_page = 64

    file_path = "/data/test.bin"
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
        dtype=dtype,
        metadata_client=Hf3fsLocalMetadataClient(),
    )

    numel = 2 * tokens_per_page * layer_num * head_num * head_dim
    assert numel * dtype.itemsize == bytes_per_page

    num_page = 128
    values = [torch.randn((numel,)).to(dtype=dtype) for _ in tqdm(range(num_page))]

    iteration = 100

    for i in tqdm(range(iteration), desc="Benchmarking write (GB/s)"):
        keys = [f"{j}" for j in range(i * num_page, (i + 1) * num_page)]
        ok = hicache_hf3fs.batch_set(keys, values)
        assert ok

    read_keys, read_results = [], []
    for i in tqdm(range(iteration), desc="Benchmarking read (GB/s)"):
        keys = random.sample(
            list(hicache_hf3fs.metadata_client.rank_metadata.key_to_index.keys()),
            num_page,
        )
        results = hicache_hf3fs.batch_get(keys)
        read_keys.extend(keys)
        read_results.extend(results)
        assert all([r is not None for r in results])

    for key, result in tqdm(zip(read_keys, read_results)):
        assert torch.allclose(values[int(key) % num_page], result, atol=1e-3)

    hicache_hf3fs.close()


def main():
    logging.basicConfig(level=logging.INFO)
    test()
    bench()
    allclose()


if __name__ == "__main__":
    main()
