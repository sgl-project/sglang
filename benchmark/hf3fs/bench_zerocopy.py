import argparse
import os
import time

import torch
from tqdm import tqdm

from sglang.srt.distributed import (
    cleanup_dist_env_and_memory,
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
from sglang.srt.mem_cache.storage.hf3fs.mini_3fs_metadata_server import (
    Hf3fsLocalMetadataClient,
)
from sglang.srt.mem_cache.storage.hf3fs.storage_hf3fs import HiCacheHF3FS
from sglang.srt.runtime_context import reset_context
from sglang.test.test_utils import publish_build_topology


def bench(file_path):
    # Qwen3-32B
    page_size = 64
    kv_cache_dtype = torch.bfloat16
    layer_num = 64
    head_num, head_dim = 8, 128
    device = "cuda"
    hicache_ratio = 2
    hicache_size = 0
    hicache_mem_layout = "page_first"
    op_size = 1024
    op_num = 16
    max_total_num_tokens = op_size * op_num

    kv_cache = MHATokenToKVPool(
        max_total_num_tokens,
        page_size=page_size,
        dtype=kv_cache_dtype,
        head_num=head_num,
        head_dim=head_dim,
        layer_num=layer_num,
        device=device,
        enable_memory_saver=True,
    )
    host_pool = MHATokenToKVPoolHost(
        kv_cache, hicache_ratio, hicache_size, page_size, hicache_mem_layout
    )

    storage = None
    try:
        storage = HiCacheHF3FS(
            rank=0,
            file_path=file_path,
            file_size=1 << 40,
            numjobs=16,
            bytes_per_page=host_pool.get_ksize_per_token() * page_size,
            entries=8,
            client_timeout=5,
            dtype=kv_cache_dtype,
            metadata_client=Hf3fsLocalMetadataClient(),
        )
        storage.register_mem_pool_host(host_pool)

        for index in tqdm(
            range(0, max_total_num_tokens, page_size), desc="Initializing host pages"
        ):
            host_pool.get_data_page(index, flat=False).normal_()

        operations = [
            (
                [f"{j}" for j in range(i, i + op_size, page_size)],
                torch.arange(i, i + op_size, dtype=torch.int64),
            )
            for i in range(0, max_total_num_tokens, op_size)
        ]
        data_size = max_total_num_tokens * host_pool.get_size_per_token() / (1 << 30)

        write_results = []
        tik = time.monotonic()
        for keys, host_indices in operations:
            write_results.extend(storage.batch_set_v1(keys, host_indices))
        elapsed = time.monotonic() - tik
        assert all(write_results)
        print(f"Write: {elapsed:.6f} s, {data_size / elapsed:.2f} GiB/s")

        read_results = []
        tik = time.monotonic()
        for keys, host_indices in operations:
            read_results.extend(storage.batch_get_v1(keys, host_indices))
        elapsed = time.monotonic() - tik
        assert all(read_results)
        print(f"Read: {elapsed:.6f} s, {data_size / elapsed:.2f} GiB/s")
    finally:
        if storage is not None:
            storage.close()
        host_pool.destroy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-path", default="/data/bench_zerocopy.bin")
    args = parser.parse_args()
    try:
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method="tcp://127.0.0.1:23456",
            local_rank=0,
            backend="gloo",
        )
        publish_build_topology()
        initialize_model_parallel()
        bench(args.file_path)
    finally:
        cleanup_dist_env_and_memory()
        reset_context()
    os.remove(args.file_path)


if __name__ == "__main__":
    main()
