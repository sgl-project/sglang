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


def bench(args):
    # Qwen3-32B
    kv_cache_dtype = torch.bfloat16
    layer_num = 64
    head_num, head_dim = 8, 128
    device = "cuda"
    hicache_ratio = 2
    hicache_size = 0
    hicache_mem_layout = "page_first"
    max_total_num_tokens = args.tokens_per_batch * args.num_batches
    bytes_per_token = 2 * layer_num * head_num * head_dim * kv_cache_dtype.itemsize
    if args.file_size < max_total_num_tokens * bytes_per_token:
        raise ValueError("--file-size must hold all K/V pages in the workload")

    kv_cache = MHATokenToKVPool(
        max_total_num_tokens,
        page_size=args.page_size,
        dtype=kv_cache_dtype,
        head_num=head_num,
        head_dim=head_dim,
        layer_num=layer_num,
        device=device,
        enable_memory_saver=True,
    )
    host_pool = MHATokenToKVPoolHost(
        kv_cache, hicache_ratio, hicache_size, args.page_size, hicache_mem_layout
    )

    storage = None
    try:
        storage = HiCacheHF3FS(
            rank=0,
            file_path=args.file_path,
            file_size=args.file_size,
            numjobs=args.numjobs,
            bytes_per_page=host_pool.get_ksize_per_token() * args.page_size,
            entries=args.entries,
            client_timeout=5,
            dtype=kv_cache_dtype,
            metadata_client=Hf3fsLocalMetadataClient(),
        )
        storage.register_mem_pool_host(host_pool)

        for index in tqdm(
            range(0, max_total_num_tokens, args.page_size),
            desc="Initializing host pages",
        ):
            host_pool.get_data_page(index, flat=False).normal_()

        operations = [
            (
                [f"{j}" for j in range(i, i + args.tokens_per_batch, args.page_size)],
                torch.arange(i, i + args.tokens_per_batch, dtype=torch.int64),
            )
            for i in range(0, max_total_num_tokens, args.tokens_per_batch)
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


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--file-path", default="/data/bench_zerocopy.bin")
    parser.add_argument(
        "--file-size", type=int, default=1 << 40, help="File size in bytes"
    )
    parser.add_argument("--numjobs", type=int, default=16, help="Concurrent clients")
    parser.add_argument(
        "--entries", type=int, default=8, help="I/O ring entries per client"
    )
    parser.add_argument("--page-size", type=int, default=64, help="Tokens per KV page")
    parser.add_argument(
        "--tokens-per-batch", type=int, default=1024, help="Tokens per read/write batch"
    )
    parser.add_argument(
        "--num-batches", type=int, default=16, help="Read/write batches"
    )
    args = parser.parse_args()
    for name in (
        "file_size",
        "numjobs",
        "entries",
        "page_size",
        "tokens_per_batch",
        "num_batches",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.tokens_per_batch % args.page_size:
        parser.error("--tokens-per-batch must be divisible by --page-size")
    return args


def main():
    args = parse_args()
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
        bench(args)
    finally:
        cleanup_dist_env_and_memory()
        reset_context()
    os.remove(args.file_path)


if __name__ == "__main__":
    main()
