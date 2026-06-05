import threading
import time

import torch
from tqdm import tqdm

from sglang.srt.distributed import (
    get_world_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.managers.cache_controller import (
    HiCacheController,
    PrefetchOperation,
    StorageOperation,
)
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.mem_cache.memory_pool_host import MHATokenToKVPoolHost

init_distributed_environment(
    world_size=1,
    rank=0,
    distributed_init_method="tcp://127.0.0.1:23456",
    local_rank=0,
    backend="gloo",
)

initialize_model_parallel(
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1,
)

group = get_world_group().cpu_group

max_total_num_tokens = 524288
page_size = 64
kv_cache_dtype = torch.bfloat16
layer_num = 64
head_num, head_dim = 8, 128
device = "cuda"
hicache_ratio = 2
hicache_size = 0
hicache_mem_layout = "page_first"
# hicache_mem_layout = "layer_first"
hicache_write_policy = "write_through"
hicache_io_backend = "kernel"
hicache_storage_backend = "hf3fs"
prefetch_threshold = 256

op_size = 1024
op_num = 16

token_to_kv_pool = MHATokenToKVPool(
    max_total_num_tokens,
    page_size=page_size,
    dtype=kv_cache_dtype,
    head_num=head_num,
    head_dim=head_dim,
    layer_num=layer_num,
    device=device,
    enable_memory_saver=True,
)

token_to_kv_pool_allocator = TokenToKVPoolAllocator(
    max_total_num_tokens,
    dtype=kv_cache_dtype,
    device=device,
    kvcache=token_to_kv_pool,
    need_sort=False,
)

kv_cache = token_to_kv_pool_allocator.get_kvcache()
token_to_kv_pool_host = MHATokenToKVPoolHost(
    kv_cache,
    hicache_ratio,
    hicache_size,
    page_size,
    hicache_mem_layout,
)

load_cache_event = threading.Event()
cache_controller = HiCacheController(
    token_to_kv_pool_allocator,
    token_to_kv_pool_host,
    page_size,
    group,
    load_cache_event=load_cache_event,
    write_policy=hicache_write_policy,
    io_backend=hicache_io_backend,
    storage_backend=hicache_storage_backend,
    prefetch_threshold=prefetch_threshold,
)

operations = [
    StorageOperation(
        torch.tensor(list(range(i, i + op_size))),
        list(range(i, i + op_size)),
        hash_value=[f"{j}" for j in range(i, i + op_size, page_size)],
    )
    for i in tqdm(range(0, op_num * op_size, op_size))
]

tik = time.monotonic()
if hicache_mem_layout == "page_first":
    for operation in operations:
        cache_controller.zerocopy_page_backup(operation, batch_size=128)
elif hicache_mem_layout == "layer_first":
    for operation in operations:
        cache_controller.generic_page_backup(operation, batch_size=128)
tok = time.monotonic()
print(f"{tok-tik:.6f} s")

operations = [
    PrefetchOperation(
        f"{i}",
        torch.tensor(list(range(i, i + op_size))),
        list(range(i, i + op_size)),
        f"{i}",
    )
    for i in tqdm(range(0, op_num * op_size, op_size))
]

for operation in operations:
    operation.hash_value = [
        f"{j}"
        for j in range(
            int(operation.last_hash), int(operation.last_hash) + op_size, page_size
        )
    ]

tik = time.monotonic()
if hicache_mem_layout == "page_first":
    for operation in operations:
        cache_controller.zerocopy_page_transfer(operation, batch_size=128)
elif hicache_mem_layout == "layer_first":
    for operation in operations:
        cache_controller.generic_page_transfer(operation, batch_size=128)
tok = time.monotonic()
print(f"{tok-tik:.6f} s")
