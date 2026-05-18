# SPDX-License-Identifier: Apache-2.0
"""gRPC (+ UDS SCM_RIGHTS FD) KV L3 cache transport for HiCache."""

from sglang.srt.mem_cache.storage.rpc.hicache_rpc_storage import (
    HiCacheRpcStorage,
    MemfdTensorAllocator,
)

__all__ = [
    "HiCacheRpcStorage",
    "MemfdTensorAllocator",
]
