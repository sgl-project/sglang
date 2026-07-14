"""Token-to-KV-slot allocators. One file per allocation strategy."""

from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.paged import (
    PagedTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator

__all__ = [
    "BaseTokenToKVPoolAllocator",
    "PagedTokenToKVPoolAllocator",
    "TokenToKVPoolAllocator",
]
