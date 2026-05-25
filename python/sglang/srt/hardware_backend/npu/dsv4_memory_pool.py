"""DSV4-NPU memory pool subclass.

Marker subclass of :class:`DeepSeekV4TokenToKVPool`. Kept as a hook point
for future NPU-only overrides (e.g. tensor layout tweaks or NPU-specific
buffer helpers). c-page allocation lives entirely in
:class:`DSV4NPUTokenToKVPoolAllocator`; the per-req mapping tables live on
:class:`DSV4NPUReqToTokenPool`.
"""

from __future__ import annotations

from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool


class DSV4NPUTokenToKVPool(DeepSeekV4TokenToKVPool):
    pass
