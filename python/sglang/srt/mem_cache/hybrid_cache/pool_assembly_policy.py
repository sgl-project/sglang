"""Shared migration eligibility for host assembly and direct linker.

Keep platform and concrete-pool selection outside generic buffer composition.
Unsupported configurations retain their existing assembly paths.
"""

from __future__ import annotations


def can_use_dsa_buffer_infos(pool, drafts, *, dcp_enabled: bool) -> bool:
    from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool
    from sglang.srt.utils import is_cuda

    return (
        is_cuda()
        and not dcp_enabled
        and all(
            type(item) is DSATokenToKVPool and not item.layer_shard_enabled
            for item in (pool, *drafts)
        )
    )
