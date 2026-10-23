# SPDX-License-Identifier: Apache-2.0
"""Request-scoped bindings between router event IDs and HiCache storage keys."""

from __future__ import annotations

from array import array
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from sglang.srt.mem_cache.radix_cache import RadixKey


def storage_key_to_event_hashes(
    key: RadixKey,
    page_size: int,
    matched_prefix_tokens: Optional[list[int]] = None,
    last_hash: Optional[str] = None,
) -> Optional[dict[str, int]]:
    """Bind requested storage pages to router IDs without changing either hash.

    None preserves legacy unnamespaced hints. An empty mapping explicitly
    declines namespaced peer access when the complete prefix cannot be proven.
    Build before enqueue: query/read threads and PP tickets share this binding.
    """
    if (
        getattr(key, "extra_key", None) is None
        and getattr(key, "cache_salt", None) is None
    ):
        return None
    from sglang.srt.mem_cache.radix_cache import RadixKey
    from sglang.srt.mem_cache.utils import get_event_hash_str, get_storage_hash_str

    prefix = matched_prefix_tokens or []
    if (
        page_size <= 0
        or len(prefix) % page_size
        or len(key) % page_size
        or (last_hash is not None and not prefix)
    ):
        return {}
    full_tokens = array("q", prefix)
    full_tokens.extend(key.raw_token_ids())
    full_key = RadixKey(
        full_tokens,
        extra_key=key.extra_key,
        cache_salt=key.cache_salt,
        is_bigram=key.is_bigram,
    )
    storage_hashes = get_storage_hash_str(full_key, page_size=page_size)
    prefix_pages = len(prefix) // page_size
    if prefix_pages and (
        prefix_pages > len(storage_hashes)
        or storage_hashes[prefix_pages - 1] != last_hash
    ):
        return {}
    event_hashes = get_event_hash_str(full_key, page_size=page_size)
    return {
        storage_hash: int(event_hash[:16], 16)
        for storage_hash, event_hash in zip(
            storage_hashes[prefix_pages:], event_hashes[prefix_pages:]
        )
    }


def storage_operation_extra_info(operation: Any, **fields: Any) -> dict[str, Any]:
    """Carry the request binding through queries, batches, and sidecar reads."""
    fields["kv_hints"] = operation.kv_hints
    mapping = getattr(operation, "storage_key_to_event_hash", None)
    if mapping is not None:
        fields["storage_key_to_event_hash"] = mapping
    return fields
