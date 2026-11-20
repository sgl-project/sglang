# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from array import array
from collections.abc import Sequence
from typing import TYPE_CHECKING

from sglang.srt.disaggregation.kv_events import namespaced_block_hash
from sglang.srt.mem_cache.utils import (
    get_event_hash_str,
    get_storage_hash_str,
    hash_str_to_int64,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.radix_cache import RadixKey


def storage_key_to_event_hashes(
    key: RadixKey,
    page_size: int,
    matched_prefix_tokens: Sequence[int] | None,
    last_hash: str | None,
    *,
    namespace_seed: bytes | None,
) -> dict[str, int]:
    """Map namespaced storage pages to published event IDs.

    An empty result means the prefix needed to reproduce the event IDs could
    not be verified. A hint consumer must treat those pages as unbound.
    """
    prefix = matched_prefix_tokens or ()
    if len(prefix) % page_size or (last_hash is not None and not prefix):
        return {}

    tokens = array("q", prefix)
    tokens.extend(key.raw_token_ids())
    full_key = type(key)(
        tokens,
        extra_key=key.extra_key,
        cache_salt=key.cache_salt,
        is_bigram=key.is_bigram,
    )
    storage_hashes = get_storage_hash_str(full_key, page_size=page_size)
    prefix_pages = len(prefix) // page_size
    if prefix_pages and storage_hashes[prefix_pages - 1] != last_hash:
        return {}

    event_hashes = get_event_hash_str(full_key, page_size=page_size)
    return {
        storage_key: namespaced_block_hash(
            hash_str_to_int64(event_hash), namespace_seed=namespace_seed
        )
        & ((1 << 64) - 1)
        for storage_key, event_hash in zip(
            storage_hashes[prefix_pages:], event_hashes[prefix_pages:], strict=True
        )
    }
