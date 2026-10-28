# SPDX-License-Identifier: Apache-2.0
"""Publish a backend page only while every required attention shard owns it."""

from __future__ import annotations

from collections import defaultdict, deque

import msgspec

from sglang.srt.disaggregation.kv_events import BlockRemoved, BlockStored


class StorageEventReducer:
    def __init__(self, shard_count: int):
        self.shard_count = shard_count
        self._resident = {}
        self._metadata = {}
        self._published = set()

    def update(self, batches: dict[int, list]) -> list:
        if set(batches) != set(range(self.shard_count)):
            raise ValueError("Storage inventory does not cover every attention shard")
        touched = set()
        for shard, events in batches.items():
            for event in events:
                if not isinstance(event, (BlockStored, BlockRemoved)):
                    raise TypeError("Unsupported backend inventory event")
                for block_hash in event.block_hashes:
                    touched.add(block_hash)
                    resident = self._resident.setdefault(block_hash, set())
                    if isinstance(event, BlockStored):
                        if len(event.block_hashes) != 1:
                            raise ValueError("Storage inventory must describe one page")
                        previous = self._metadata.get(block_hash)
                        if previous is not None and previous != event:
                            raise ValueError(
                                "Attention shards disagree on page metadata"
                            )
                        self._metadata[block_hash] = event
                        resident.add(shard)
                    else:
                        resident.discard(shard)
        result = []
        for block_hash in sorted(touched):
            resident = self._resident[block_hash]
            ready = len(resident) == self.shard_count
            if ready and block_hash not in self._published:
                metadata = self._metadata[block_hash]
                # The framework recorder coalesces adjacent stores in place.
                # Never let that mutate the per-page inventory identity.
                result.append(
                    msgspec.structs.replace(
                        metadata,
                        block_hashes=list(metadata.block_hashes),
                        token_ids=list(metadata.token_ids),
                    )
                )
                self._published.add(block_hash)
            elif not ready and block_hash in self._published:
                metadata = self._metadata[block_hash]
                result.append(
                    BlockRemoved(
                        block_hashes=[block_hash],
                        medium=metadata.medium,
                        ownership=metadata.ownership,
                    )
                )
                self._published.remove(block_hash)
            if not resident:
                del self._resident[block_hash]
                self._metadata.pop(block_hash, None)
        # Native completions may arrive out of order. A trie consumer must see
        # each parent before its newly stored children, independent of hash order.
        stores = {
            event.block_hashes[0]: event
            for event in result
            if isinstance(event, BlockStored)
        }
        children = defaultdict(list)
        roots = deque()
        for block_hash, event in stores.items():
            if event.parent_block_hash in stores:
                children[event.parent_block_hash].append(block_hash)
            else:
                roots.append(block_hash)
        ordered = [event for event in result if isinstance(event, BlockRemoved)]
        visited = 0
        while roots:
            block_hash = roots.popleft()
            ordered.append(stores[block_hash])
            visited += 1
            roots.extend(children[block_hash])
        if visited != len(stores):
            raise ValueError("Cyclic storage inventory prefix metadata")
        return ordered
