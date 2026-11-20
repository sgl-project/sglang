# SPDX-License-Identifier: Apache-2.0
"""Join KVCR's native DRAM residency with the framework's page identities."""

from __future__ import annotations

from collections import Counter, deque

import msgspec

from sglang.srt.disaggregation.kv_events import (
    BlockRemoved,
    BlockStored,
    StorageMedium,
)

# Dynamo's KV state agent accepts exactly this owner name for KVCR residency
# and fails the whole event source on any other value.
KVCR_OWNERSHIP = "kvcr"
# KVCR's local DRAM holds deposits in HiCache-registered pinned host memory.
# Dynamo maps CPU_PINNED to its HostPinned tier, which KVCR events may use.
KVCR_MEDIUM = StorageMedium.CPU


class KVCRInventory:
    """This shard's published KVCR DRAM pages.

    Called under the store's lock, except take(): the scheduler thread is the
    only consumer of ``events`` and drains it without that lock.

    A page is advertised from native residency events only, never from a
    host-side acknowledgement: it becomes resident once every ALL_PAGES pool
    holds it, and stops being resident when any of them drops it. Trailing
    pools (SWA, Mamba, ...) are checked at lookup, not advertised here.

    The framework identity of a page (its BlockStored) arrives with the IO
    that deposits or delivers it, as ``extra_info["kv_event_metadata"]``. It
    is kept while that IO is in flight or while the page is resident, and
    pages without one are never advertised.
    """

    def __init__(self):
        self.namespace = None
        self.required_pools = frozenset()
        self._metadata = {}
        self._resident = {}
        self._pending = Counter()
        self._published = set()
        self.events = deque()

    def configure(self, namespace: str, required_pools) -> None:
        self.namespace = namespace
        self.required_pools = frozenset(required_pools)
        if not self.required_pools:
            raise ValueError("KVCR inventory requires a physical ALL_PAGES pool")

    def _parse(self, physical_key: bytes):
        """(page, pool) of one of this shard's block keys, else None."""
        parts = physical_key.split(b"#")
        if len(parts) != 3:
            return None
        page, namespace, pool = (part.decode("ascii") for part in parts)
        return (page, pool) if namespace == self.namespace else None

    def register(self, metadata: dict, keys) -> set:
        """Hold the identities of an IO's pages; returns what release() takes."""
        retained = set()
        for physical_key in keys:
            parsed = self._parse(physical_key)
            if parsed is None or parsed[0] in retained:
                continue
            page = parsed[0]
            event = metadata.get(page)
            if event is None:
                continue
            if not isinstance(event, BlockStored) or len(event.block_hashes) != 1:
                raise ValueError("KVCR inventory requires one store payload per page")
            # The first identity stays while held: a published page must be
            # removed under the hash it was stored with.
            if page not in self._metadata:
                self._metadata[page] = msgspec.structs.replace(
                    event,
                    block_hashes=list(event.block_hashes),
                    token_ids=list(event.token_ids),
                    medium=KVCR_MEDIUM,
                    ownership=KVCR_OWNERSHIP,
                )
            self._pending[page] += 1
            retained.add(page)
            self._refresh(page)
        return retained

    def release(self, pages) -> None:
        for page in pages:
            self._pending[page] -= 1
            if self._pending[page] == 0:
                del self._pending[page]
                self._prune(page)

    def update(self, event) -> None:
        """Apply one native InventoryEvent for KVCR's local DRAM tier."""
        for physical_key in event.keys:
            parsed = self._parse(physical_key)
            if parsed is None:
                continue
            page, pool = parsed
            resident = self._resident.setdefault(page, set())
            if event.removed:
                resident.discard(pool)
            else:
                resident.add(pool)
            self._refresh(page)
            if not resident:
                del self._resident[page]
                self._prune(page)

    def _refresh(self, page) -> None:
        ready = self.required_pools.issubset(self._resident.get(page, ()))
        metadata = self._metadata.get(page)
        if ready and metadata is not None and page not in self._published:
            self.events.append(metadata)
            self._published.add(page)
        elif not ready and page in self._published:
            self.events.append(
                BlockRemoved(
                    block_hashes=list(metadata.block_hashes),
                    medium=KVCR_MEDIUM,
                    ownership=KVCR_OWNERSHIP,
                )
            )
            self._published.remove(page)

    def _prune(self, page) -> None:
        if page not in self._resident and page not in self._pending:
            self._metadata.pop(page, None)

    def take(self) -> list:
        # deque.popleft is atomic, so producers may append concurrently; a
        # list()+clear() pair could drop an event appended in between.
        events = []
        while True:
            try:
                events.append(self.events.popleft())
            except IndexError:
                return events
