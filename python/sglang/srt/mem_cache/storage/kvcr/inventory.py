# SPDX-License-Identifier: Apache-2.0
"""Join native KVCR residency with framework-owned prefix event metadata."""

from __future__ import annotations

from collections import Counter, deque

import msgspec

from sglang.srt.disaggregation.kv_events import (
    BlockRemoved,
    BlockStored,
    StorageMedium,
)


class KVCRInventory:
    """Called under the store's lock; never infer residency from a host ACK.

    ALL_PAGES pools jointly own one advertised logical page. Trailing state is
    still checked at lookup/admission, not claimed as a contiguous prefix here.
    Metadata is held only by an active IO or an actual native residency.
    """

    def __init__(self):
        self.namespace = None
        self.required_pools = frozenset()
        self._metadata = {}
        self._resident = {}
        self._pending = Counter()
        self._published = set()
        self.events = deque()

    def configure(self, namespace, required_pools):
        self.namespace = namespace
        self.required_pools = frozenset(required_pools)
        if not self.required_pools:
            raise ValueError("KVCR inventory requires a physical ALL_PAGES pool")

    def register(self, metadata, keys):
        retained = set()
        for physical_key in keys:
            page, namespace, _ = physical_key.decode("ascii").split("#")
            if namespace != self.namespace or page in retained:
                continue
            event = metadata.get(page)
            if event is None:
                continue
            if not isinstance(event, BlockStored) or len(event.block_hashes) != 1:
                raise ValueError("KVCR inventory requires one store payload per page")
            self._metadata[page] = msgspec.structs.replace(
                event, medium=StorageMedium.CPU, ownership="kvcr"
            )
            self._pending[page] += 1
            retained.add(page)
            self._refresh(page)
        return retained

    def release(self, pages):
        for page in pages:
            self._pending[page] -= 1
            if self._pending[page] == 0:
                del self._pending[page]
                self._prune(page)

    def update(self, event):
        for physical_key in event.keys:
            page, namespace, pool = physical_key.decode("ascii").split("#")
            if namespace != self.namespace:
                continue
            resident = self._resident.setdefault(page, set())
            if event.removed:
                resident.discard(pool)
            else:
                resident.add(pool)
            self._refresh(page)
            if not resident:
                del self._resident[page]
                self._prune(page)

    def _refresh(self, page):
        ready = self.required_pools.issubset(self._resident.get(page, ()))
        metadata = self._metadata.get(page)
        if ready and metadata is not None and page not in self._published:
            self.events.append(metadata)
            self._published.add(page)
        elif not ready and page in self._published:
            self.events.append(
                BlockRemoved(
                    block_hashes=list(metadata.block_hashes),
                    medium=StorageMedium.CPU,
                    ownership="kvcr",
                )
            )
            self._published.remove(page)

    def _prune(self, page):
        if page not in self._resident and page not in self._pending:
            self._metadata.pop(page, None)

    def take(self):
        events = list(self.events)
        self.events.clear()
        return events
