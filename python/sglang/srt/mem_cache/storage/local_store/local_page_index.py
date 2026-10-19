from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from typing import Optional

from sglang.srt.mem_cache.hicache_storage import PoolName


class PageEntry:
    __slots__ = ("key", "slot", "present", "pending", "pins")

    def __init__(self, key: str, slot: int):
        self.key = key
        self.slot = slot
        self.present: set[PoolName] = set()
        self.pending: set[PoolName] = set()
        self.pins = 0


# Per pool: (pool, [(position in the pool's keys, entry to write)]).
Reservation = list[tuple[PoolName, list[tuple[int, PageEntry]]]]


class LocalPageIndex:
    """LRU map from page key to a slot holding the page for every pool. Pinned
    entries are never evicted: lookups pin until their load ends, writes until published."""

    def __init__(self, num_slots: int):
        self.num_slots = num_slots
        self.free_slots = list(range(num_slots - 1, -1, -1))
        self.entries: OrderedDict[str, PageEntry] = OrderedDict()
        self.lookup_pins: dict[str, list[PageEntry]] = {}

    def page_exists(self, keys: Sequence[str], pool: PoolName) -> list[bool]:
        entries = self.entries
        return [key in entries and pool in entries[key].present for key in keys]

    def slot_of(self, key: str, pool: PoolName) -> Optional[int]:
        entry = self.entries.get(key)
        return entry.slot if entry is not None and pool in entry.present else None

    def pin_lookup(self, rid: str, keys: Sequence[str]) -> None:
        self.release_lookup(rid)
        entries = [self.entries[key] for key in keys if key in self.entries]
        for entry in entries:
            entry.pins += 1
        self._touch(entries)
        if entries:
            self.lookup_pins[rid] = entries

    def take_lookup_pins(self, rid: str) -> list[PageEntry]:
        return self.lookup_pins.pop(rid, [])

    def release_lookup(self, rid: str) -> None:
        self.unpin(self.lookup_pins.pop(rid, []))

    def unpin(self, entries: Sequence[PageEntry]) -> None:
        for entry in entries:
            entry.pins -= 1

    def reserve(
        self, writes: Sequence[tuple[PoolName, Sequence[str]]]
    ) -> Optional[Reservation]:
        """Claim a slot for every (pool, key) not stored yet; None when full."""
        reservation: Reservation = []
        for pool, keys in writes:
            positions = []
            reservation.append((pool, positions))
            for position, key in enumerate(keys):
                entry = self.entries.get(key)
                if entry is None:
                    entry = self._allocate(key)
                    if entry is None:
                        self.abort(reservation)
                        return None
                elif pool in entry.present or pool in entry.pending:
                    continue
                entry.pending.add(pool)
                entry.pins += 1
                positions.append((position, entry))
        return reservation

    def publish(self, reservation: Reservation) -> None:
        for pool, positions in reservation:
            for _, entry in positions:
                entry.pending.discard(pool)
                entry.present.add(pool)
                entry.pins -= 1
            self._touch([entry for _, entry in positions])

    def abort(self, reservation: Reservation) -> None:
        for pool, positions in reservation:
            for _, entry in positions:
                entry.pending.discard(pool)
                entry.pins -= 1
                if not entry.present and not entry.pending and entry.pins == 0:
                    self._free(entry)

    def clear_pins(self) -> None:
        self.lookup_pins.clear()
        for entry in list(self.entries.values()):
            entry.pins = 0
            entry.pending.clear()
            if not entry.present:
                self._free(entry)

    def _touch(self, entries: Sequence[PageEntry]) -> None:
        # Reversed so a prefix head outlives its tail.
        for entry in reversed(entries):
            if self.entries.get(entry.key) is entry:
                self.entries.move_to_end(entry.key)

    def _allocate(self, key: str) -> Optional[PageEntry]:
        if not self.free_slots:
            victim = next((e for e in self.entries.values() if e.pins == 0), None)
            if victim is None:
                return None
            self._free(victim)
        entry = PageEntry(key, self.free_slots.pop())
        self.entries[key] = entry
        return entry

    def _free(self, entry: PageEntry) -> None:
        if self.entries.get(entry.key) is entry:
            del self.entries[entry.key]
            self.free_slots.append(entry.slot)
