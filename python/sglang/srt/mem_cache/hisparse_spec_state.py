# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""CPU reference union planning for page64 HiSparse verification.

Coordinator adapter contract: snapshot ONE request/layer, plan, submit every
miss copy, publish pending hot metadata, then consume every immutable row table
on the same stream. Hold hot slots and arena until the final reader completes.
This module never publishes state, allocates, frees, or translates indexer IDs.
Logical page64 IDs remain owned by the shared scheduler/indexer allocator.
"""

from dataclasses import dataclass
from typing import Mapping, Sequence

PAGE_SIZE = 64


class UnionCapacityError(ValueError):
    """No complete union can reside in the supplied hot buffer."""


def _integer(value: int, minimum: int, name: str) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


@dataclass(frozen=True)
class SpecTxnKey:
    request_slot: int
    request_generation: int
    iteration_id: int

    def __post_init__(self) -> None:
        for value in (self.request_slot, self.request_generation, self.iteration_id):
            _integer(value, 0, "transaction identity")


@dataclass(frozen=True)
class ProvisionalArena:
    """Whole physical pages, including padding; never borrowed from hot pages.

    Caller allocates ``rounded_rows(reserved_rows)`` via the physical paged
    allocator and retains every page. Active maps contain request-relative
    positions, NOT logical IDs. The coordinator separately tracks/clears logical
    write mappings. Release only the returned whole page IDs, once, after backup,
    readers and mapping removal. This immutable descriptor is NOT a release token.
    """

    key: SpecTxnKey
    reserved_rows: int
    page_ids: tuple[int, ...]
    position_slots: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if type(self.key) is not SpecTxnKey:
            raise ValueError("arena key must be a SpecTxnKey")
        if type(self.page_ids) is not tuple or type(self.position_slots) is not tuple:
            raise ValueError("arena ownership must be immutable tuples")
        expected = rounded_rows(self.reserved_rows) // PAGE_SIZE
        if len(self.page_ids) != expected or len(set(self.page_ids)) != expected:
            raise ValueError("arena must own exactly the rounded number of pages")
        for page in self.page_ids:
            _integer(page, 1, "physical page (page zero is reserved)")
        positions, slots = set(), set()
        for entry in self.position_slots:
            if type(entry) is not tuple or len(entry) != 2:
                raise ValueError("position mappings must be immutable pairs")
            position, slot = entry
            _integer(position, 0, "provisional position")
            _integer(slot, 1, "physical slot")
            if (
                position in positions
                or slot in slots
                or slot // PAGE_SIZE not in self.page_ids
            ):
                raise ValueError("duplicate or unowned provisional mapping")
            positions.add(position)
            slots.add(slot)
        if len(slots) > self.reserved_rows:
            raise ValueError("active mappings exceed reservation")


def rounded_rows(reserved_rows: int) -> int:
    _integer(reserved_rows, 1, "reserved rows")
    return ((reserved_rows + PAGE_SIZE - 1) // PAGE_SIZE) * PAGE_SIZE


@dataclass(frozen=True)
class VerifyRow:
    key: SpecTxnKey
    position: int
    selections: tuple[int, ...]


@dataclass(frozen=True)
class UnionPlan:
    key: SpecTxnKey
    layer_id: int
    row_device_tables: tuple[tuple[int, ...], ...]
    committed_positions: tuple[int, ...]
    pinned_hot_indices: tuple[int, ...]
    miss_positions: tuple[int, ...]
    miss_src: tuple[int, ...]
    miss_dst: tuple[int, ...]
    pending_hot_tokens: tuple[int, ...]


def plan_union(
    *,
    key: SpecTxnKey,
    layer_id: int,
    hot_key: SpecTxnKey,
    hot_layer_id: int,
    old_kv_len: int,
    rows: Sequence[VerifyRow],
    host_slots: Mapping[int, int],
    hot_tokens: Sequence[int],
    hot_slots: Sequence[int],
    victim_order: Sequence[int],
    arena: ProvisionalArena,
) -> UnionPlan:
    """Plan all rows atomically; caller inputs are never mutated.

    Positions are request-relative, inclusive causal bounds are row.position,
    and -1 is the sole selection/empty-hot sentinel. Rows are real verifier rows
    (batch padding must be removed). Victim order is an explicit permutation of
    hot *indices*, oldest/preferred first; no implicit LRU mutation occurs.
    Host slot zero is valid; target physical page zero is always invalid.
    ``hot_key`` stamps the coordinator snapshot for this transaction, including
    generation and iteration, rather than describing when cache data was loaded.
    """
    _integer(layer_id, 0, "layer")
    _integer(old_kv_len, 0, "committed length")
    if key != hot_key or key != arena.key or layer_id != hot_layer_id:
        raise ValueError("stale transaction or wrong request/layer snapshot")
    tokens, slots, order = tuple(hot_tokens), tuple(hot_slots), tuple(victim_order)
    if len(tokens) != len(slots) or len(set(slots)) != len(slots):
        raise ValueError("hot metadata/physical slots mismatch or alias")
    for index in order:
        _integer(index, 0, "victim index")
    if sorted(order) != list(range(len(slots))):
        raise ValueError("victim order must be a permutation")
    resident = {}
    for index, (position, slot) in enumerate(zip(tokens, slots)):
        _integer(slot, PAGE_SIZE, "hot physical slot (page zero is reserved)")
        if slot // PAGE_SIZE in arena.page_ids:
            raise ValueError(
                "hot slot overlaps owned provisional page, including padding"
            )
        _integer(position, -1, "hot position")
        if position != -1:
            if position >= old_kv_len or position in resident:
                raise ValueError("hot positions must be unique committed positions")
            resident[position] = index
    provisional = dict(arena.position_slots)
    if any(position < old_kv_len for position in provisional):
        raise ValueError("provisional mapping overlaps committed prefix")
    normalized_rows, union = [], {}
    for row in rows:
        if row.key != key:
            raise ValueError("row belongs to another transaction")
        _integer(row.position, old_kv_len, "verifier position")
        if row.position not in provisional:
            raise ValueError("verifier row has no provisional write mapping")
        selections = tuple(row.selections)
        for position in selections:
            _integer(position, -1, "selection")
            if position == -1:
                continue
            if position > row.position:
                raise ValueError("noncausal selection")
            if position < old_kv_len:
                union.setdefault(position, None)
            elif position not in provisional:
                raise ValueError("unmapped provisional selection")
        normalized_rows.append(selections)
    hits = {resident[position] for position in union if position in resident}
    misses = tuple(position for position in union if position not in resident)
    victims = tuple(index for index in order if index not in hits)
    if len(misses) > len(victims):
        raise UnionCapacityError("committed union exceeds hot buffer capacity")
    sources = []
    for position in misses:
        if position not in host_slots:
            raise ValueError("committed miss has no valid host mapping")
        source = host_slots[position]
        _integer(source, 0, "host slot")
        sources.append(source)
    if len(set(sources)) != len(sources):
        raise ValueError("distinct committed misses alias host storage")
    pending = list(tokens)
    resolved = {
        position: slots[resident[position]]
        for position in union
        if position in resident
    }
    destinations = []
    for position, index in zip(misses, victims):
        pending[index] = position
        resolved[position] = slots[index]
        destinations.append(slots[index])
    resolved.update(provisional)
    tables = tuple(
        tuple(-1 if p == -1 else resolved[p] for p in row) for row in normalized_rows
    )
    return UnionPlan(
        key,
        layer_id,
        tables,
        tuple(union),
        tuple(sorted(hits)),
        misses,
        tuple(sources),
        tuple(destinations),
        tuple(pending),
    )
