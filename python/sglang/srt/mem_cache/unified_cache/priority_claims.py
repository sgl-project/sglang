# SPDX-License-Identifier: Apache-2.0
"""Optional copy-lifetime side metadata owned by one Python TreeCore."""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

from sglang.srt.managers.kv_control.contracts import ControlError, Identifier, validated
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType

if TYPE_CHECKING:
    from sglang.srt.managers.kv_control.contracts import SetPriority
    from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeNode


@dataclass(frozen=True)
class CacheControlConfig:
    namespace: str
    max_copy_units: int = 65536
    max_claims_per_copy: int = 16
    max_query_nodes: int = 4096
    max_result_targets: int = 4096

    def __post_init__(self):
        validated(self.namespace, Identifier)
        for name in (
            "max_copy_units",
            "max_claims_per_copy",
            "max_query_nodes",
            "max_result_targets",
        ):
            if (
                type(getattr(self, name)) is not int
                or not 1 <= getattr(self, name) <= 1 << 24
            ):
                raise ValueError(f"{name} must be a positive bounded integer")


@dataclass(frozen=True)
class CopySelection:
    node: UnifiedTreeNode
    component: ComponentType
    start: int
    end: int


@dataclass(frozen=True)
class _Claim:
    revision: int
    value: int | None
    ttl: float | None
    expires_at: float


@dataclass
class _Copy:
    value: object
    incarnation: int
    units: int
    claims: dict[tuple[str, str], _Claim] = field(default_factory=dict)


class CopyPriorityClaims:
    """No allocator ownership, expiry heap, per-token table or cache-hit renewal.

    A token copy reserves its maximum possible page-fragment count when first
    claimed. Splits can therefore preserve every revision/expiry without growing
    past max_copy_units or refusing ordinary cache work. Metadata is allocated
    only for actually claimed fragments, including cleared/expired tombstones.
    """

    def __init__(
        self,
        config: CacheControlConfig,
        page_size: int,
        tree_core,
        clock: Callable[[], float] = time.monotonic,
    ):
        if not isinstance(config, CacheControlConfig):
            raise TypeError("kv_control_config must be CacheControlConfig")
        self.config, self.page_size, self.clock = config, page_size, clock
        self.tree_core = tree_core
        self.generation = 0
        self._incarnation = 0
        self._units = 0
        self._copies: dict[tuple[int, ComponentType], _Copy] = {}
        self._binding: tuple[str, str] | None = None
        self._binding_generation = -1
        self.ordering_revision = 0

    def bind(self, engine_id: str, engine_epoch: str):
        binding = validated(engine_id, Identifier), validated(engine_epoch, Identifier)
        if self._binding is not None:
            if self._binding_generation == self.generation:
                if binding != self._binding:
                    raise ControlError(
                        "busy",
                        "Reset the cache before changing its control owner or epoch",
                    )
            elif engine_epoch == self._binding[1]:
                raise ControlError(
                    "stale_epoch", "Cache reset requires a fresh control epoch"
                )
        self._binding, self._binding_generation = binding, self.generation

    def reset(self):
        self._copies.clear()
        self._units = 0
        self.generation += 1
        self.ordering_revision += 1

    def forget(self, node, component):
        old = self._copies.pop((node.id, component), None)
        if old is not None:
            self._units -= old.units

    def _record(self, node, component):
        record = self._copies.get((node.id, component))
        # Tensor identity is an additional guard against replacement. Physical
        # slot numbers are never used as lifetime or content identities.
        if (
            record is not None
            and record.value is not node.component_data[component].value
        ):
            self.forget(node, component)
            return None
        return record

    def split(self, parent, child):
        self.ordering_revision += 1
        for component in child.component_types:
            old = self._copies.pop((child.id, component), None)
            if old is None:
                continue
            self._units -= old.units
            for node in (parent, child):
                value = node.component_data[component].value
                if value is None:
                    continue
                units = (
                    1
                    if component == ComponentType.MAMBA
                    else len(node.key) // self.page_size
                )
                self._copies[node.id, component] = _Copy(
                    value, old.incarnation, units, dict(old.claims)
                )
                self._units += units

    def preflight(
        self, selections: list[CopySelection], action: SetPriority, caller_id: str
    ):
        identity = caller_id, action.claim_id
        grouped: dict[tuple[int, ComponentType], list[CopySelection]] = {}
        for selection in selections:
            grouped.setdefault((selection.node.id, selection.component), []).append(
                selection
            )
        extra = 0
        for (_, component), parts in grouped.items():
            record = self._record(parts[0].node, component)
            if record is not None:
                old = record.claims.get(identity)
                if old is not None:
                    if action.revision == old.revision and (
                        action.value,
                        action.ttl_seconds,
                    ) != (old.value, old.ttl):
                        raise ControlError(
                            "conflict", "Conflicting revision on a selected cache copy"
                        )
                elif len(record.claims) >= self.config.max_claims_per_copy:
                    raise ControlError(
                        "capacity", "Per-copy revision/claim capacity exhausted"
                    )
            elif component == ComponentType.MAMBA:
                extra += 1
            else:
                cursor = -1
                for part in sorted(parts, key=lambda p: p.start):
                    extra += (
                        max(0, part.end - max(cursor, part.start)) // self.page_size
                    )
                    cursor = max(cursor, part.end)
        if self._units + extra > self.config.max_copy_units:
            raise ControlError("capacity", "Copy-claim metadata capacity exhausted")

    def apply(
        self, selections: list[CopySelection], action: SetPriority, caller_id: str
    ):
        self.preflight(selections, action, caller_id)
        now = self.clock()
        identity = caller_id, action.claim_id
        claim = _Claim(
            action.revision,
            action.value,
            action.ttl_seconds,
            now + (action.ttl_seconds or 0),
        )
        changed = set()
        for selection in selections:
            node, component = selection.node, selection.component
            record = self._record(node, component)
            if record is None:
                self._incarnation += 1
                units = (
                    1
                    if component == ComponentType.MAMBA
                    else len(node.key) // self.page_size
                )
                record = _Copy(
                    node.component_data[component].value, self._incarnation, units
                )
                self._copies[node.id, component] = record
                self._units += units
            old = record.claims.get(identity)
            if old is None or action.revision > old.revision:
                record.claims[identity] = claim
                changed.add((node.id, component))
        if changed:
            self.ordering_revision += 1
        return changed

    def next_expiry(self, now):
        return min(
            (
                claim.expires_at
                for record in self._copies.values()
                for claim in record.claims.values()
                if claim.value is not None and claim.expires_at > now
            ),
            default=float("inf"),
        )

    def effective(self, node, component) -> int | None:
        record = self._record(node, component)
        if record is None:
            return None
        now = self.clock()
        return max(
            (
                claim.value
                for claim in record.claims.values()
                if claim.value is not None and claim.expires_at > now
            ),
            default=None,
        )

    def eviction_key(self, node, component):
        """Account for every component the structural eviction would reclaim."""
        core = self.tree_core
        trigger = core.components_by_type[component]
        leaf = node in core.evictable_device_leaves
        values = []
        for candidate in core.components:
            ct = candidate.component_type
            cd = node.component_data[ct]
            if cd.value is not None and candidate.eviction_priority(
                leaf
            ) <= trigger.eviction_priority(leaf):
                priority = self.effective(node, ct)
                if priority is not None:
                    values.append(priority)
        return max(values, default=0), node.last_access_time, node.id

    def capacity(self):
        return {
            "copies": len(self._copies),
            "reserved_copy_units": self._units,
            "claims": sum(len(copy.claims) for copy in self._copies.values()),
        }


class PriorityEvictionWalk:
    """One candidate heap per walk, refreshed on edits or a live TTL boundary.

    Removal of a victim does not invalidate every other key. Newly exposed
    parents are refreshed explicitly. Candidate membership is checked again at
    selection, so cascaded removals and locks cannot yield stale victims.
    """

    def __init__(self, claims, component, candidates, eligible, *, auxiliary=False):
        self.claims, self.component = claims, component
        self.candidates, self.eligible = candidates, eligible
        self.auxiliary = auxiliary
        self._revision = -1
        self._deadline = float("-inf")
        self._serial = 0
        self._last = None
        self._keys = {}
        self._heap = []

    def _push(self, node, rank):
        if not self.eligible(node):
            self._keys.pop(node.id, None)
            return
        key = self.claims.eviction_key(node, self.component)
        if self.auxiliary:
            key = key[0], rank, node.id
        self._serial += 1
        self._keys[node.id] = self._serial, rank
        heapq.heappush(self._heap, (key, self._serial, node))

    def next(self):
        now = self.claims.clock()
        if self._revision != self.claims.ordering_revision or now >= self._deadline:
            self._heap, self._keys = [], {}
            for rank, node in enumerate(self.candidates()):
                self._push(node, rank)
            self._revision = self.claims.ordering_revision
            self._deadline = self.claims.next_expiry(now)
        elif self._last is not None and self._last.parent is not None:
            parent = self._last.parent
            rank = self._keys.get(parent.id, (0, len(self._keys)))[1]
            self._push(parent, rank)
        self._last = None
        while self._heap:
            _, serial, node = heapq.heappop(self._heap)
            if self._keys.get(node.id, (None,))[0] != serial:
                continue
            self._keys.pop(node.id)
            if self.eligible(node):
                self._last = node
                return node
        return None
