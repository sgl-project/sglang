# SPDX-License-Identifier: Apache-2.0
"""Owner-thread L2 eviction exclusion, independent of operation lifetimes."""

from __future__ import annotations

import math
import threading
import time
from array import array
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

import msgspec

from sglang.srt.managers.kv_control.contracts import ControlError
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType

if TYPE_CHECKING:
    from sglang.srt.mem_cache.base_prefix_cache import DecLockRefParams
    from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeNode
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

_MAX_INTEGER = (1 << 53) - 1
_FULL = ComponentType.FULL
ProtectionState = Literal["active", "expired", "released", "invalidated"]


def _identifier(value: str, name: str) -> None:
    if type(value) is not str or not 1 <= len(value) <= 256:
        raise ControlError("invalid", f"{name} must be a bounded nonempty identifier")


def _revision(value: int) -> None:
    if type(value) is not int or not 0 <= value <= _MAX_INTEGER:
        raise ControlError("invalid", "Revision/incarnation must be a bounded integer")


class ProtectionKey(msgspec.Struct, frozen=True, kw_only=True):
    caller_id: str
    task_id: str
    task_incarnation: int
    claim_id: str

    def __post_init__(self) -> None:
        _identifier(self.caller_id, "caller_id")
        _identifier(self.task_id, "task_id")
        _identifier(self.claim_id, "claim_id")
        _revision(self.task_incarnation)


class HostPrefix(msgspec.Struct, frozen=True, kw_only=True):
    """A whole FULL token prefix in this cache's physical compatibility domain.

    ``cache_salt`` is the already resolved physical salt: callers apply the
    executor's compatibility/namespace transform exactly once before creating
    this value. It is not a raw application salt. No token material is inferred
    and resolving a prefix never inserts or splits a radix node.
    """

    token_ids: tuple[int, ...]
    cache_salt: str | None = None


class ProtectionSnapshot(msgspec.Struct, frozen=True, kw_only=True):
    key: ProtectionKey
    revision: int
    state: ProtectionState
    prefixes: tuple[HostPrefix, ...]
    physical_tokens: int
    activated_at: float | None
    expires_at: float | None
    engine_id: str
    engine_epoch: str
    namespace: str
    condition: Literal["eviction_exclusion"] = "eviction_exclusion"
    reason: str = ""


class _Record(msgspec.Struct, kw_only=True):
    snapshot: ProtectionSnapshot
    fingerprint: tuple
    slots: frozenset[int] = frozenset()
    holds: tuple[tuple[int, DecLockRefParams], ...] = ()


class ProtectionRegistry:
    """One registry per Engine/namespace/cache generation on its cache owner.

    Only existing, complete FULL host prefixes at materialized radix boundaries
    are supported. Native UUID-bounded host-lock receipts survive radix splits;
    an operation completing has no effect on these independently owned locks.

    Active claims and terminal revision records have separate finite bounds.
    Terminal records are never evicted: exhaustion rejects new claim keys.
    Released/expired keys cannot be renewed or reacquired; use a new claim ID.
    Retiring this registry and advancing the control epoch is the reclamation
    boundary. ``invalidate`` closes it, and cache reset also closes it lazily
    on the next owner access. The service must publish the returned changes and
    call ``expire`` while idle; it must invalidate before an explicit flush.
    """

    def __init__(
        self,
        cache: UnifiedRadixCache,
        *,
        engine_id: str,
        engine_epoch: str,
        namespace: str,
        max_claims: int = 1024,
        max_records: int = 4096,
        max_protected_tokens: int = 65536,
        max_scope_tokens: int = 65536,
        max_scope_prefixes: int = 256,
        max_ttl_seconds: float = 86400.0 * 365,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        for name, value in (
            ("engine_id", engine_id),
            ("engine_epoch", engine_epoch),
            ("namespace", namespace),
        ):
            _identifier(value, name)
        for value in (
            max_claims,
            max_records,
            max_protected_tokens,
            max_scope_tokens,
            max_scope_prefixes,
        ):
            if type(value) is not int or not 1 <= value <= 1 << 24:
                raise ControlError(
                    "invalid", "Protection budgets must be bounded positive integers"
                )
        if max_claims > max_records:
            raise ControlError("invalid", "Claim budget exceeds revision record budget")
        if (
            type(max_ttl_seconds) not in (int, float)
            or not math.isfinite(max_ttl_seconds)
            or max_ttl_seconds <= 0
        ):
            raise ControlError("invalid", "Maximum TTL must be finite and positive")
        self._thread = threading.get_ident()
        if cache._kv_control_owner_thread != self._thread:
            raise ControlError(
                "wrong_owner", "Bind protection on the cache control owner"
            )
        self._cache = cache
        self._core = cache.tree_core
        self._root = self._core.root_node
        self._supported()
        state = self._core.control_state
        if state is not None and state.config.namespace != namespace:
            raise ControlError("incompatible_namespace", "Cache namespace differs")
        self.engine_id, self.engine_epoch, self.namespace = (
            engine_id,
            engine_epoch,
            namespace,
        )
        self.max_claims, self.max_records = max_claims, max_records
        self.max_protected_tokens = max_protected_tokens
        self.max_scope_tokens, self.max_scope_prefixes = (
            max_scope_tokens,
            max_scope_prefixes,
        )
        self.max_ttl_seconds = float(max_ttl_seconds)
        self._clock = clock
        self._last_now = -math.inf
        self._records: dict[ProtectionKey, _Record] = {}
        self._physical_refs: dict[int, int] = {}
        self._pending_changes: list[ProtectionSnapshot] = []
        self._active_count = 0
        self._invalidated = ""

    def _supported(self) -> None:
        from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore

        if (
            not isinstance(self._core, UnifiedTreeCore)
            or not self._core.enable_hicache
            or self._core.component_types != (_FULL,)
            or self._cache.host_memory_mode != "cache"
            or self._cache.buffer_pipeline is not None
            or self._cache.sidecar_pool_specs
            or self._cache.disable
            or self._core.enable_external_cache_linker
            or self._cache._kv_control_unsupported_reason
        ):
            raise ControlError(
                "unsupported",
                "Protection requires the native FULL-only persistent host cache",
            )

    def _check(self) -> None:
        if threading.get_ident() != self._thread:
            raise ControlError(
                "wrong_owner", "Protection must run on its cache owner thread"
            )
        if not self._invalidated and (
            self._cache.tree_core is not self._core
            or self._core.root_node is not self._root
        ):
            # Reset destroyed the old tree/allocator ownership. Replaying old
            # receipts against recycled node IDs would release unrelated locks.
            self._pending_changes.extend(
                self._invalidate("cache_reset", release_holds=False)
            )

    def _now(self) -> float:
        now = self._clock()
        if (
            type(now) not in (int, float)
            or not math.isfinite(now)
            or now < self._last_now
        ):
            raise ControlError(
                "invalid_clock", "Protection requires a finite monotonic clock"
            )
        self._last_now = now
        return float(now)

    def _ttl(self, ttl_seconds: float, now: float) -> float:
        if (
            type(ttl_seconds) not in (int, float)
            or not math.isfinite(ttl_seconds)
            or not 0 < ttl_seconds <= self.max_ttl_seconds
            or not math.isfinite(now + ttl_seconds)
        ):
            raise ControlError(
                "invalid",
                "TTL must be finite, positive and within the protection limit",
            )
        return float(ttl_seconds)

    def _record(self, key: ProtectionKey, now: float) -> _Record | None:
        if not isinstance(key, ProtectionKey):
            raise ControlError("invalid", "Protection requires a typed claim key")
        record = self._records.get(key)
        if (
            record is not None
            and record.snapshot.state == "active"
            and record.snapshot.expires_at <= now
        ):
            self._retire(record, state="expired", reason="ttl_expired")
            self._pending_changes.append(record.snapshot)
        return record

    def _replay(
        self, record: _Record | None, revision: int, fingerprint: tuple
    ) -> ProtectionSnapshot | None:
        if record is None:
            return None
        if revision < record.snapshot.revision:
            raise ControlError(
                "stale_revision",
                "Protection revision is older than the recorded revision",
            )
        if revision == record.snapshot.revision:
            if fingerprint != record.fingerprint:
                raise ControlError(
                    "conflict", "Protection revision was used by a different request"
                )
            return record.snapshot
        return None

    def _resolve(
        self, prefixes: tuple[HostPrefix, ...]
    ) -> tuple[tuple[UnifiedTreeNode, ...], frozenset[int]]:
        from sglang.srt.mem_cache.radix_cache import RadixKey

        if (
            type(prefixes) is not tuple
            or not 1 <= len(prefixes) <= self.max_scope_prefixes
        ):
            raise ControlError(
                "invalid", "Protection requires a bounded tuple of whole prefixes"
            )
        if any(
            not isinstance(p, HostPrefix) or type(p.token_ids) is not tuple
            for p in prefixes
        ):
            raise ControlError(
                "invalid", "Protection prefixes must contain immutable token tuples"
            )
        if sum(len(p.token_ids) for p in prefixes) > self.max_scope_tokens:
            raise ControlError(
                "capacity", "Protection scope exceeds its traversal budget"
            )
        nodes, slots = {}, set()
        for prefix in prefixes:
            tokens = prefix.token_ids
            if not tokens or len(tokens) % self._core.page_size:
                raise ControlError(
                    "unsupported",
                    "Protection requires a nonempty page-aligned whole prefix",
                )
            if any(type(t) is not int or not 0 <= t <= _MAX_INTEGER for t in tokens):
                raise ControlError(
                    "invalid", "Prefix tokens must be bounded nonnegative integers"
                )
            if prefix.cache_salt is not None:
                _identifier(prefix.cache_salt, "cache_salt")
            key = RadixKey(array("q", tokens), cache_salt=prefix.cache_salt)
            node, offset = self._root, 0
            while offset < len(tokens):
                node = node.children.get(key.child_key_at(offset, self._core.page_size))
                if node is None:
                    raise ControlError("missing", "Protection prefix is not resident")
                if node.key.is_bigram:
                    raise ControlError(
                        "unsupported", "Bigram host prefixes are unsupported"
                    )
                matched = node.key.match_at(key, offset, self._core.page_size)
                if matched != len(node.key):
                    code = (
                        "unsupported" if offset + matched == len(tokens) else "missing"
                    )
                    raise ControlError(
                        code, "Protection requires an existing complete radix boundary"
                    )
                value = node.component_data[_FULL].host_value
                if value is None:
                    raise ControlError(
                        "missing", "Complete prefix dependencies must be host resident"
                    )
                if node.write_through_pending_id is not None:
                    raise ControlError(
                        "incomplete",
                        "Host prefix backup has not reached its publication fence",
                    )
                if value.numel() != len(node.key):
                    raise ControlError(
                        "unsupported",
                        "Host prefix has an unsupported physical span layout",
                    )
                nodes[node.id] = node
                slots.update(value.tolist())
                offset += matched
        return tuple(nodes.values()), frozenset(slots)

    def _hold(
        self, nodes: tuple[UnifiedTreeNode, ...]
    ) -> tuple[tuple[int, DecLockRefParams], ...]:
        holds = []
        try:
            for node in nodes:
                receipt = self._cache.inc_host_lock_ref(node.id).to_dec_params()
                holds.append((node.id, receipt))
        except Exception:
            self._release(tuple(holds))
            raise
        return tuple(holds)

    def _release(self, holds: tuple[tuple[int, DecLockRefParams], ...]) -> None:
        for node_id, receipt in reversed(holds):
            self._cache.dec_host_lock_ref(node_id, receipt)

    def acquire(
        self,
        *,
        key: ProtectionKey,
        revision: int,
        prefixes: tuple[HostPrefix, ...],
        ttl_seconds: float,
    ) -> ProtectionSnapshot:
        self._check()
        _revision(revision)
        now = self._now()
        ttl = self._ttl(ttl_seconds, now)
        record = self._record(key, now)
        fingerprint = ("acquire", prefixes, ttl)
        replay = self._replay(record, revision, fingerprint)
        if replay is not None:
            return replay
        self._admit_key(record)
        nodes, slots = self._resolve(prefixes)
        old_slots = record.slots if record is not None else frozenset()
        added = slots.difference(self._physical_refs)
        removed = sum(self._physical_refs[s] == 1 for s in old_slots - slots)
        if len(self._physical_refs) + len(added) - removed > self.max_protected_tokens:
            raise ControlError("capacity", "L2 protection token capacity exhausted")
        holds = self._hold(nodes)
        try:
            # The deadline starts only after the native holds are installed.
            now = self._now()
            self._ttl(ttl, now)
        except Exception:
            self._release(holds)
            raise
        # Acquire the replacement first: an update never opens an eviction gap.
        if record is not None:
            self._drop_holds(record)
        snapshot = ProtectionSnapshot(
            key=key,
            revision=revision,
            state="active",
            prefixes=prefixes,
            physical_tokens=len(slots),
            activated_at=now,
            expires_at=now + ttl,
            engine_id=self.engine_id,
            engine_epoch=self.engine_epoch,
            namespace=self.namespace,
        )
        self._records[key] = _Record(
            snapshot=snapshot, fingerprint=fingerprint, slots=slots, holds=holds
        )
        for slot in slots:
            self._physical_refs[slot] = self._physical_refs.get(slot, 0) + 1
        self._active_count += 1
        return snapshot

    def _admit_key(self, record: _Record | None) -> None:
        if self._invalidated:
            raise ControlError(
                "invalidated", "Protection registry requires a new control epoch"
            )
        self._supported()
        if record is not None and record.snapshot.state != "active":
            raise ControlError(
                record.snapshot.state,
                "A terminal protection key requires a new claim ID",
            )
        if record is None and (
            len(self._records) >= self.max_records
            or self._active_count >= self.max_claims
        ):
            raise ControlError(
                "capacity", "L2 protection claim/revision capacity exhausted"
            )

    def renew(
        self, *, key: ProtectionKey, revision: int, ttl_seconds: float
    ) -> ProtectionSnapshot:
        self._check()
        _revision(revision)
        now = self._now()
        ttl = self._ttl(ttl_seconds, now)
        record = self._record(key, now)
        fingerprint = ("renew", ttl)
        replay = self._replay(record, revision, fingerprint)
        if replay is not None:
            return replay
        if record is None:
            raise ControlError(
                "missing", "Renew requires an existing active protection claim"
            )
        self._admit_key(record)
        record.snapshot = msgspec.structs.replace(
            record.snapshot, revision=revision, expires_at=now + ttl
        )
        record.fingerprint = fingerprint
        return record.snapshot

    def release(self, *, key: ProtectionKey, revision: int) -> ProtectionSnapshot:
        self._check()
        _revision(revision)
        now = self._now()
        record = self._record(key, now)
        fingerprint = ("release",)
        replay = self._replay(record, revision, fingerprint)
        if replay is not None:
            return replay
        if record is None:
            if len(self._records) >= self.max_records:
                raise ControlError("capacity", "Protection revision capacity exhausted")
            record = _Record(
                snapshot=ProtectionSnapshot(
                    key=key,
                    revision=revision,
                    state="released",
                    prefixes=(),
                    physical_tokens=0,
                    activated_at=None,
                    expires_at=None,
                    engine_id=self.engine_id,
                    engine_epoch=self.engine_epoch,
                    namespace=self.namespace,
                ),
                fingerprint=fingerprint,
            )
            self._records[key] = record
        else:
            self._retire(record, state="released", reason="explicit_release")
            record.snapshot = msgspec.structs.replace(
                record.snapshot, revision=revision
            )
            record.fingerprint = fingerprint
        return record.snapshot

    def _drop_holds(self, record: _Record, *, release_holds: bool = True) -> None:
        if release_holds:
            self._release(record.holds)
        for slot in record.slots:
            count = self._physical_refs[slot] - 1
            if count:
                self._physical_refs[slot] = count
            else:
                del self._physical_refs[slot]
        record.holds, record.slots = (), frozenset()
        self._active_count -= 1

    def _retire(
        self,
        record: _Record,
        *,
        state: ProtectionState,
        reason: str,
        release_holds: bool = True,
    ) -> None:
        if record.snapshot.state == "active":
            self._drop_holds(record, release_holds=release_holds)
            record.snapshot = msgspec.structs.replace(
                record.snapshot, state=state, reason=reason
            )

    def get(self, key: ProtectionKey) -> ProtectionSnapshot | None:
        self._check()
        record = self._record(key, self._now())
        return record.snapshot if record is not None else None

    def expire(self, *, limit: int = 64) -> tuple[ProtectionSnapshot, ...]:
        """Retire a bounded batch and return unreported expiry/reset changes.

        Lazy expiry in ``get``/``renew`` and reset detection also enqueue their
        terminal change here. Each key contributes at most one such change,
        keeping publication work within the revision record bound.
        """
        self._check()
        if type(limit) is not int or not 1 <= limit <= 1 << 24:
            raise ControlError(
                "invalid", "Expiry batch must be a bounded positive integer"
            )
        now = self._now()
        expired = self._pending_changes[:limit]
        del self._pending_changes[:limit]
        if len(expired) == limit:
            return tuple(expired)
        for record in self._records.values():
            if record.snapshot.state == "active" and record.snapshot.expires_at <= now:
                self._retire(record, state="expired", reason="ttl_expired")
                expired.append(record.snapshot)
                if len(expired) == limit:
                    break
        return tuple(expired)

    def cleanup_task(
        self, *, caller_id: str, task_id: str, task_incarnation: int
    ) -> tuple[ProtectionSnapshot, ...]:
        self._check()
        _identifier(caller_id, "caller_id")
        _identifier(task_id, "task_id")
        _revision(task_incarnation)
        retired = []
        for key, record in self._records.items():
            if (key.caller_id, key.task_id, key.task_incarnation) == (
                caller_id,
                task_id,
                task_incarnation,
            ) and record.snapshot.state == "active":
                self._retire(record, state="released", reason="task_finished")
                retired.append(record.snapshot)
        return tuple(retired)

    def _invalidate(
        self, reason: str, *, release_holds: bool
    ) -> tuple[ProtectionSnapshot, ...]:
        changes = []
        for record in self._records.values():
            if record.snapshot.state == "active":
                self._retire(
                    record,
                    state="invalidated",
                    reason=reason,
                    release_holds=release_holds,
                )
                changes.append(record.snapshot)
        self._invalidated = reason
        self._root = None
        return tuple(changes)

    def invalidate(
        self, *, reason: str = "cache_cleared"
    ) -> tuple[ProtectionSnapshot, ...]:
        self._check()
        _identifier(reason, "reason")
        pending = tuple(self._pending_changes)
        self._pending_changes.clear()
        return pending + self._invalidate(reason, release_holds=not self._invalidated)

    @property
    def next_deadline(self) -> float | None:
        self._check()
        if self._pending_changes:
            return self._now()
        return min(
            (
                r.snapshot.expires_at
                for r in self._records.values()
                if r.snapshot.state == "active"
            ),
            default=None,
        )

    @property
    def has_background_work(self) -> bool:
        self._check()
        return self._active_count > 0 or bool(self._pending_changes)

    @property
    def protected_tokens(self) -> int:
        self._check()
        return len(self._physical_refs)

    @property
    def active_claims(self) -> int:
        self._check()
        return self._active_count
