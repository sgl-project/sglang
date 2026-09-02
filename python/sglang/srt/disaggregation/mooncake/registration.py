"""Process-local ownership for Mooncake memory registrations.

PD runtime role switching constructs both Prefill and Decode managers over the
same in-place KV tensors.  Registering and deregistering the same CUDA address
once per manager is unsafe: the first manager can remove the RDMA memory region
while another manager still advertises its rkey.  This registry makes the
transfer-engine registration shared and lets hot reconfiguration pin an
in-place allocation for the remaining lifetime of the worker process.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Dict, Iterable, Set, Tuple


BufferKey = Tuple[int, int]


@dataclass
class _Registration:
    length: int
    owners: Set[int] = field(default_factory=set)
    process_pinned: bool = False


class SharedBufferRegistrationRegistry:
    """Reference-count Mooncake registrations shared by manager wrappers."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: Dict[BufferKey, _Registration] = {}

    @staticmethod
    def _normalize_pairs(pairs: Iterable[Tuple[int, int]]) -> Dict[int, int]:
        normalized: Dict[int, int] = {}
        for raw_ptr, raw_length in pairs:
            ptr = int(raw_ptr)
            length = int(raw_length)
            if ptr <= 0 or length <= 0:
                continue
            previous = normalized.get(ptr)
            if previous is not None and previous != length:
                raise RuntimeError(
                    "Mooncake buffer address has conflicting lengths: "
                    f"ptr={ptr:#x} old={previous} new={length}"
                )
            normalized[ptr] = length
        return normalized

    @staticmethod
    def _coalesce_overlapping_ranges(ranges: Dict[int, int]) -> Dict[int, int]:
        """Merge overlapping views before they acquire registry ownership.

        Hybrid Mamba cache profiles can expose one in-place allocation as
        several tensor views.  A target-role view may bridge two source-role
        registrations.  Processing those views one at a time temporarily adds
        the new manager as an owner of the first reused range, which makes the
        later bridging view look like an overlap with an active manager.  Plan
        the union first so the old ownerless pinned ranges can be replaced
        atomically from the registry's perspective.

        Adjacent ranges remain separate because they may be distinct CUDA
        allocations even when their virtual addresses happen to touch.
        """

        coalesced = []
        for ptr, length in sorted(ranges.items()):
            end = ptr + length
            if coalesced and ptr < coalesced[-1][1]:
                coalesced[-1] = (
                    coalesced[-1][0],
                    max(coalesced[-1][1], end),
                )
            else:
                coalesced.append((ptr, end))
        return {ptr: end - ptr for ptr, end in coalesced}

    def register(self, engine, owner: object, pairs: Iterable[Tuple[int, int]]) -> Set[BufferKey]:
        owner_id = id(owner)
        engine_id = id(engine)
        normalized = self._coalesce_overlapping_ranges(
            self._normalize_pairs(pairs)
        )
        keys: Set[BufferKey] = set()

        with self._lock:
            # A runtime D/P cache morph can expose the same allocation through
            # different tensor views.  Mooncake rejects registering a second
            # range that overlaps an existing MR, even when the underlying
            # allocation is unchanged.
            #
            # Plan every stale replacement before acquiring any ownership for
            # this manager.  CPU/CUDA allocators may repartition one old large
            # buffer into several new, disjoint views.  If the first small view
            # acquired the covering old MR immediately, a later view that only
            # partially overlaps that MR would incorrectly see this same
            # manager as an active conflict.  The whole registration call is
            # one cache-plane transaction, so first remove every ownerless,
            # process-pinned MR that cannot cover one of the requested views.
            replace_keys = set()
            for ptr, length in normalized.items():
                end = ptr + length
                for key, entry in self._entries.items():
                    if key[0] != engine_id:
                        continue
                    entry_ptr = key[1]
                    entry_end = entry_ptr + entry.length
                    if not (ptr < entry_end and entry_ptr < end):
                        continue
                    if entry_ptr <= ptr and end <= entry_end:
                        continue
                    if entry.owners or not entry.process_pinned:
                        raise RuntimeError(
                            "Mooncake buffer overlaps an active registration: "
                            f"requested=[{ptr:#x}, {end:#x}) "
                            f"existing=[{entry_ptr:#x}, {entry_end:#x})"
                        )
                    replace_keys.add(key)

            if replace_keys:
                stale_ptrs = sorted(key[1] for key in replace_keys)
                engine.batch_deregister(stale_ptrs)
                for key in replace_keys:
                    self._entries.pop(key, None)

            # After the plan is committed, every requested view either has a
            # covering MR or is disjoint from all live MRs.  New requested
            # views were coalesced above, so registering one cannot create a
            # partial overlap with another view in this transaction.
            for ptr, length in normalized.items():
                end = ptr + length
                covering = []
                overlaps = []
                for key, entry in self._entries.items():
                    if key[0] != engine_id:
                        continue
                    entry_ptr = key[1]
                    entry_end = entry_ptr + entry.length
                    if ptr < entry_end and entry_ptr < end:
                        overlaps.append((key, entry))
                        if entry_ptr <= ptr and end <= entry_end:
                            covering.append((key, entry))

                if covering:
                    # Prefer the tightest covering interval so a broad pinned
                    # allocation does not hide a more precise active owner.
                    key, entry = min(covering, key=lambda item: item[1].length)
                    entry.owners.add(owner_id)
                    keys.add(key)
                    continue

                if overlaps:
                    ranges = ", ".join(
                        f"[{key[1]:#x}, {key[1] + entry.length:#x})"
                        for key, entry in overlaps
                    )
                    raise RuntimeError(
                        "Mooncake registration planning left an overlapping range: "
                        f"requested=[{ptr:#x}, {end:#x}) existing={ranges}"
                    )

                engine.batch_register([ptr], [length])
                key = (engine_id, ptr)
                self._entries[key] = _Registration(
                    length=length, owners={owner_id}
                )
                keys.add(key)
        return keys

    def release(
        self,
        engine,
        owner: object,
        keys: Iterable[BufferKey],
        *,
        preserve_for_process: bool = False,
    ) -> None:
        owner_id = id(owner)
        engine_id = id(engine)
        keys = set(keys)
        with self._lock:
            deregister_ptrs = []
            removable_keys = []
            for key in keys:
                if key[0] != engine_id:
                    continue
                entry = self._entries.get(key)
                if entry is None:
                    continue
                entry.owners.discard(owner_id)
                if preserve_for_process:
                    entry.process_pinned = True
                if not entry.owners and not entry.process_pinned:
                    deregister_ptrs.append(key[1])
                    removable_keys.append(key)

            if deregister_ptrs:
                engine.batch_deregister(deregister_ptrs)
                for key in removable_keys:
                    self._entries.pop(key, None)


SHARED_MOONCAKE_BUFFER_REGISTRY = SharedBufferRegistrationRegistry()
