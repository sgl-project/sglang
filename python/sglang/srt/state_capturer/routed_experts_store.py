"""Held routes for routed-experts forcing.

Row ``i`` holds the experts token ``i`` was routed to in the forward that
produced token ``i + 1``, so ``n`` tokens have ``n - 1`` rows.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from typing import Optional

import msgspec
import numpy as np
import torch

_TORCH_DTYPES = {np.dtype(np.uint8): torch.uint8, np.dtype(np.int16): torch.int16}


class RouteLayout(msgspec.Struct, frozen=True, kw_only=True):
    """The held layers (MoE only) and the narrowest dtype for their expert ids."""

    moe_layer_ids: tuple[int, ...]
    top_k: int
    num_logical_experts: int

    @property
    def dtype(self) -> np.dtype:
        return _route_dtype(self.num_logical_experts)

    def narrow(self, captured: np.ndarray) -> np.ndarray:
        """Captured ``[n, layers, top_k]`` ids as held ``[n, moe_layers, top_k]``."""
        if captured.ndim != 3 or captured.shape[2] != self.top_k:
            raise ValueError(f"captured {captured.shape} for a top-{self.top_k} gate")
        rows = captured[:, list(self.moe_layer_ids)]
        if rows.size and (rows.min() < 0 or rows.max() >= self.num_logical_experts):
            raise ValueError(f"expert ids outside [0, {self.num_logical_experts})")
        return rows.astype(self.dtype)


class RouteSegment(msgspec.Struct, frozen=True, kw_only=True, eq=False):
    """Read-only rows ``[start, start + n)`` of shape ``[n, moe_layers, top_k]``."""

    start: int
    routes: np.ndarray

    def __post_init__(self) -> None:
        if self.start < 0 or self.routes.ndim != 3:
            raise ValueError(f"segment at {self.start} of shape {self.routes.shape}")
        self.routes.flags.writeable = False

    @property
    def end(self) -> int:
        return self.start + len(self.routes)


class HeldRoutes(msgspec.Struct, frozen=True, kw_only=True, eq=False):
    """Rows ``[0, end)`` as a chain of segments, one per turn or retract snapshot.

    Appending never copies the rows already held, and a request keeps the chain it
    was admitted with, so a later commit or eviction of its key cannot change them.
    """

    segments: tuple[RouteSegment, ...] = ()

    def __post_init__(self) -> None:
        expected = 0
        for segment in self.segments:
            if segment.start != expected:
                raise ValueError(f"segment starts at {segment.start}, not {expected}")
            expected = segment.end

    @property
    def end(self) -> int:
        return self.segments[-1].end if self.segments else 0

    def truncated(self, end: int) -> HeldRoutes:
        """The rows ``[0, min(end, self.end))``; a cut segment is a view."""
        kept = []
        for segment in self.segments:
            if segment.start >= end:
                break
            if segment.end > end:
                routes = segment.routes[: end - segment.start]
                segment = RouteSegment(start=segment.start, routes=routes)
            kept.append(segment)
        return HeldRoutes(segments=tuple(kept))

    def extended(self, segment: RouteSegment) -> HeldRoutes:
        if len(segment.routes) == 0:
            return self
        return HeldRoutes(segments=self.segments + (segment,))

    def copy_rows(self, start: int, stop: int, out: np.ndarray) -> None:
        """Copy rows ``[start, stop)`` into ``out[: stop - start]``."""
        if not 0 <= start <= stop <= self.end:
            raise ValueError(f"rows [{start}, {stop}) are outside [0, {self.end})")
        for s in self.segments:
            lo, hi = max(start, s.start), min(stop, s.end)
            if lo < hi:
                out[lo - start : hi - start] = s.routes[lo - s.start : hi - s.start]


class Admission(msgspec.Struct, frozen=True, kw_only=True, eq=False):
    """A request's held rows and the store's generation when it was admitted."""

    rows: HeldRoutes
    generation: int


class _Entry(msgspec.Struct, frozen=True, kw_only=True, eq=False):
    tokens: np.ndarray
    rows: HeldRoutes
    generation: int


class HeldRoutesStore:
    """Held rows per session key, LRU-evicted by total tokens.

    Every TP rank runs the same admits and commits in the same order, so all ranks
    evict the same keys and force the same experts. An evicted key keeps a record
    of its last change, so a request still running for it can commit.
    """

    def __init__(self, max_tokens: int, *, max_evicted_keys: int = 1 << 16):
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}")
        self.max_tokens = max_tokens
        self._entries: OrderedDict[str, _Entry] = OrderedDict()
        self._total_tokens = 0
        # Bumped by every commit and invalidation; admissions and entries record it.
        self._generation = 0
        # Last change of recently evicted keys, oldest first, and the newest change
        # among the records dropped to bound them.
        self._evicted: OrderedDict[str, int] = OrderedDict()
        self._max_evicted_keys = max_evicted_keys
        self._forgotten_through = 0

    def admit(self, key: str, *, token_ids: Sequence[int], force_len: int) -> Admission:
        """The key's rows ``[0, n)`` for the largest ``n <= force_len`` whose tokens
        ``[0, n)`` match ``token_ids``: row ``i`` holds while tokens ``[0, i]`` do."""
        rows = HeldRoutes()
        entry = self._entries.get(key)
        if entry is not None:
            self._entries.move_to_end(key)
            limit = min(force_len, entry.rows.end, len(token_ids))
            if limit > 0:
                matched = _common_prefix_len(entry.tokens[:limit], token_ids[:limit])
                rows = entry.rows.truncated(matched)
        return Admission(rows=rows, generation=self._generation)

    def commit(
        self,
        key: str,
        *,
        token_ids: Sequence[int],
        rows: HeldRoutes,
        admission: Admission,
    ) -> bool:
        """Make ``rows`` the key's rows for ``token_ids``; return whether it applied.

        Apply only if the key is unchanged since ``admission``; otherwise drop its
        rows so the next turn is refused. A late commit may come from a retried
        turn's abandoned attempt, which the store cannot tell from the kept one.
        """
        if rows.end != len(token_ids) - 1:
            raise ValueError(f"{rows.end} rows for {len(token_ids)} tokens")
        if self._last_change(key) > admission.generation:
            self.invalidate(key)
            return False
        self._discard(key)
        self._generation += 1
        if rows.end > self.max_tokens:
            # Keeping it would first evict every other key, then itself.
            self._remember_evicted(key, self._generation)
            return True
        self._entries[key] = _Entry(
            tokens=np.array(token_ids, dtype=np.int64),
            rows=_owning(rows),
            generation=self._generation,
        )
        self._total_tokens += rows.end
        while self._total_tokens > self.max_tokens:
            evicted_key, evicted = self._entries.popitem(last=False)
            self._total_tokens -= evicted.rows.end
            self._remember_evicted(evicted_key, evicted.generation)
        return True

    def invalidate(self, key: str) -> None:
        """Drop the key's rows; no request admitted before this can commit."""
        self._discard(key)
        self._generation += 1
        self._remember_evicted(key, self._generation)

    def _last_change(self, key: str) -> int:
        if (entry := self._entries.get(key)) is not None:
            return entry.generation
        if (generation := self._evicted.get(key)) is not None:
            return generation
        # A key with no record either never committed or had its record
        # dropped, which may have been newer than any admission since.
        return self._forgotten_through

    def _discard(self, key: str) -> None:
        if (entry := self._entries.pop(key, None)) is not None:
            self._total_tokens -= entry.rows.end
        self._evicted.pop(key, None)

    def _remember_evicted(self, key: str, generation: int) -> None:
        self._evicted[key] = generation
        while len(self._evicted) > self._max_evicted_keys:
            _, dropped = self._evicted.popitem(last=False)
            self._forgotten_through = max(self._forgotten_through, dropped)


class ForcingBatch(msgspec.Struct, frozen=True, kw_only=True, eq=False):
    """An extend forward's forced rows: ``mask[t]`` forces token row ``t``.

    ``routes`` is ``[tokens, moe_layers, top_k]`` logical expert ids, zero where the
    mask is off, in the held dtype on the host and int32 on the device.
    """

    mask: torch.Tensor
    routes: torch.Tensor

    def padded_to(self, num_tokens: int) -> ForcingBatch:
        """The batch with unforced rows appended (a mixed batch's decode rows)."""
        extra = num_tokens - self.mask.shape[0]
        if extra < 0:
            raise ValueError(f"cannot pad {self.mask.shape[0]} rows to {num_tokens}")
        if extra == 0:
            return self
        pad = self.routes.new_zeros((extra, *self.routes.shape[1:]))
        return ForcingBatch(
            mask=torch.cat([self.mask, self.mask.new_zeros(extra)]),
            routes=torch.cat([self.routes, pad]),
        )

    def to_device(self, device) -> ForcingBatch:
        return ForcingBatch(
            mask=self.mask.to(device, non_blocking=True),
            routes=self.routes.to(device, non_blocking=True).to(torch.int32),
        )


def unforced_recompute_tokens(
    *, prefix_len: int, extend_len: int, held_end: int, force_len: int
) -> int:
    """Positions a forward recomputes that the client holds rows for but the engine
    does not; admission refuses every request that would have one."""
    return max(0, min(prefix_len + extend_len, force_len) - max(prefix_len, held_end))


def build_forcing_batch(
    *,
    held: Sequence[HeldRoutes],
    prefix_lens: Sequence[int],
    extend_lens: Sequence[int],
    layout: RouteLayout,
    pin_memory: bool = False,
) -> Optional[ForcingBatch]:
    """The forced rows of an extend forward, or None if it recomputes no held row.

    Request ``i`` computes positions from ``prefix_lens[i]`` at forward rows from
    ``sum(extend_lens[:i])``; each one below its held end is forced.
    """
    num_tokens = sum(extend_lens)
    mask = routes = None
    offset = 0
    for held_routes, prefix_len, extend_len in zip(
        held, prefix_lens, extend_lens, strict=True
    ):
        stop = min(prefix_len + extend_len, held_routes.end)
        if stop > prefix_len:
            if mask is None:
                mask = torch.zeros(num_tokens, dtype=torch.bool, pin_memory=pin_memory)
                routes = torch.zeros(
                    (num_tokens, len(layout.moe_layer_ids), layout.top_k),
                    dtype=_TORCH_DTYPES[layout.dtype],
                    pin_memory=pin_memory,
                )
            n = stop - prefix_len
            mask[offset : offset + n] = True
            held_routes.copy_rows(
                prefix_len, stop, out=routes.numpy()[offset : offset + n]
            )
        offset += extend_len
    return None if mask is None else ForcingBatch(mask=mask, routes=routes)


def _route_dtype(num_logical_experts: int) -> np.dtype:
    """The narrowest dtype that holds expert ids ``[0, num_logical_experts)``."""
    if 0 < num_logical_experts <= 256:
        return np.dtype(np.uint8)
    if 0 < num_logical_experts <= np.iinfo(np.int16).max + 1:
        return np.dtype(np.int16)
    raise ValueError(f"no expert id dtype for {num_logical_experts} experts")


def _owning(rows: HeldRoutes) -> HeldRoutes:
    """Copy view segments so the store never pins a larger parent array."""
    segments = []
    for s in rows.segments:
        owned = s.routes.base is None
        segments.append(
            s if owned else RouteSegment(start=s.start, routes=s.routes.copy())
        )
    return HeldRoutes(segments=tuple(segments))


def _common_prefix_len(stored: np.ndarray, token_ids: Sequence[int]) -> int:
    mismatch = np.flatnonzero(stored != np.asarray(token_ids, dtype=np.int64))
    return int(mismatch[0]) if len(mismatch) else len(stored)
