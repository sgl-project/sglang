"""Debug check that HiCache load-backs restore exactly the bytes that were backed up.

Enabled with SGLANG_DEBUG_HICACHE_VERIFY=1. The L2 transfer engine waits for
every device-to-host backup and records a 128-bit hash of each backed-up
device row under its host slot. After every host-to-device load it hashes the
loaded device rows and compares them with the hashes recorded for their host
slots. Any changed byte changes the hash.

Rows are read from MHA (K/V), MLA (KV) and Mamba (conv/temporal) device pools.
Rows of other pools, and host slots with no recorded backup (for example
packed draft-model layers, which have no backup transfer of their own), are
counted as unverified. Every check synchronizes the transfer stream, so this
is for debugging only. Host slots filled from a storage backend, or by a
prefill peer under PD decode host receive, never pass a device-to-host backup,
so the check is turned off when either is in use.
"""

from __future__ import annotations

import functools
import logging
import random
import time
from collections import Counter
from typing import Callable, Iterator, Optional

import torch

from sglang.srt.mem_cache.memory_pool import MambaPool

logger = logging.getLogger(__name__)

_STATS_INTERVAL_S = 60.0
# int64 words of each row hashed per step; bounds the temporaries to
# num_rows * 2 * _CHUNK_WORDS * 8 bytes, whatever the row size.
_CHUNK_WORDS = 1 << 16


def _row_parts(pool, rows: torch.Tensor) -> Optional[Iterator[torch.Tensor]]:
    """The device rows' state as [n, ...] tensors, one layer at a time, or
    None for an unsupported pool."""
    if isinstance(pool, MambaPool):
        cache = pool.mamba_cache
        return (
            state[layer, rows]
            for state in (*cache.conv, cache.temporal)
            for layer in range(state.shape[0])
        )
    # MHATokenToKVPool and its subclasses: one K and one V buffer per layer
    k_buffer = getattr(pool, "k_buffer", None)
    if k_buffer is not None:
        v_buffer = getattr(pool, "v_buffer", None)
        buffers = [*k_buffer, *(v_buffer if v_buffer is not None else ())]
        return (buf[rows] for buf in buffers)
    # MLATokenToKVPool and its subclasses: one KV buffer per layer
    kv_buffer = getattr(pool, "kv_buffer", None)
    if kv_buffer is not None:
        return (buf[rows] for buf in kv_buffer)
    # HybridLinearKVPool and SWAKVPool: rows of the wrapped full-attention pool
    full_kv_pool = getattr(pool, "full_kv_pool", None)
    if full_kv_pool is not None:
        return _row_parts(full_kv_pool, rows)
    return None


@functools.lru_cache(maxsize=8)
def _coeffs(device: torch.device) -> torch.Tensor:
    """[2, _CHUNK_WORDS] fixed odd int64 multipliers, one row per hash lane."""
    gen = torch.Generator(device="cpu").manual_seed(0)
    coeffs = torch.randint(
        -(1 << 62), 1 << 62, (2, _CHUNK_WORDS), generator=gen, dtype=torch.int64
    )
    return (coeffs | 1).to(device)


@functools.lru_cache(maxsize=4096)
def _chunk_multipliers(part: int, chunk: int) -> tuple[int, int]:
    """Two odd multipliers that tie a chunk's sums to its part and position."""
    rng = random.Random(part * 1_000_003 + chunk)
    return rng.getrandbits(62) | 1, rng.getrandbits(62) | 1


def _row_hashes(
    parts: Iterator[torch.Tensor], num_rows: int, device: torch.device
) -> torch.Tensor:
    """[n, 2] int64: two polynomial hashes (wrapping) over each row's bytes."""
    hashes = torch.zeros(num_rows, 2, dtype=torch.int64, device=device)
    coeffs = _coeffs(device)
    for index, part in enumerate(parts):
        row_bytes = part.contiguous().view(torch.uint8).reshape(num_rows, -1)
        pad = (-row_bytes.shape[1]) % 8
        if pad:
            row_bytes = torch.nn.functional.pad(row_bytes, (0, pad))
        words = row_bytes.view(torch.int64)
        for chunk, start in enumerate(range(0, words.shape[1], _CHUNK_WORDS)):
            block = words[:, start : start + _CHUNK_WORDS]
            sums = (block.unsqueeze(1) * coeffs[:, : block.shape[1]]).sum(dim=2)
            for lane, multiplier in enumerate(_chunk_multipliers(index, chunk)):
                hashes[:, lane] += sums[:, lane] * multiplier
    return hashes.cpu()


class _HashTable:
    """Hashes of backed-up rows, indexed by host slot; grows on demand."""

    def __init__(self):
        self.hashes = torch.zeros(0, 2, dtype=torch.int64)
        self.valid = torch.zeros(0, dtype=torch.bool)

    def put(self, slots: torch.Tensor, hashes: torch.Tensor) -> None:
        size = int(slots.max()) + 1
        if size > self.valid.numel():
            size = max(size, 2 * self.valid.numel())
            grown = torch.zeros(size, 2, dtype=torch.int64)
            grown[: self.hashes.shape[0]] = self.hashes
            valid = torch.zeros(size, dtype=torch.bool)
            valid[: self.valid.numel()] = self.valid
            self.hashes, self.valid = grown, valid
        self.hashes[slots] = hashes
        self.valid[slots] = True

    def get(self, slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(known, hashes): which slots have a record, and their hashes."""
        known = torch.zeros(slots.numel(), dtype=torch.bool)
        hashes = torch.zeros(slots.numel(), 2, dtype=torch.int64)
        in_range = slots < self.valid.numel()
        known[in_range] = self.valid[slots[in_range]]
        hashes[known] = self.hashes[slots[known]]
        return known, hashes


class HiCacheVerifier:
    """Records backed-up rows and checks loaded ones.

    Callers pass the transfers after they finished. `resolve_rows` maps a
    transfer to the device rows its pool indexes.
    """

    def __init__(self, resolve_rows: Callable[..., torch.Tensor]):
        self._resolve_rows = resolve_rows
        self._tables: dict[tuple[int, int], _HashTable] = {}
        self._unsupported: set[str] = set()
        self._last_stats = time.monotonic()
        self.counts: Counter[str] = Counter()

    def _hashes(self, transfer) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """(host slots, row hashes), or None for an unsupported pool."""
        pool = transfer.device_pool
        rows = self._resolve_rows(transfer)
        rows = rows.to(device=_pool_device(pool), dtype=torch.int64)
        parts = _row_parts(pool, rows)
        if parts is None:
            name = type(pool).__name__
            if name not in self._unsupported:
                self._unsupported.add(name)
                logger.warning("HiCache verify: rows of %s are not checked", name)
            return None
        slots = transfer.host_indices.to(device="cpu", dtype=torch.int64)
        return slots, _row_hashes(parts, num_rows=rows.shape[0], device=rows.device)

    def _table(self, transfer) -> _HashTable:
        key = (id(transfer.host_pool), id(transfer.device_pool))
        if key not in self._tables:
            self._tables[key] = _HashTable()
        return self._tables[key]

    def record_backup(self, transfers) -> None:
        for transfer in transfers:
            if transfer.host_indices.numel() == 0:
                continue
            result = self._hashes(transfer)
            if result is None:
                self.counts["unsupported_written"] += transfer.host_indices.numel()
                continue
            slots, hashes = result
            self._table(transfer).put(slots, hashes)
            self.counts["written"] += slots.numel()
        self._maybe_log_stats()

    def check_load(self, transfers) -> None:
        for transfer in transfers:
            if transfer.host_indices.numel() == 0:
                continue
            result = self._hashes(transfer)
            if result is None:
                self.counts["unverified"] += transfer.host_indices.numel()
                continue
            slots, hashes = result
            known, expected = self._table(transfer).get(slots)
            bad = known & (hashes != expected).any(dim=1)
            num_bad = int(bad.sum())
            self.counts["checked"] += int(known.sum())
            self.counts["unverified"] += int((~known).sum())
            self.counts["mismatch"] += num_bad
            if num_bad:
                logger.error(
                    "HiCache verify: %d of %d rows loaded into %s differ from their "
                    "backup (host slots %s)",
                    num_bad,
                    slots.numel(),
                    type(transfer.device_pool).__name__,
                    slots[bad][:8].tolist(),
                )
        self._maybe_log_stats()

    def _maybe_log_stats(self) -> None:
        now = time.monotonic()
        if now - self._last_stats >= _STATS_INTERVAL_S:
            self._last_stats = now
            logger.info(
                "HiCache verify: %s",
                " ".join(f"{k}={v}" for k, v in sorted(self.counts.items())),
            )


def _pool_device(pool) -> torch.device:
    if isinstance(pool, MambaPool):
        return pool.mamba_cache.temporal.device
    return torch.device(getattr(pool, "device", "cpu"))
