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
is for debugging only.
"""

from __future__ import annotations

import functools
import logging
import time
from collections import Counter
from typing import Callable, Optional

import torch

from sglang.srt.mem_cache.memory_pool import MambaPool

logger = logging.getLogger(__name__)

_STATS_INTERVAL_S = 60.0


def _row_parts(pool, rows: torch.Tensor) -> Optional[list[torch.Tensor]]:
    """The device rows' state as [n, ...] tensors, or None for an unsupported pool."""
    if isinstance(pool, MambaPool):
        cache = pool.mamba_cache
        return [
            state[:, rows].transpose(0, 1) for state in (*cache.conv, cache.temporal)
        ]
    k_buffer = getattr(pool, "k_buffer", None)
    if k_buffer is not None:
        v_buffer = getattr(pool, "v_buffer", None)
        buffers = [*k_buffer, *(v_buffer if v_buffer is not None else ())]
        return [buf[rows] for buf in buffers]
    kv_buffer = getattr(pool, "kv_buffer", None)
    if kv_buffer is not None:
        return [buf[rows] for buf in kv_buffer]
    full_kv_pool = getattr(pool, "full_kv_pool", None)
    if full_kv_pool is not None:
        return _row_parts(full_kv_pool, rows)
    return None


@functools.lru_cache(maxsize=256)
def _coeffs(part: int, width: int, device: torch.device) -> torch.Tensor:
    """Two fixed odd int64 multipliers per word, the same for backup and load."""
    gen = torch.Generator(device="cpu").manual_seed(part * 1_000_003 + width)
    coeffs = torch.randint(
        -(1 << 62), 1 << 62, (2, width), generator=gen, dtype=torch.int64
    )
    return (coeffs | 1).to(device)


def _row_hashes(parts: list[torch.Tensor]) -> torch.Tensor:
    """[n, 2] int64: two polynomial hashes (wrapping) over each row's bytes."""
    n = parts[0].shape[0]
    hashes = torch.zeros(n, 2, dtype=torch.int64, device=parts[0].device)
    for index, part in enumerate(parts):
        row_bytes = part.contiguous().view(torch.uint8).reshape(n, -1)
        pad = (-row_bytes.shape[1]) % 8
        if pad:
            row_bytes = torch.nn.functional.pad(row_bytes, (0, pad))
        words = row_bytes.view(torch.int64)
        coeffs = _coeffs(index, words.shape[1], words.device)
        hashes += (words.unsqueeze(1) * coeffs).sum(dim=2)
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
        return slots, _row_hashes(parts)

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
