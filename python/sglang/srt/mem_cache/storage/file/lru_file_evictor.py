# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

"""LRU/size-based file eviction for the HiCache file storage backend.

``HiCacheFile`` is a thin raw-bytes store: it suffixes keys, reads/writes
``.bin`` pages, and answers existence queries. Everything that bounds how much
disk those pages consume -- the LRU recency index, per-file size accounting,
free-space probing, scanning pre-existing files on startup, and unlinking
victims -- lives here so the backend stays a plain key/value store.

A backend constructs one evictor and drives it through a small lifecycle::

    touch(key, path)                 # read hit / already-on-disk: bump recency
    reserve(key, n_bytes) -> bool    # admit a new write, evicting if needed
        commit(key)                  #   write landed on disk
        abort(key)                   #   write failed; release the reservation
    clear()                          # backend wiped all files

When eviction is not configured the evictor is inert: ``reserve`` always admits
and the other calls are no-ops, so the backend behaves as unbounded storage.
"""

from __future__ import annotations

import argparse
import logging
import os
import threading
from collections import OrderedDict
from typing import Any, Callable, Optional, Set, Tuple

from sglang.srt.environ import envs
from sglang.srt.utils.common import human_readable_int

logger = logging.getLogger(__name__)


def _parse_size_to_bytes(value: Any) -> int:
    """Parse a size to bytes via human_readable_int (e.g. '200G', '1Gi', '1048576').
    None / empty / '0' disables; an invalid value also disables (with a warning)."""
    if value is None:
        return 0
    if isinstance(value, (int, float)):
        return max(0, int(value))
    s = str(value).strip()
    if not s or s == "0":
        return 0
    try:
        return max(0, human_readable_int(s))
    except (argparse.ArgumentTypeError, ValueError):
        logger.warning(f"Invalid size {value!r} for HiCacheFile; disabling.")
        return 0


class LRUFileEvictor:
    """Bounds the on-disk size of a HiCacheFile directory via LRU eviction.

    Tracks one ``.bin`` file per suffixed key (oldest at the front of the LRU),
    enforces an optional byte cap and an optional free-space watermark, and
    unlinks the least-recently-used files to stay within those bounds. Eviction
    config comes from ``extra_config`` (per-backend, takes precedence) falling
    back to the ``SGLANG_HICACHE_FILE_BACKEND_*`` env vars.
    """

    def __init__(
        self,
        file_path: str,
        config_suffix: str,
        *,
        tp_rank: int,
        is_mla_model: bool,
        extra_config: Optional[dict] = None,
        on_evict: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.file_path = file_path
        self.config_suffix = config_suffix
        self._tp_rank = tp_rank
        self._on_evict = on_evict

        # MLA ranks share the same physical files, so centralize LRU bookkeeping
        # on rank 0; non-MLA ranks each own their own files via the suffix.
        self._is_storage_owner = (not is_mla_model) or (tp_rank == 0)

        # suffixed_key -> file size in bytes; oldest at front.
        self._lru: OrderedDict[str, int] = OrderedDict()
        self._pending_writes: Set[str] = set()
        self._total_bytes: int = 0
        self._lock = threading.Lock()

        self._load_config(extra_config or {})

        self._eviction_configured = self.max_size_bytes > 0 or self.min_free_bytes > 0
        self._eviction_enabled = self._eviction_configured and self._is_storage_owner
        if self._eviction_configured and not self._is_storage_owner:
            logger.info(
                f"HiCacheFile rank {self._tp_rank} (MLA): eviction handled by rank 0; "
                f"this rank skips LRU bookkeeping and will not create new files."
            )

        if not self._eviction_enabled:
            return

        # Clamp max_size to the filesystem capacity so a too-large cap can't OOM tmpfs.
        fs = self._fs_stats()
        if fs is not None and self.max_size_bytes > 0:
            safe_max = max(0, fs[0] - self.min_free_bytes)
            if self.max_size_bytes > safe_max:
                logger.warning(
                    f"HiCacheFile max_size exceeds filesystem capacity; "
                    f"clamping to {safe_max} B."
                )
                self.max_size_bytes = safe_max

        self._scan_existing_files()
        with self._lock:
            if self.max_size_bytes > 0 and self._total_bytes > self.max_size_bytes:
                self._evict_locked(0)
            if self.min_free_bytes > 0:
                self._enforce_free_space_locked(0)
        logger.info(
            f"HiCacheFile eviction enabled: cap={self.max_size_bytes} B, "
            f"watermark={self.eviction_ratio:.2f}, min_free={self.min_free_bytes} B, "
            f"existing={self._total_bytes} B ({len(self._lru)} entries)"
        )

    def _load_config(self, extra: dict) -> None:
        # extra_config (per-backend) takes precedence over env vars.
        def _cfg(key, env):
            val = extra.get(key)
            return env.get() if val is None else val

        self.max_size_bytes = _parse_size_to_bytes(
            _cfg("max_size", envs.SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE)
        )
        self.min_free_bytes = _parse_size_to_bytes(
            _cfg("min_free_space", envs.SGLANG_HICACHE_FILE_BACKEND_MIN_FREE_SPACE)
        )

        ratio_raw = _cfg(
            "eviction_ratio", envs.SGLANG_HICACHE_FILE_BACKEND_EVICTION_RATIO
        )
        try:
            self.eviction_ratio = float(ratio_raw)
        except (TypeError, ValueError):
            self.eviction_ratio = 0.9
        if not (0.0 < self.eviction_ratio <= 1.0):
            self.eviction_ratio = 0.9

    @property
    def enabled(self) -> bool:
        """True when this rank actively evicts (configured AND storage owner)."""
        return self._eviction_enabled

    @property
    def configured(self) -> bool:
        """True when a cap or free-space watermark is set (on any rank)."""
        return self._eviction_configured

    @property
    def is_storage_owner(self) -> bool:
        """True when this rank owns (and may create/evict) the on-disk files."""
        return self._is_storage_owner

    def reserve(self, suffixed_key: str, value_bytes: int, key: str = "") -> bool:
        """Admit a new write of ``value_bytes``, evicting LRU victims as needed.

        On success the key is pre-reserved at MRU and flagged in-flight so a
        concurrent ``reserve`` won't evict it before the file is committed; the
        caller must then call ``commit`` (write landed) or ``abort`` (write
        failed). Returns ``False`` -- reserving nothing -- when the write is
        refused: this rank is not the storage owner, the value is larger than
        the cap, there is no evictable space, or the free-space watermark cannot
        be met. When eviction is not configured the write is always admitted.
        """
        if not self._eviction_configured:
            return True  # unbounded storage: nothing to enforce
        if not self._is_storage_owner:
            logger.warning(
                f"HiCacheFile rank {self._tp_rank} is not the MLA storage owner; "
                f"not caching new key {key} because file eviction is enabled."
            )
            return False
        if self.max_size_bytes > 0 and value_bytes > self.max_size_bytes:
            logger.warning(
                f"HiCacheFile: value {value_bytes} B exceeds cap "
                f"{self.max_size_bytes} B; not caching {key}"
            )
            return False

        with self._lock:
            # Cap-based eviction: evict, then bail if still over cap.
            if (
                self.max_size_bytes > 0
                and (self._total_bytes + value_bytes) > self.max_size_bytes
            ):
                self._evict_locked(value_bytes)
                if (self._total_bytes + value_bytes) > self.max_size_bytes:
                    logger.warning(
                        f"HiCacheFile: no evictable space for {value_bytes} B "
                        f"under cap {self.max_size_bytes} B; not caching {key}"
                    )
                    return False
            # Free-space watermark.
            if self.min_free_bytes > 0 and not self._enforce_free_space_locked(
                value_bytes
            ):
                logger.warning(
                    f"HiCacheFile: filesystem hosting {self.file_path!r} "
                    f"would fall below min_free={self.min_free_bytes} B "
                    f"after writing {value_bytes} B; refusing {key} "
                    f"to avoid OOM/ENOSPC."
                )
                return False
            # Pre-reserve at MRU so a concurrent evict won't grab this slot.
            prev = self._lru.pop(suffixed_key, None)
            if prev is not None:
                self._total_bytes -= prev
            self._lru[suffixed_key] = value_bytes
            self._pending_writes.add(suffixed_key)
            self._total_bytes += value_bytes
        return True

    def commit(self, suffixed_key: str) -> None:
        """Mark a reserved write as durably on disk (clears its in-flight flag)."""
        if not self._eviction_enabled:
            return
        with self._lock:
            self._pending_writes.discard(suffixed_key)

    def abort(self, suffixed_key: str) -> None:
        """Release a reservation whose write failed: drop it and refund the bytes."""
        if not self._eviction_enabled:
            return
        with self._lock:
            cur = self._lru.pop(suffixed_key, None)
            self._pending_writes.discard(suffixed_key)
            if cur is not None:
                self._total_bytes -= cur

    def touch(self, suffixed_key: str, tensor_path: str) -> None:
        """Mark key as MRU, adopting an untracked on-disk file if needed."""
        if not self._eviction_enabled:
            return
        with self._lock:
            if suffixed_key in self._lru:
                self._lru.move_to_end(suffixed_key, last=True)
                return
        # Untracked file: stat without holding the lock.
        try:
            size = os.path.getsize(tensor_path)
        except OSError:
            return
        with self._lock:
            if suffixed_key in self._lru:
                self._lru.move_to_end(suffixed_key, last=True)
            else:
                self._lru[suffixed_key] = size
                self._total_bytes += size

    def clear(self) -> None:
        """Reset all bookkeeping after the backend has removed the files."""
        with self._lock:
            self._lru.clear()
            self._pending_writes.clear()
            self._total_bytes = 0

    def _fs_stats(self) -> Optional[tuple]:
        """(total, available) bytes for the filesystem; None if unavailable."""
        try:
            st = os.statvfs(self.file_path)
        except (OSError, AttributeError):
            return None
        total = st.f_blocks * st.f_frsize
        free = st.f_bavail * st.f_frsize
        return total, free

    def _enforce_free_space_locked(self, value_bytes: int) -> bool:
        """Evict until writing value_bytes still leaves min_free_bytes free.
        Caller holds _lock. Returns False if the write can't be satisfied."""
        if self.min_free_bytes <= 0:
            return True
        fs = self._fs_stats()
        if fs is None:
            return True  # cannot probe -> permissive, fall back to OS errors
        # tmpfs frees space on unlink, so credit reclaimed bytes back to the
        # estimate rather than re-probing statvfs on every eviction.
        free = fs[1]
        self._evict_while(
            lambda reclaimed: (free + reclaimed) - value_bytes < self.min_free_bytes
        )
        # Re-probe: external writers may have changed free space meanwhile.
        fs = self._fs_stats()
        if fs is None:
            return True
        return fs[1] - value_bytes >= self.min_free_bytes

    def _scan_existing_files(self) -> None:
        """Seed LRU index from disk on startup (oldest mtime first)."""
        try:
            names = os.listdir(self.file_path)
        except FileNotFoundError:
            return
        entries = []
        for fn in names:
            if not fn.endswith(".bin"):
                continue
            stem = fn[:-4]
            # Only files belonging to this rank/model.
            if not stem.endswith(self.config_suffix):
                continue
            fp = os.path.join(self.file_path, fn)
            try:
                st = os.stat(fp)
            except OSError:
                continue
            entries.append((st.st_mtime, stem, st.st_size))
        entries.sort(key=lambda e: e[0])  # oldest first
        for _, stem, size in entries:
            self._lru[stem] = size
            self._total_bytes += size

    def _evict_one_lru_locked(self) -> Tuple[str, int]:
        """Evict the single oldest evictable LRU entry. Caller holds _lock.

        The shared pop / skip-pending / unlink / ``_total_bytes`` step driven by
        `_evict_while`. Returns ``(outcome, freed_bytes)``:

        - ``("evicted", n)``: oldest entry dropped from the index; ``n`` disk
          bytes reclaimed (0 if the file was already gone).
        - ``("skipped", 0)``: oldest entry is an in-flight write; re-pinned at MRU
          so the writer is not evicted out from under itself.
        - ``("stop", 0)``: nothing evictable (empty index) or the unlink failed
          (entry re-pinned at LRU); the caller should stop its eviction loop.
        """
        if not self._lru:
            return "stop", 0
        evict_stem, evict_size = self._lru.popitem(last=False)  # oldest
        if evict_stem in self._pending_writes:
            # Keep in-flight reservations; their file isn't committed yet.
            self._lru[evict_stem] = evict_size
            return "skipped", 0
        tensor_path = os.path.join(self.file_path, f"{evict_stem}.bin")
        try:
            os.remove(tensor_path)
            freed = evict_size
            if self._on_evict is not None:
                self._on_evict(evict_stem)
        except FileNotFoundError:
            freed = 0  # file already gone; still drop the stale index entry
            if self._on_evict is not None:
                self._on_evict(evict_stem)
        except OSError as e:
            logger.warning(f"HiCacheFile eviction failed for {evict_stem}: {e}")
            self._lru[evict_stem] = evict_size
            self._lru.move_to_end(evict_stem, last=False)
            return "stop", 0
        self._total_bytes -= evict_size
        return "evicted", freed

    def _evict_while(self, should_continue) -> int:
        """Evict oldest non-pending entries while ``should_continue(reclaimed)``.

        ``should_continue`` is passed the disk bytes reclaimed so far and returns
        whether to keep evicting. In-flight writes are skipped; the loop is bounded
        so it can't spin once every remaining entry is pending. Caller holds _lock.
        Returns the total disk bytes reclaimed.
        """
        reclaimed = 0
        attempts_left = len(self._lru)
        while self._lru and attempts_left > 0 and should_continue(reclaimed):
            outcome, freed = self._evict_one_lru_locked()
            if outcome == "stop":
                break
            if outcome == "skipped":
                attempts_left -= 1
                continue
            # An entry left the index; reset the skip budget and bank the bytes.
            reclaimed += freed
            attempts_left = len(self._lru)
        return reclaimed

    def _evict_locked(self, needed_bytes: int) -> None:
        """Evict LRU entries until total + needed <= cap*ratio. Caller holds _lock."""
        if self.max_size_bytes <= 0:
            return
        target = max(0, int(self.max_size_bytes * self.eviction_ratio) - needed_bytes)
        reclaimed = self._evict_while(lambda _: self._total_bytes > target)
        if reclaimed:
            logger.debug(
                f"HiCacheFile reclaimed {reclaimed} bytes; "
                f"now {self._total_bytes} bytes used"
            )
