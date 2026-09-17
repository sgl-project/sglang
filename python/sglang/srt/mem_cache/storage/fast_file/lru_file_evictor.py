# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

"""Background LRU eviction for the ``fast_file`` HiCache storage backend.

The evictor owns the LRU recency index, per-file size accounting, free-space
probing, the startup scan, and victim removal for one namespace directory.
Without ``max_size`` or ``min_free_space`` it is inert: ``reserve`` always
admits and every other call is a no-op.
"""

from __future__ import annotations

import argparse
import logging
import os
import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Optional

from sglang.srt.utils.common import human_readable_int

logger = logging.getLogger(__name__)

_DEFAULT_EVICT_HIGH_WATERMARK = 0.95
_DEFAULT_EVICT_LOW_WATERMARK = 0.85
_DEFAULT_PREEVICT_INTERVAL_MS = 10
_DEFAULT_EVICT_BATCH_SIZE = 128
_DEFAULT_STALE_TEMP_AGE_S = 3600.0


def setting(extra_config: dict, key: str, default: Any) -> Any:
    """extra_config value for ``key``; ``None`` counts as unset."""
    value = extra_config.get(key)
    return default if value is None else value


def parse_size_to_bytes(value: Any) -> int:
    """Parse a byte size such as ``"200G"``, ``"1Gi"`` or ``1048576``; 0 disables."""
    if value is None:
        return 0
    if isinstance(value, (int, float)):
        parsed = int(value)
    else:
        text = str(value).strip()
        if not text or text == "0":
            return 0
        try:
            parsed = human_readable_int(text)
        except (argparse.ArgumentTypeError, ValueError) as exc:
            raise ValueError(f"Invalid HiCacheFastFile size: {value!r}") from exc
    if parsed < 0:
        raise ValueError(f"HiCacheFastFile size must be nonnegative, got {value!r}")
    return parsed


def _parse_ratio(name: str, value: Any) -> float:
    try:
        ratio = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"HiCacheFastFile {name} must be a number") from exc
    if not (0.0 < ratio <= 1.0):
        raise ValueError(f"HiCacheFastFile {name} must be in (0, 1], got {ratio}")
    return ratio


def _parse_positive_int(name: str, value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"HiCacheFastFile {name} must be a positive integer") from exc
    if isinstance(value, bool) or parsed <= 0:
        raise ValueError(f"HiCacheFastFile {name} must be a positive integer")
    return parsed


def _parse_nonnegative_float(name: str, value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"HiCacheFastFile {name} must be a nonnegative number"
        ) from exc
    if parsed < 0:
        raise ValueError(f"HiCacheFastFile {name} must be nonnegative, got {parsed}")
    return parsed


class LRUFileEvictor:
    """Bounds one HiCacheFastFile namespace directory via LRU eviction.

    Tracks one ``.bin`` file per suffixed key (oldest first), enforces an
    optional byte cap and an optional filesystem free-space floor, and unlinks
    the least recently used files to stay within them. A background thread
    drains the namespace from ``evict_high_watermark`` down to
    ``evict_low_watermark`` so foreground writers rarely have to evict.
    """

    def __init__(
        self,
        file_path: str,
        config_suffix: str,
        *,
        tp_rank: int,
        is_mla_model: bool,
        extra_config: dict,
        on_evict: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.file_path = file_path
        self.config_suffix = config_suffix
        self._tp_rank = tp_rank
        self._on_evict = on_evict
        # MLA ranks share one namespace, so rank 0 owns the bookkeeping.
        self._is_storage_owner = (not is_mla_model) or tp_rank == 0

        # suffixed_key -> file size in bytes; oldest at the front.
        self._lru: OrderedDict[str, int] = OrderedDict()
        self._pending_writes: set[str] = set()
        self._total_bytes = 0
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._eviction_thread: Optional[threading.Thread] = None
        self._draining = False
        self._foreground_evicted_entries = 0
        self._background_evicted_entries = 0

        self._load_config(extra_config)
        self._eviction_configured = self.max_size_bytes > 0 or self.min_free_bytes > 0
        self._eviction_enabled = self._eviction_configured and self._is_storage_owner
        if self._eviction_configured and not self._is_storage_owner:
            logger.info(
                f"HiCacheFastFile rank {tp_rank} (MLA): eviction is handled by rank 0; "
                "this rank skips LRU bookkeeping and will not create new files."
            )

        if self._eviction_enabled:
            fs = self._fs_stats()
            if fs is None and self.min_free_bytes > 0:
                raise OSError(
                    "HiCacheFastFile cannot enforce min_free_space because filesystem "
                    f"statistics are unavailable for {self.file_path!r}."
                )
            if fs is not None and self.max_size_bytes > 0:
                # A cap above the filesystem size would let tmpfs OOM the host.
                safe_max = max(0, fs[0] - self.min_free_bytes)
                if self.max_size_bytes > safe_max:
                    logger.warning(
                        "HiCacheFastFile max_size exceeds filesystem capacity; "
                        f"clamping to {safe_max} B."
                    )
                    self.max_size_bytes = safe_max

        if self._is_storage_owner and (
            self._eviction_enabled or self.stale_temp_age_s > 0
        ):
            self._scan_existing_files(track_entries=self._eviction_enabled)

        if not self._eviction_enabled:
            return

        with self._lock:
            if self.max_size_bytes > 0 and self._total_bytes > self.max_size_bytes:
                self._evict_to_fit_locked(0)
            if self.min_free_bytes > 0:
                self._enforce_free_space_locked(0)
        logger.info(
            f"HiCacheFastFile eviction enabled: cap={self.max_size_bytes} B, "
            f"high={self.evict_high_watermark:.2f}, low={self.evict_low_watermark:.2f}, "
            f"min_free={self.min_free_bytes} B, existing={self._total_bytes} B "
            f"({len(self._lru)} entries)"
        )
        if self.max_size_bytes > 0 and self.evict_high_watermark < 1.0:
            self._eviction_thread = threading.Thread(
                target=self._preevict_loop,
                name=f"HiCacheFastFileEvict-{tp_rank}",
                daemon=True,
            )
            self._eviction_thread.start()

    def _load_config(self, extra: dict) -> None:
        self.max_size_bytes = parse_size_to_bytes(extra.get("max_size"))
        self.min_free_bytes = parse_size_to_bytes(extra.get("min_free_space"))
        self.evict_high_watermark = _parse_ratio(
            "evict_high_watermark",
            setting(extra, "evict_high_watermark", _DEFAULT_EVICT_HIGH_WATERMARK),
        )
        self.evict_low_watermark = _parse_ratio(
            "evict_low_watermark",
            setting(extra, "evict_low_watermark", _DEFAULT_EVICT_LOW_WATERMARK),
        )
        if self.evict_low_watermark > self.evict_high_watermark:
            raise ValueError(
                "HiCacheFastFile evict_low_watermark must not exceed "
                "evict_high_watermark"
            )
        self.preevict_interval_s = (
            _parse_positive_int(
                "preevict_interval_ms",
                setting(extra, "preevict_interval_ms", _DEFAULT_PREEVICT_INTERVAL_MS),
            )
            / 1000.0
        )
        self.evict_batch_size = _parse_positive_int(
            "evict_batch_size",
            setting(extra, "evict_batch_size", _DEFAULT_EVICT_BATCH_SIZE),
        )
        self.stale_temp_age_s = _parse_nonnegative_float(
            "stale_temp_age_s",
            setting(extra, "stale_temp_age_s", _DEFAULT_STALE_TEMP_AGE_S),
        )

    @property
    def enabled(self) -> bool:
        """True when this rank actively evicts (configured and storage owner)."""
        return self._eviction_enabled

    @property
    def configured(self) -> bool:
        """True when a cap or free-space floor is set (on any rank)."""
        return self._eviction_configured

    @property
    def is_storage_owner(self) -> bool:
        return self._is_storage_owner

    def reserve(self, suffixed_key: str, value_bytes: int, *, key: str = "") -> bool:
        """Admit a write of ``value_bytes``, evicting LRU victims as needed.

        On success the key is pre-reserved at MRU and flagged in flight so a
        concurrent eviction cannot remove it before ``commit``; a failed write
        must ``abort``. Returns False when the write is refused.
        """
        if not self._eviction_configured:
            return True
        if not self._is_storage_owner:
            logger.warning(
                f"HiCacheFastFile rank {self._tp_rank} is not the MLA storage owner; "
                f"not caching new key {key} because file eviction is enabled."
            )
            return False
        if self.max_size_bytes > 0 and value_bytes > self.max_size_bytes:
            logger.warning(
                f"HiCacheFastFile: value {value_bytes} B exceeds cap "
                f"{self.max_size_bytes} B; not caching {key}"
            )
            return False

        with self._lock:
            if (
                self.max_size_bytes > 0
                and self._total_bytes + value_bytes > self.max_size_bytes
            ):
                self._evict_to_fit_locked(value_bytes)
                if self._total_bytes + value_bytes > self.max_size_bytes:
                    logger.warning(
                        f"HiCacheFastFile: no evictable space for {value_bytes} B "
                        f"under cap {self.max_size_bytes} B; not caching {key}"
                    )
                    return False
            if self.min_free_bytes > 0 and not self._enforce_free_space_locked(
                value_bytes
            ):
                logger.warning(
                    f"HiCacheFastFile: filesystem hosting {self.file_path!r} would "
                    f"fall below min_free={self.min_free_bytes} B after writing "
                    f"{value_bytes} B; refusing {key}."
                )
                return False
            previous = self._lru.pop(suffixed_key, None)
            if previous is not None:
                self._total_bytes -= previous
            self._lru[suffixed_key] = value_bytes
            self._pending_writes.add(suffixed_key)
            self._total_bytes += value_bytes
        return True

    def commit(self, suffixed_key: str) -> None:
        if not self._eviction_enabled:
            return
        with self._lock:
            self._pending_writes.discard(suffixed_key)

    def abort(self, suffixed_key: str) -> None:
        if not self._eviction_enabled:
            return
        with self._lock:
            size = self._lru.pop(suffixed_key, None)
            self._pending_writes.discard(suffixed_key)
            if size is not None:
                self._total_bytes -= size

    def touch(
        self, suffixed_key: str, tensor_path: str, *, size: Optional[int] = None
    ) -> None:
        """Mark a page MRU, adopting an untracked on-disk file if needed.

        Pass ``size`` when the caller already knows the on-disk size; it also
        corrects a stale accounted size.
        """
        if not self._eviction_enabled:
            return
        if size is None:
            with self._lock:
                if suffixed_key in self._lru:
                    self._lru.move_to_end(suffixed_key, last=True)
                    return
            try:
                size = os.path.getsize(tensor_path)
            except OSError:
                return
        with self._lock:
            previous = self._lru.pop(suffixed_key, None)
            if previous is not None:
                self._total_bytes -= previous
            self._lru[suffixed_key] = size
            self._total_bytes += size

    def forget(self, suffixed_key: str) -> bool:
        """Drop the accounting of a page that is missing or was removed.

        An in-flight reservation is kept: a reader can miss the file between
        ``reserve`` and the rename that publishes it.
        """
        if not self._eviction_enabled:
            return False
        with self._lock:
            if suffixed_key in self._pending_writes:
                return False
            size = self._lru.pop(suffixed_key, None)
            if size is None:
                return False
            self._total_bytes -= size
            return True

    def clear_storage(self) -> bool:
        """Remove this namespace's pages and temp files and reset bookkeeping.

        Files that belong to another key suffix are left alone. Returns False
        if any matching file could not be removed.
        """
        success = True
        remaining: OrderedDict[str, int] = OrderedDict()
        with self._lock:
            try:
                entries = list(os.scandir(self.file_path))
            except FileNotFoundError:
                entries = []
            except OSError as exc:
                logger.error(
                    "Failed to scan HiCacheFastFile storage during clear: %s", exc
                )
                return False
            for entry in entries:
                page_stem = self._matching_page_stem(entry.name)
                is_temp = self._is_matching_temp(entry.name)
                if page_stem is None and not is_temp:
                    continue
                try:
                    if not entry.is_file(follow_symlinks=False):
                        continue
                    os.remove(entry.path)
                except FileNotFoundError:
                    pass
                except OSError as exc:
                    success = False
                    logger.warning(
                        "Failed to clear HiCacheFastFile path %s: %s", entry.path, exc
                    )
                    if page_stem is not None and self._eviction_enabled:
                        try:
                            remaining[page_stem] = entry.stat(
                                follow_symlinks=False
                            ).st_size
                        except OSError:
                            pass
            self._lru = remaining
            self._pending_writes.clear()
            self._total_bytes = sum(remaining.values())
            self._draining = False
            self._foreground_evicted_entries = 0
            self._background_evicted_entries = 0
        return success

    def close(self) -> None:
        self._stop_event.set()
        thread = self._eviction_thread
        self._eviction_thread = None
        if thread is not None:
            thread.join()

    def snapshot(self) -> dict[str, int | bool]:
        with self._lock:
            return {
                "total_bytes": self._total_bytes,
                "entries": len(self._lru),
                "pending_writes": len(self._pending_writes),
                "draining": self._draining,
                "foreground_evicted_entries": self._foreground_evicted_entries,
                "background_evicted_entries": self._background_evicted_entries,
            }

    def _matching_page_stem(self, filename: str) -> Optional[str]:
        if not filename.endswith(".bin"):
            return None
        stem = filename[:-4]
        return stem if stem.endswith(self.config_suffix) else None

    def _is_matching_temp(self, filename: str) -> bool:
        stem, marker, _ = filename.partition(".bin.tmp.")
        return bool(marker) and stem.endswith(self.config_suffix)

    def _fs_stats(self) -> Optional[tuple[int, int]]:
        """(total, available) bytes of the filesystem; None if unavailable."""
        try:
            st = os.statvfs(self.file_path)
        except (OSError, AttributeError):
            return None
        return st.f_blocks * st.f_frsize, st.f_bavail * st.f_frsize

    def _enforce_free_space_locked(self, value_bytes: int) -> bool:
        """Evict until writing ``value_bytes`` still leaves ``min_free_bytes``.

        Caller holds ``_lock``. Returns False if the floor cannot be met, or
        if filesystem statistics are unavailable (fail closed).
        """
        if self.min_free_bytes <= 0:
            return True
        fs = self._fs_stats()
        if fs is None:
            logger.error(
                "HiCacheFastFile cannot enforce min_free_space: filesystem "
                "statistics are unavailable for %r.",
                self.file_path,
            )
            return False
        # Credit reclaimed bytes instead of re-probing statvfs per victim;
        # the final probe below catches filesystems that free space lazily.
        free = fs[1]
        before_entries = len(self._lru)
        self._evict_while(
            lambda reclaimed: (free + reclaimed) - value_bytes < self.min_free_bytes
        )
        self._foreground_evicted_entries += before_entries - len(self._lru)
        fs = self._fs_stats()
        if fs is None:
            return False
        return fs[1] - value_bytes >= self.min_free_bytes

    def _scan_existing_files(self, *, track_entries: bool) -> None:
        """Remove stale temp files and optionally seed the LRU (oldest first)."""
        try:
            entries = list(os.scandir(self.file_path))
        except FileNotFoundError:
            return
        tracked: list[tuple[float, str, int]] = []
        removed_temp_files = 0
        cutoff = time.time() - self.stale_temp_age_s
        for entry in entries:
            is_temp = self._is_matching_temp(entry.name)
            page_stem = self._matching_page_stem(entry.name) if track_entries else None
            if not is_temp and page_stem is None:
                continue
            try:
                if not entry.is_file(follow_symlinks=False):
                    continue
                stat = entry.stat(follow_symlinks=False)
            except OSError:
                continue
            if is_temp:
                if self.stale_temp_age_s > 0 and stat.st_mtime <= cutoff:
                    try:
                        os.remove(entry.path)
                        removed_temp_files += 1
                    except FileNotFoundError:
                        pass
                    except OSError as exc:
                        logger.warning(
                            "Failed to remove stale HiCacheFastFile temp file %s: %s",
                            entry.path,
                            exc,
                        )
                continue
            tracked.append((stat.st_mtime, page_stem, stat.st_size))

        tracked.sort(key=lambda item: item[0])
        for _, stem, size in tracked:
            self._lru[stem] = size
            self._total_bytes += size
        if removed_temp_files:
            logger.info(
                "Removed %d stale HiCacheFastFile temp files", removed_temp_files
            )

    def _evict_one_lru_locked(self) -> tuple[str, int]:
        """Evict the oldest evictable entry. Caller holds ``_lock``.

        Returns ``(outcome, freed_bytes)`` with outcome ``evicted``,
        ``skipped`` (in-flight write, re-pinned at MRU) or ``stop`` (nothing
        evictable, or unlink failed and the entry stays at LRU).
        """
        if not self._lru:
            return "stop", 0
        evict_stem, evict_size = self._lru.popitem(last=False)
        if evict_stem in self._pending_writes:
            self._lru[evict_stem] = evict_size
            return "skipped", 0
        tensor_path = os.path.join(self.file_path, f"{evict_stem}.bin")
        try:
            os.remove(tensor_path)
            freed = evict_size
        except FileNotFoundError:
            freed = 0
        except OSError as exc:
            logger.warning(f"HiCacheFastFile eviction failed for {evict_stem}: {exc}")
            self._lru[evict_stem] = evict_size
            self._lru.move_to_end(evict_stem, last=False)
            return "stop", 0
        if self._on_evict is not None:
            self._on_evict(evict_stem)
        self._total_bytes -= evict_size
        return "evicted", freed

    def _evict_while(self, should_continue, max_entries: Optional[int] = None) -> int:
        """Evict oldest entries while ``should_continue(reclaimed_bytes)``.

        Bounded so it cannot spin once every remaining entry is an in-flight
        write. Caller holds ``_lock``. Returns the disk bytes reclaimed.
        """
        reclaimed = 0
        evicted_entries = 0
        attempts_left = len(self._lru)
        while (
            self._lru
            and attempts_left > 0
            and should_continue(reclaimed)
            and (max_entries is None or evicted_entries < max_entries)
        ):
            outcome, freed = self._evict_one_lru_locked()
            if outcome == "stop":
                break
            if outcome == "skipped":
                attempts_left -= 1
                continue
            reclaimed += freed
            evicted_entries += 1
            attempts_left = len(self._lru)
        return reclaimed

    def _evict_to_fit_locked(self, needed_bytes: int) -> None:
        """Evict just enough to admit ``needed_bytes`` under the cap."""
        if self.max_size_bytes <= 0:
            return
        target = max(0, self.max_size_bytes - needed_bytes)
        before_entries = len(self._lru)
        self._evict_while(lambda _: self._total_bytes > target)
        self._foreground_evicted_entries += before_entries - len(self._lru)

    def _preevict_batch(self) -> int:
        """Drain one bounded batch toward the low watermark; returns bytes freed."""
        with self._lock:
            high = int(self.max_size_bytes * self.evict_high_watermark)
            low = int(self.max_size_bytes * self.evict_low_watermark)
            if self._total_bytes > high:
                self._draining = True
            if not self._draining:
                return 0
            if self._total_bytes <= low:
                self._draining = False
                return 0
            before_entries = len(self._lru)
            reclaimed = self._evict_while(
                lambda _: self._total_bytes > low,
                max_entries=self.evict_batch_size,
            )
            self._background_evicted_entries += before_entries - len(self._lru)
            if self._total_bytes <= low:
                self._draining = False
            return reclaimed

    def _preevict_loop(self) -> None:
        while not self._stop_event.wait(self.preevict_interval_s):
            deadline = time.monotonic() + self.preevict_interval_s
            while not self._stop_event.is_set():
                # Batches release the lock between them so writers keep flowing.
                if self._preevict_batch() == 0 or time.monotonic() >= deadline:
                    break
                time.sleep(0)
