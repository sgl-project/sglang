"""Manifested kernel cache with invalidation and profile-guided prewarm.

Replaces ad-hoc reliance on ~/.triton/cache/ with an explicit persistent
store at ~/.cache/phantom_sok/. Each cached kernel gets a manifest entry
with fingerprint, source hash, timestamps, and validation status.

The cache is cumulative: each inference run teaches future runs which
kernels are hot and should be pre-compiled at model load time.

Cache miss is transparent — falls through to normal Triton JIT. This
module never blocks the inference hot path.
"""

import hashlib
import json
import logging
import os
import shutil
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from sglang.srt.speculative.sok.config import SOKConfig
from sglang.srt.speculative.sok.fingerprint import KernelFingerprint

logger = logging.getLogger(__name__)

# Default cache location
_DEFAULT_CACHE_DIR = os.path.expanduser("~/.cache/phantom_sok")


@dataclass
class CacheEntry:
    """Manifest entry for a single cached kernel."""
    kernel_name: str
    source_hash: str          # SHA256 of kernel source (.py or .triton)
    fingerprint_digest: str   # hex digest of KernelFingerprint
    triton_cache_key: str     # directory name in ~/.triton/cache/
    created_at: float         # time.time()
    last_used: float          # time.time()
    hit_count: int = 0
    compile_time_ms: float = 0.0
    validated: bool = False   # set True after first successful dispatch

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CacheEntry":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class HotShape:
    """A frequently-used kernel shape for prewarm prioritization."""
    kernel_name: str
    triton_cache_key: str
    frequency: int            # how many times observed
    avg_latency_us: float = 0.0
    last_seen: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "HotShape":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class KernelCache:
    """Manifested kernel cache with fingerprint-keyed persistence.

    Usage:
        cache = KernelCache(config, fingerprint)
        cache.load_manifest()       # at server start
        cache.prewarm()             # background pre-JIT
        cache.record_hit(name, key) # during inference
        cache.save_manifest()       # periodically or at shutdown
    """

    def __init__(self, config: SOKConfig, fingerprint: KernelFingerprint):
        self._config = config
        self._fingerprint = fingerprint
        self._fp_digest = fingerprint.hex_digest

        # Resolve cache directory
        if config.cache_dir:
            self._cache_dir = Path(config.cache_dir)
        else:
            self._cache_dir = Path(_DEFAULT_CACHE_DIR)
        self._fp_dir = self._cache_dir / self._fp_digest
        self._manifest_path = self._fp_dir / "manifest.json"
        self._hot_shapes_path = self._fp_dir / "hot_shapes.json"
        self._fingerprint_path = self._fp_dir / "fingerprint.json"

        # In-memory state
        self._entries: Dict[str, CacheEntry] = {}  # triton_cache_key → entry
        self._hot_shapes: Dict[str, HotShape] = {}  # triton_cache_key → shape
        self._lock = threading.Lock()

        # Telemetry counters
        self._hits = 0
        self._misses = 0
        self._prewarm_loaded = 0
        self._prewarm_failed = 0

    @property
    def cache_dir(self) -> Path:
        return self._cache_dir

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    # ── Manifest I/O ──

    def load_manifest(self) -> int:
        """Load manifest from disk. Returns number of entries loaded."""
        if not self._manifest_path.exists():
            logger.info("PHANTOM-SOK: no existing manifest at %s", self._manifest_path)
            return 0

        try:
            with open(self._manifest_path) as f:
                data = json.load(f)

            # Validate fingerprint match
            stored_fp = data.get("fingerprint_digest", "")
            if stored_fp != self._fp_digest:
                logger.warning(
                    "PHANTOM-SOK: fingerprint mismatch (stored=%s, current=%s), "
                    "invalidating cache",
                    stored_fp[:8], self._fp_digest[:8],
                )
                self._invalidate_all()
                return 0

            entries = data.get("entries", {})
            with self._lock:
                for key, edict in entries.items():
                    try:
                        self._entries[key] = CacheEntry.from_dict(edict)
                    except (TypeError, KeyError) as e:
                        logger.debug("PHANTOM-SOK: skipping corrupt entry %s: %s", key, e)

            logger.info("PHANTOM-SOK: loaded %d cache entries from %s",
                        len(self._entries), self._manifest_path)
            return len(self._entries)

        except (json.JSONDecodeError, OSError) as e:
            logger.warning("PHANTOM-SOK: failed to load manifest: %s", e)
            return 0

    def load_hot_shapes(self) -> int:
        """Load hot shape priorities from disk."""
        if not self._hot_shapes_path.exists():
            return 0

        try:
            with open(self._hot_shapes_path) as f:
                data = json.load(f)

            with self._lock:
                for key, sdict in data.items():
                    try:
                        self._hot_shapes[key] = HotShape.from_dict(sdict)
                    except (TypeError, KeyError):
                        pass

            logger.info("PHANTOM-SOK: loaded %d hot shapes", len(self._hot_shapes))
            return len(self._hot_shapes)

        except (json.JSONDecodeError, OSError):
            return 0

    def save_manifest(self):
        """Persist manifest and hot shapes to disk."""
        self._fp_dir.mkdir(parents=True, exist_ok=True)

        # Save fingerprint (human-readable provenance)
        try:
            with open(self._fingerprint_path, "w") as f:
                f.write(self._fingerprint.to_json())
        except OSError as e:
            logger.warning("PHANTOM-SOK: failed to save fingerprint: %s", e)

        # Save manifest
        with self._lock:
            data = {
                "fingerprint_digest": self._fp_digest,
                "saved_at": time.time(),
                "entries": {k: v.to_dict() for k, v in self._entries.items()},
            }
        try:
            tmp = self._manifest_path.with_suffix(".tmp")
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2)
            tmp.rename(self._manifest_path)
        except OSError as e:
            logger.warning("PHANTOM-SOK: failed to save manifest: %s", e)

        # Save hot shapes
        with self._lock:
            shapes_data = {k: v.to_dict() for k, v in self._hot_shapes.items()}
        try:
            tmp = self._hot_shapes_path.with_suffix(".tmp")
            with open(tmp, "w") as f:
                json.dump(shapes_data, f, indent=2)
            tmp.rename(self._hot_shapes_path)
        except OSError as e:
            logger.warning("PHANTOM-SOK: failed to save hot shapes: %s", e)

        logger.info("PHANTOM-SOK: saved manifest (%d entries, %d hot shapes)",
                     len(self._entries), len(self._hot_shapes))

    def save_hot_shapes(self):
        """Persist only the hot shapes to disk (lightweight save).

        Use this for frequent periodic saves where full manifest I/O is
        unnecessary. The hot shapes file is small and fast to write.
        """
        self._fp_dir.mkdir(parents=True, exist_ok=True)

        with self._lock:
            shapes_data = {k: v.to_dict() for k, v in self._hot_shapes.items()}

        if not shapes_data:
            return

        try:
            tmp = self._hot_shapes_path.with_suffix(".tmp")
            with open(tmp, "w") as f:
                json.dump(shapes_data, f, indent=2)
            tmp.rename(self._hot_shapes_path)
        except OSError as e:
            logger.warning("PHANTOM-SOK: failed to save hot shapes: %s", e)

    # ── Prewarm ──

    def prewarm(self) -> int:
        """Pre-verify that hot kernels are in Triton's cache.

        Returns number of kernels confirmed present. This does NOT
        recompile — it checks that the Triton cache directory still
        contains the expected .hsaco files. If present, Triton will
        skip JIT on first use.

        Call this in a background thread during model load.
        """
        if not self._config.enable_prewarm:
            return 0

        triton_cache = Path.home() / ".triton" / "cache"
        if not triton_cache.exists():
            logger.info("PHANTOM-SOK: no Triton cache at %s, prewarm skipped", triton_cache)
            return 0

        # Sort hot shapes by frequency (most frequent first)
        with self._lock:
            shapes = sorted(
                self._hot_shapes.values(),
                key=lambda s: s.frequency,
                reverse=True,
            )[:self._config.prewarm_top_n]

        loaded = 0
        failed = 0
        for shape in shapes:
            cache_key = shape.triton_cache_key
            kernel_dir = triton_cache / cache_key
            if kernel_dir.exists():
                # Check for .hsaco (compiled GPU binary)
                hsaco_files = list(kernel_dir.glob("*.hsaco"))
                if hsaco_files:
                    loaded += 1
                    # Touch the cache entry to keep it from being pruned
                    with self._lock:
                        if cache_key in self._entries:
                            self._entries[cache_key].last_used = time.time()
                else:
                    failed += 1
            else:
                failed += 1

        self._prewarm_loaded = loaded
        self._prewarm_failed = failed
        logger.info(
            "PHANTOM-SOK: prewarm verified %d/%d hot kernels (%.1f%% hit rate)",
            loaded, loaded + failed,
            loaded / max(loaded + failed, 1) * 100,
        )
        return loaded

    # ── Recording ──

    def record_jit(self, kernel_name: str, triton_cache_key: str,
                   source_hash: str = "", compile_time_ms: float = 0.0):
        """Record a Triton JIT compilation event.

        Called after a kernel is compiled for the first time (or cache miss).
        Adds or updates the manifest entry.
        """
        now = time.time()
        with self._lock:
            if triton_cache_key in self._entries:
                entry = self._entries[triton_cache_key]
                entry.last_used = now
                entry.hit_count += 1
            else:
                self._entries[triton_cache_key] = CacheEntry(
                    kernel_name=kernel_name,
                    source_hash=source_hash,
                    fingerprint_digest=self._fp_digest,
                    triton_cache_key=triton_cache_key,
                    created_at=now,
                    last_used=now,
                    hit_count=1,
                    compile_time_ms=compile_time_ms,
                )
                self._misses += 1

    def record_hit(self, kernel_name: str, triton_cache_key: str):
        """Record a kernel cache hit (no JIT needed).

        Called on each dispatch to track hot shapes and hit rate.
        """
        now = time.time()
        with self._lock:
            self._hits += 1

            if triton_cache_key in self._entries:
                entry = self._entries[triton_cache_key]
                entry.last_used = now
                entry.hit_count += 1
                entry.validated = True

            # Update hot shape frequency
            if triton_cache_key in self._hot_shapes:
                shape = self._hot_shapes[triton_cache_key]
                shape.frequency += 1
                shape.last_seen = now
            else:
                self._hot_shapes[triton_cache_key] = HotShape(
                    kernel_name=kernel_name,
                    triton_cache_key=triton_cache_key,
                    frequency=1,
                    last_seen=now,
                )

    # ── Scanning ──

    def scan_triton_cache(self) -> int:
        """Scan Triton's cache dir and register any kernels we don't know about.

        This bootstraps the manifest on first run by discovering what
        Triton has already compiled.
        """
        triton_cache = Path.home() / ".triton" / "cache"
        if not triton_cache.exists():
            return 0

        discovered = 0
        for cache_dir in triton_cache.iterdir():
            if not cache_dir.is_dir():
                continue
            cache_key = cache_dir.name

            with self._lock:
                if cache_key in self._entries:
                    continue

            # Try to read kernel name from Triton's JSON metadata
            kernel_name = "unknown"
            source_hash = ""
            json_files = list(cache_dir.glob("*.json"))
            for jf in json_files:
                if jf.name.startswith("__grp__"):
                    continue
                try:
                    with open(jf) as f:
                        meta = json.load(f)
                    kernel_name = meta.get("name", jf.stem)
                    source_hash = meta.get("hash", "")
                    break
                except (json.JSONDecodeError, OSError):
                    pass

            # Only register if it has a compiled binary
            hsaco_files = list(cache_dir.glob("*.hsaco"))
            if not hsaco_files:
                continue

            now = time.time()
            with self._lock:
                self._entries[cache_key] = CacheEntry(
                    kernel_name=kernel_name,
                    source_hash=source_hash,
                    fingerprint_digest=self._fp_digest,
                    triton_cache_key=cache_key,
                    created_at=now,
                    last_used=now,
                    hit_count=0,
                    validated=False,
                )
                # Also register as a hot shape with frequency=0 (bootstrap)
                if cache_key not in self._hot_shapes:
                    self._hot_shapes[cache_key] = HotShape(
                        kernel_name=kernel_name,
                        triton_cache_key=cache_key,
                        frequency=0,
                        last_seen=now,
                    )
            discovered += 1

        if discovered > 0:
            logger.info("PHANTOM-SOK: scanned Triton cache, discovered %d new kernels",
                        discovered)
        return discovered

    # ── Pruning ──

    def prune_stale(self, max_age_days: float = 30.0) -> int:
        """Remove manifest entries not used in max_age_days."""
        cutoff = time.time() - (max_age_days * 86400)
        pruned = 0
        with self._lock:
            stale_keys = [
                k for k, v in self._entries.items()
                if v.last_used < cutoff
            ]
            for k in stale_keys:
                del self._entries[k]
                self._hot_shapes.pop(k, None)
                pruned += 1

        if pruned > 0:
            logger.info("PHANTOM-SOK: pruned %d stale entries (>%.0f days old)",
                        pruned, max_age_days)
        return pruned

    # ── Invalidation ──

    def _invalidate_all(self):
        """Clear all cached entries (fingerprint mismatch)."""
        with self._lock:
            self._entries.clear()
            self._hot_shapes.clear()
        # Remove stale manifest files
        for p in [self._manifest_path, self._hot_shapes_path]:
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass
        logger.info("PHANTOM-SOK: invalidated all cache entries")

    # ── Diagnostics ──

    def get_stats(self) -> dict:
        """Return cache statistics for logging."""
        with self._lock:
            n_entries = len(self._entries)
            n_validated = sum(1 for e in self._entries.values() if e.validated)
            n_hot = len(self._hot_shapes)
            top_hot = sorted(
                self._hot_shapes.values(),
                key=lambda s: s.frequency,
                reverse=True,
            )[:5]

        return {
            "entries": n_entries,
            "validated": n_validated,
            "hot_shapes": n_hot,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self.hit_rate,
            "prewarm_loaded": self._prewarm_loaded,
            "prewarm_failed": self._prewarm_failed,
            "top_kernels": [
                f"{s.kernel_name}({s.frequency})" for s in top_hot
            ],
        }
