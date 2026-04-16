"""Shape frequency profiling for profile-guided kernel prewarm.

Tracks which (kernel, shape) combinations are dispatched most frequently.
Raw shapes are recorded; bucketing is deferred to F4 selector. The
frequency table persists across runs so cumulative usage data informs
prewarm ordering and (later) kernel family selection.

Not on the hot path — called from _periodic_log aggregation, not per-token.
"""

import json
import logging
import math
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ShapeKey:
    """Immutable kernel dispatch shape descriptor.

    Generic enough for attention, GEMM, and quantization kernels.
    kernel_name disambiguates; dims holds the shape-specific values.
    """
    kernel_name: str
    dims: Tuple[int, ...]   # e.g. (batch, seq_len, head_dim, n_heads) or (M, N, K)
    dtype: str = "fp16"

    def to_key_str(self) -> str:
        dim_str = "x".join(str(d) for d in self.dims)
        return f"{self.kernel_name}:{dim_str}:{self.dtype}"

    @classmethod
    def from_key_str(cls, s: str) -> "ShapeKey":
        parts = s.split(":")
        if len(parts) < 3:
            raise ValueError(f"Invalid shape key: {s}")
        name = parts[0]
        dims = tuple(int(d) for d in parts[1].split("x"))
        dtype = parts[2]
        return cls(kernel_name=name, dims=dims, dtype=dtype)

    def bucket_pow2(self) -> "ShapeKey":
        """Return a copy with each dim rounded up to the next power of 2.

        Useful for F4 shape family grouping. Not used in F2 recording.
        """
        def _next_pow2(v: int) -> int:
            if v <= 0:
                return 1
            return 1 << (v - 1).bit_length()

        return ShapeKey(
            kernel_name=self.kernel_name,
            dims=tuple(_next_pow2(d) for d in self.dims),
            dtype=self.dtype,
        )


@dataclass
class ShapeRecord:
    """Mutable accumulator for one shape key's dispatch statistics."""
    count: int = 0
    total_us: float = 0.0
    min_us: float = float("inf")
    max_us: float = 0.0
    last_seen: float = 0.0

    @property
    def avg_us(self) -> float:
        return self.total_us / self.count if self.count > 0 else 0.0

    def merge(self, other: "ShapeRecord"):
        """Merge another record into this one (for reload)."""
        self.count += other.count
        self.total_us += other.total_us
        self.min_us = min(self.min_us, other.min_us)
        self.max_us = max(self.max_us, other.max_us)
        self.last_seen = max(self.last_seen, other.last_seen)

    def to_dict(self) -> dict:
        return {
            "count": self.count,
            "total_us": round(self.total_us, 2),
            "min_us": round(self.min_us, 2) if self.min_us != float("inf") else 0.0,
            "max_us": round(self.max_us, 2),
            "avg_us": round(self.avg_us, 2),
            "last_seen": self.last_seen,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ShapeRecord":
        return cls(
            count=d.get("count", 0),
            total_us=d.get("total_us", 0.0),
            min_us=d.get("min_us", float("inf")),
            max_us=d.get("max_us", 0.0),
            last_seen=d.get("last_seen", 0.0),
        )


class ShapeProfile:
    """Persistent shape frequency tracker.

    Thread-safe. Records are keyed by ShapeKey.to_key_str().
    Persistence format is JSON, co-located with the kernel cache manifest.
    """

    def __init__(self):
        self._records: Dict[str, ShapeRecord] = {}
        self._lock = threading.Lock()
        self._dirty = False

    def record(self, shape: ShapeKey, latency_us: float = 0.0):
        """Record one dispatch observation."""
        key = shape.to_key_str()
        now = time.time()
        with self._lock:
            if key not in self._records:
                self._records[key] = ShapeRecord()
            rec = self._records[key]
            rec.count += 1
            if latency_us > 0:
                rec.total_us += latency_us
                rec.min_us = min(rec.min_us, latency_us)
                rec.max_us = max(rec.max_us, latency_us)
            rec.last_seen = now
            self._dirty = True

    def top_shapes(self, n: int = 20) -> List[Tuple[ShapeKey, ShapeRecord]]:
        """Return top-N shapes by dispatch frequency."""
        with self._lock:
            items = list(self._records.items())
        items.sort(key=lambda x: x[1].count, reverse=True)
        result = []
        for key_str, rec in items[:n]:
            try:
                result.append((ShapeKey.from_key_str(key_str), rec))
            except ValueError:
                continue
        return result

    def save(self, path: Path):
        """Persist profiles to JSON. Atomic write via tmp+rename."""
        with self._lock:
            if not self._dirty:
                return
            data = {k: v.to_dict() for k, v in self._records.items()}
            self._dirty = False

        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        try:
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2)
            tmp.rename(path)
        except OSError as e:
            logger.warning("SOK shape_profile: save failed: %s", e)

    def load(self, path: Path) -> int:
        """Load profiles from JSON. Merges with any existing in-memory data."""
        if not path.exists():
            return 0
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("SOK shape_profile: load failed: %s", e)
            return 0

        loaded = 0
        with self._lock:
            for key_str, rec_dict in data.items():
                try:
                    rec = ShapeRecord.from_dict(rec_dict)
                    if key_str in self._records:
                        self._records[key_str].merge(rec)
                    else:
                        self._records[key_str] = rec
                    loaded += 1
                except (TypeError, KeyError):
                    continue

        if loaded > 0:
            logger.info("SOK shape_profile: loaded %d shape records", loaded)
        return loaded

    def get_stats(self) -> dict:
        """Summary statistics for logging."""
        with self._lock:
            n = len(self._records)
            total_dispatches = sum(r.count for r in self._records.values())
            top3 = sorted(
                self._records.items(),
                key=lambda x: x[1].count,
                reverse=True,
            )[:3]
        return {
            "shapes": n,
            "total_dispatches": total_dispatches,
            "top3": [
                f"{k}({r.count})" for k, r in top3
            ],
        }
