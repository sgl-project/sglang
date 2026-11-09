"""Load an all-reduce tune table and pick a backend for a message size.

The tuner writes ``allreduce_configs.json``. Message sizes at runtime are not
the handful of sizes that were measured, so a call uses the largest measured
size that does not exceed the tensor. A tensor smaller than every row uses
the smallest row.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# One read and one log per (path, world size). Several groups share a table.
_LOAD_CACHE: Dict[Tuple[str, int], Optional["AllreduceTuneTable"]] = {}

# Groups whose all-reduce is the tensor-parallel reduction the tuner measured.
TUNE_TABLE_GROUPS = frozenset(
    {"tp", "attention_tp", "moe_tp", "shared_experts_tp"}
)

_QR_PREFIX = "quick_allreduce_"


@dataclass(frozen=True)
class TuneRow:
    size_bytes: int
    backend: str
    dtype: str


class AllreduceTuneTable:
    def __init__(
        self,
        rows: Sequence[TuneRow],
        tp_size: Optional[int],
        source: str,
    ) -> None:
        self.rows = tuple(sorted(rows, key=lambda row: row.size_bytes))
        self.tp_size = tp_size
        self.source = source
        self._cache: Dict[Tuple[int, str], str] = {}

    @property
    def qr_codecs(self) -> Tuple[str, ...]:
        codecs: List[str] = []
        for row in self.rows:
            if not row.backend.startswith(_QR_PREFIX):
                continue
            codec = row.backend[len(_QR_PREFIX) :]
            if codec and codec not in codecs:
                codecs.append(codec)
        return tuple(codecs)

    def backend_for(self, nbytes: int, dtype: str) -> str:
        key = (nbytes, dtype)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        matched = [row for row in self.rows if not row.dtype or row.dtype == dtype]
        rows = matched or list(self.rows)
        chosen = rows[0]
        for row in rows:
            if row.size_bytes <= nbytes:
                chosen = row
            else:
                break
        self._cache[key] = chosen.backend
        return chosen.backend


def parse_tune_backend(backend: str) -> Optional[str]:
    """Map a tuner backend name to the dispatch method.

    ``None`` means the name is not a backend this table can select.
    ``"nccl"`` is the process-group all-reduce. On ROCm that group is RCCL.
    ``"qr:INT4"`` carries the codec.
    """
    if backend == "custom_allreduce":
        return "ca"
    if backend == "nccl":
        return "nccl"
    if backend == "torch_symm_mem":
        return "torch_symm_mem"
    if backend.startswith(_QR_PREFIX) and backend[len(_QR_PREFIX) :]:
        return "qr:" + backend[len(_QR_PREFIX) :]
    return None


def load_allreduce_tune_table(
    path: Optional[str],
    world_size: int,
) -> Optional[AllreduceTuneTable]:
    if not path:
        return None
    cached = _LOAD_CACHE.get((path, world_size))
    if cached is not None or (path, world_size) in _LOAD_CACHE:
        return cached
    if not os.path.isfile(path):
        logger.warning("[AR] Tune table %s does not exist. Using the built-in order.", path)
        _LOAD_CACHE[(path, world_size)] = None
        return None
    try:
        with open(path) as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("[AR] Could not read tune table %s: %s", path, exc)
        _LOAD_CACHE[(path, world_size)] = None
        return None

    tp_size = payload.get("tp_size")
    if tp_size is not None and int(tp_size) != world_size:
        logger.warning(
            "[AR] Tune table %s is for tp=%s, this group has world size %s. "
            "Using the built-in order.",
            path,
            tp_size,
            world_size,
        )
        _LOAD_CACHE[(path, world_size)] = None
        return None

    rows: List[TuneRow] = []
    for entry in payload.get("configs") or []:
        try:
            rows.append(
                TuneRow(
                    size_bytes=int(entry["size_bytes"]),
                    backend=str(entry["backend"]),
                    dtype=str(entry.get("dtype") or ""),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("[AR] Skipping a tune-table row in %s: %s", path, exc)
    if not rows:
        logger.warning("[AR] Tune table %s has no rows. Using the built-in order.", path)
        _LOAD_CACHE[(path, world_size)] = None
        return None

    table = AllreduceTuneTable(rows, tp_size, path)
    logger.info(
        "[AR] Tune table %s (tp=%s): %s",
        path,
        tp_size,
        ", ".join(f"{row.size_bytes}B->{row.backend}" for row in table.rows),
    )
    _LOAD_CACHE[(path, world_size)] = table
    return table
