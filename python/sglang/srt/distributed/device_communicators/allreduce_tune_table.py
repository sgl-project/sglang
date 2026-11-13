"""Load an all-reduce tune table and pick a backend for a message size.

``python -m sglang.auto_tune.run_allreduce`` writes ``allreduce_configs.json``.
A table from an eager run goes in ``SGLANG_TUNED_EAGER_CONFIG``; a table from
a ``--graph 1`` run goes in ``SGLANG_TUNED_GRAPH_CONFIG``. Either may be
unset, which keeps the built-in order for that stage.

Message sizes at runtime are not the handful of sizes that were measured, so
a call uses the largest measured size that does not exceed the tensor. A
tensor smaller than every row uses the smallest row.
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
TUNE_TABLE_GROUPS = frozenset({"tp", "attention_tp", "moe_tp", "shared_experts_tp"})

_QR_PREFIX = "quick_allreduce_"
# FlyDSL INT4 is a different communicator from HIP QuickReduce. Its table
# name starts with the same prefix, so it has to be recognized first.
_FLYDSL_QR_PREFIX = "quick_allreduce_flydsl"


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
        mode: Optional[str],
        source: str,
    ) -> None:
        self.rows = tuple(sorted(rows, key=lambda row: row.size_bytes))
        self.tp_size = tp_size
        self.mode = mode
        self.source = source
        self._cache: Dict[Tuple[int, str], str] = {}

    @property
    def qr_codecs(self) -> Tuple[str, ...]:
        codecs: List[str] = []
        for row in self.rows:
            if not row.backend.startswith(_QR_PREFIX):
                continue
            if row.backend.startswith(_FLYDSL_QR_PREFIX):
                continue
            codec = row.backend[len(_QR_PREFIX) :]
            if codec and codec not in codecs:
                codecs.append(codec)
        return tuple(codecs)

    def has_backend(self, name: str) -> bool:
        return any(row.backend == name for row in self.rows)

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
    ``"nccl"`` is the NCCL all-reduce. On ROCm the library is RCCL.
    ``"qr:INT4"`` carries the HIP QuickReduce codec.
    ``"qr_flydsl"`` is AITER FlyDSL INT4, not a HIP codec.
    """
    if backend == "custom_allreduce":
        return "ca"
    if backend == "sglang_custom_allreduce":
        return "sglang_ca"
    if backend == "aiter_custom_allreduce":
        return "aiter_ca"
    if backend == "custom_allreduce_v2":
        return "car_v2"
    if backend == "nccl":
        return "nccl"
    if backend == "torch_symm_mem":
        return "torch_symm_mem"
    if backend.startswith(_FLYDSL_QR_PREFIX):
        return "qr_flydsl"
    if backend.startswith(_QR_PREFIX) and backend[len(_QR_PREFIX) :]:
        return "qr:" + backend[len(_QR_PREFIX) :]
    return None


def load_allreduce_tune_table(
    path: Optional[str],
    world_size: int,
    expected_mode: str,
) -> Optional[AllreduceTuneTable]:
    """Read one table. ``expected_mode`` is ``"eager"`` or ``"graph"``."""
    if not path:
        return None
    key = (path, world_size)
    if key in _LOAD_CACHE:
        return _LOAD_CACHE[key]
    _LOAD_CACHE[key] = None
    if not os.path.isfile(path):
        logger.warning(
            "[AR] Tune table %s does not exist. Using the built-in order.", path
        )
        return None
    try:
        with open(path) as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("[AR] Could not read tune table %s: %s", path, exc)
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
        return None

    mode = payload.get("mode")
    if mode is not None and mode != expected_mode:
        # Still used: the file is the user's choice. The warning is there
        # because eager and graph timings of AITER differ by the copy-in.
        logger.warning(
            "[AR] Tune table %s was measured in %s mode but is set as the %s "
            "table.",
            path,
            mode,
            expected_mode,
        )

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
        logger.warning(
            "[AR] Tune table %s has no rows. Using the built-in order.", path
        )
        return None

    table = AllreduceTuneTable(rows, tp_size, mode, path)
    logger.info(
        "[AR] %s tune table %s (tp=%s): %s",
        expected_mode,
        path,
        tp_size,
        ", ".join(f"{row.size_bytes}B->{row.backend}" for row in table.rows),
    )
    _LOAD_CACHE[key] = table
    return table
