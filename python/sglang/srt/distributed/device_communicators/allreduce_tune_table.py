"""Load an all-reduce tune table and pick a communicator class for a message size.

``python -m sglang.auto_tune.run_allreduce`` writes ``allreduce_configs_<mode>.json``.
Each row's ``backend`` is the full class name the tuner timed, for example
``sglang.srt.distributed.device_communicators.pynccl.PyNcclCommunicator``.
``method`` is the collective that row timed: ``allreduce``,
``quick_all_reduce``, ``custom_all_reduce``, ``outplace_all_reduce``, or
``all_reduce``. The server opens one communicator per class and calls the
collective that class defines. The same method name is a different call on
another class. A table written before ``method`` existed names a class and
records no collective, so that class is not opened.
A table from an eager run goes in ``SGLANG_TUNED_EAGER_CONFIG``; a table from
a ``--graph 1`` run goes in ``SGLANG_TUNED_GRAPH_CONFIG``. Either may be
unset, which keeps the built-in order for that stage.

Message sizes at runtime are not the handful of sizes that were measured, so
a call uses the largest measured size that does not exceed the tensor. A
tensor smaller than every row uses the smallest row.

``qr_quant`` and ``qr_cast_bf16_to_fp16`` are kept beside the class name.
``QuickAllReduce`` reads both from the environment at construction, and the
class name does not carry the codec that was measured.
``fly_algorithm`` is the FlyDSL schedule (``mesh`` or ``ring``). Both
``QuickAllReduceInt4`` and ``FlyQuickAllReduce`` take it at construction,
and the class name does not carry it.
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


@dataclass(frozen=True)
class TuneRow:
    size_bytes: int
    class_name: str
    dtype: str
    method: str = ""


class AllreduceTuneTable:
    def __init__(
        self,
        rows: Sequence[TuneRow],
        tp_size: Optional[int],
        mode: Optional[str],
        source: str,
        qr_quant: Optional[str] = None,
        qr_cast_bf16_to_fp16: Optional[bool] = None,
        fly_algorithm: str = "mesh",
    ) -> None:
        self.rows: Tuple[TuneRow, ...] = tuple(
            sorted(rows, key=lambda row: row.size_bytes)
        )
        self.tp_size: Optional[int] = tp_size
        self.mode: Optional[str] = mode
        self.source: str = source
        self.qr_quant: Optional[str] = qr_quant
        self.qr_cast_bf16_to_fp16: Optional[bool] = qr_cast_bf16_to_fp16
        self.fly_algorithm: str = fly_algorithm
        self._cache: Dict[Tuple[int, str], str] = {}

    def class_names(self) -> Tuple[str, ...]:
        """Full class names named by at least one row, smallest size first."""
        return tuple(dict.fromkeys(row.class_name for row in self.rows))

    def method_for(self, class_name: str) -> str:
        """Collective the tuner timed for this class.

        Empty when the rows omit it or name more than one. One class is
        opened once, so every row that names it must agree.
        """
        found: List[str] = []
        for row in self.rows:
            if row.class_name != class_name or not row.method:
                continue
            if row.method not in found:
                found.append(row.method)
        if len(found) == 1:
            return found[0]
        if len(found) > 1:
            logger.warning(
                "[AR] Tune table %s names %s with methods %s. Not opening it.",
                self.source,
                class_name,
                ", ".join(found),
            )
        return ""

    def class_for(self, nbytes: int, dtype: str) -> str:
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
        self._cache[key] = chosen.class_name
        return chosen.class_name


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
                    class_name=str(entry["backend"]),
                    dtype=str(entry.get("dtype") or ""),
                    method=str(entry.get("method") or ""),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("[AR] Skipping a tune-table row in %s: %s", path, exc)
    if not rows:
        logger.warning(
            "[AR] Tune table %s has no rows. Using the built-in order.", path
        )
        return None

    qr_quant = payload.get("qr_quant") or None
    if qr_quant is not None:
        qr_quant = str(qr_quant)
    cast = payload.get("qr_cast_bf16_to_fp16")
    qr_cast = None if cast is None else bool(int(cast))
    fly_algorithm = str(payload.get("fly_algorithm") or "mesh")
    if fly_algorithm not in ("mesh", "ring"):
        logger.warning(
            "[AR] Tune table %s has fly_algorithm=%s. FlyDSL opens as mesh.",
            path,
            fly_algorithm,
        )
        fly_algorithm = "mesh"
    table = AllreduceTuneTable(
        rows,
        tp_size,
        mode,
        path,
        qr_quant=qr_quant,
        qr_cast_bf16_to_fp16=qr_cast,
        fly_algorithm=fly_algorithm,
    )
    logger.info(
        "[AR] %s tune table %s (tp=%s, qr_quant=%s, fly_algorithm=%s): %s",
        expected_mode,
        path,
        tp_size,
        qr_quant,
        fly_algorithm,
        ", ".join(f"{row.size_bytes}B->{row.class_name}" for row in table.rows),
    )
    _LOAD_CACHE[key] = table
    return table
