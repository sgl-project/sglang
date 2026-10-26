# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Binary ModelRunner step-shape recorder.

The forward thread only copies per-request N_Q and KV lengths into an
in-memory buffer. The file is written once, at process exit. Median and text
are computed by :func:`read_step_shape_records`.

On-disk layout, little-endian::

    magic ``SGLSTEP1`` (8 bytes)
    repeated records:
        u64 time_ns          wall-clock nanoseconds (``time.time_ns``)
        u8  stage            see ``_STAGE_BY_NAME``
        u8  flags            bit 0 is set for a draft worker
        u16 batch_size
        u32 forward_id
        i32 q[batch_size]    per-request query length
        i32 k[batch_size]    per-request KV length

``summary`` and ``vector`` record the same bytes. Holding every request
length until exit is a few tens of megabytes for an hour-long run.
"""

from __future__ import annotations

import atexit
import logging
import os
import struct
import sys
import threading
import time
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

MAGIC = b"SGLSTEP1"
_HEADER = struct.Struct("<QBBHI")
_FLAG_DRAFT = 1
_EMPTY = np.empty(0, dtype="<i4")

_STAGE_BY_NAME = {
    "EXTEND": 1,
    "DECODE": 2,
    "MIXED": 3,
    "TARGET_VERIFY": 4,
    "DRAFT_EXTEND_V2": 5,
    "IDLE": 6,
    "PREBUILT": 7,
    "SPLIT_PREFILL": 8,
    "DLLM_EXTEND": 9,
}
_NAME_BY_STAGE = {value: name for name, value in _STAGE_BY_NAME.items()}
_EXTEND_MODES = {"EXTEND", "MIXED", "DRAFT_EXTEND_V2", "SPLIT_PREFILL", "DLLM_EXTEND"}
_DECODE_MODES = {"DECODE", "TARGET_VERIFY", "PREBUILT"}
_ENABLED_MODES = {"summary", "vector"}


def _query_width(forward_batch) -> int:
    spec = getattr(forward_batch, "spec_info", None)
    width = getattr(spec, "num_tokens_per_req", -1) if spec is not None else -1
    return width if isinstance(width, int) and width > 0 else 1


def _as_i32(value) -> np.ndarray:
    if isinstance(value, np.ndarray):
        array = value
    else:
        numpy = getattr(value, "numpy", None)
        array = numpy() if numpy is not None else value
    return np.ascontiguousarray(array, dtype="<i4")


def _lengths(forward_batch) -> tuple[int, np.ndarray, np.ndarray]:
    mode_name = getattr(forward_batch.forward_mode, "name", "")
    stage = _STAGE_BY_NAME.get(mode_name, 0)
    if mode_name in _EXTEND_MODES:
        queries = forward_batch.extend_seq_lens_cpu
        prefixes = forward_batch.extend_prefix_lens_cpu
        if queries is None or prefixes is None:
            return stage, _EMPTY, _EMPTY
        query = _as_i32(queries)
        kv = query + _as_i32(prefixes)
        return stage, query, kv
    if mode_name in _DECODE_MODES:
        seq_lens = forward_batch.seq_lens_cpu
        if seq_lens is None:
            return stage, _EMPTY, _EMPTY
        kv = _as_i32(seq_lens)
        query = np.full(kv.shape[0], _query_width(forward_batch), dtype="<i4")
        return stage, query, kv
    return stage, _EMPTY, _EMPTY


def _median(values: np.ndarray) -> float | None:
    count = int(values.shape[0])
    if count == 0:
        return None
    ordered = np.sort(values)
    mid = count // 2
    if count % 2:
        return float(ordered[mid])
    return float(ordered[mid - 1] + ordered[mid]) / 2.0


@dataclass(eq=False)
class StepShapeRecord:
    """One forward step. Aggregates are derived from the stored lengths."""

    time_ns: int
    stage: str
    draft: bool
    batch_size: int
    forward_id: int
    q: np.ndarray
    k: np.ndarray

    @property
    def q_sum(self) -> int:
        return int(self.q.sum()) if self.batch_size else 0

    @property
    def k_sum(self) -> int:
        return int(self.k.sum()) if self.batch_size else 0

    @property
    def q_min(self) -> int | None:
        return int(self.q.min()) if self.batch_size else None

    @property
    def q_max(self) -> int | None:
        return int(self.q.max()) if self.batch_size else None

    @property
    def k_min(self) -> int | None:
        return int(self.k.min()) if self.batch_size else None

    @property
    def k_max(self) -> int | None:
        return int(self.k.max()) if self.batch_size else None

    @property
    def q_median(self) -> float | None:
        return _median(self.q)

    @property
    def k_median(self) -> float | None:
        return _median(self.k)


class StepShapeRecorder:
    """Append-only memory buffer. ``flush`` writes the file once."""

    def __init__(self, path: str):
        self.path = path
        self._buf = bytearray()
        self._lock = threading.Lock()
        self._flushed = False
        self._broken = False

    @property
    def buffered_bytes(self) -> int:
        return len(self._buf)

    def record(self, forward_batch, *, is_draft_worker: bool, forward_id: int) -> None:
        if self._broken or self._flushed:
            return
        try:
            stage, query, kv = _lengths(forward_batch)
            if query.shape[0] != kv.shape[0] or query.shape[0] > 0xFFFF:
                raise ValueError(
                    f"step-shape length mismatch q={query.shape[0]} k={kv.shape[0]}"
                )
            batch_size = int(query.shape[0])
            flags = _FLAG_DRAFT if is_draft_worker else 0
            header = _HEADER.pack(
                time.time_ns(), stage, flags, batch_size, forward_id & 0xFFFFFFFF
            )
            payload_q = query.tobytes()
            payload_k = kv.tobytes()
        except Exception:
            self._disable_after_error()
            return
        with self._lock:
            if self._flushed or self._broken:
                return
            self._buf.extend(header)
            self._buf.extend(payload_q)
            self._buf.extend(payload_k)

    def flush(self) -> None:
        with self._lock:
            if self._flushed:
                return
            self._write(self._buf)
            self._flushed = True
            self._buf = bytearray()

    def _write(self, payload: bytearray) -> None:
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        temporary = self.path + ".part"
        with open(temporary, "wb") as handle:
            handle.write(MAGIC)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.path)
        logger.info("Wrote step-shape log to %s (%d bytes)", self.path, len(payload))

    def _disable_after_error(self) -> None:
        if self._broken:
            return
        self._broken = True
        logger.exception("Disabling step-shape log after a record failure")


_RECORDER: StepShapeRecorder | None = None
_DISABLED = False


def read_step_shape_records(path: str) -> list[StepShapeRecord]:
    """Read a flushed step-shape file."""
    with open(path, "rb") as handle:
        blob = handle.read()
    if not blob.startswith(MAGIC):
        raise ValueError(f"{path} is not an SGLSTEP1 step-shape log")
    offset = len(MAGIC)
    header_size = _HEADER.size
    records: list[StepShapeRecord] = []
    while offset < len(blob):
        if offset + header_size > len(blob):
            raise ValueError(f"{path} truncated at header offset {offset}")
        time_ns, stage, flags, batch_size, forward_id = _HEADER.unpack_from(
            blob, offset
        )
        offset += header_size
        nbytes = batch_size * 4
        end = offset + nbytes * 2
        if end > len(blob):
            raise ValueError(f"{path} truncated at record offset {offset}")
        query = np.frombuffer(blob, dtype="<i4", count=batch_size, offset=offset)
        kv = np.frombuffer(blob, dtype="<i4", count=batch_size, offset=offset + nbytes)
        offset = end
        records.append(
            StepShapeRecord(
                time_ns=int(time_ns),
                stage=_NAME_BY_STAGE.get(stage, "UNKNOWN"),
                draft=bool(flags & _FLAG_DRAFT),
                batch_size=int(batch_size),
                forward_id=int(forward_id),
                q=query.copy(),
                k=kv.copy(),
            )
        )
    return records


def get_step_shape_recorder() -> StepShapeRecorder | None:
    """Return the process recorder, or None when the flag or path is off.

    The env var is read once. Call this only from the primary attention rank.
    """
    global _RECORDER, _DISABLED
    if _DISABLED:
        return None
    if _RECORDER is not None:
        return _RECORDER
    from sglang.srt.environ import envs

    mode = envs.SGLANG_LOG_STEP_SHAPES.get().lower()
    path = envs.SGLANG_LOG_STEP_SHAPES_PATH.get().strip()
    if mode not in _ENABLED_MODES or not path:
        if mode in _ENABLED_MODES and not path:
            logger.warning(
                "SGLANG_LOG_STEP_SHAPES=%s but SGLANG_LOG_STEP_SHAPES_PATH is empty; "
                "step shapes will not be recorded",
                mode,
            )
        _DISABLED = True
        return None
    recorder = StepShapeRecorder(path)
    _RECORDER = recorder
    atexit.register(flush_step_shape_log)
    logger.info("Buffering step shapes until process exit: %s", path)
    return recorder


def flush_step_shape_log() -> None:
    """Write the buffered records. Safe to call more than once."""
    recorder = _RECORDER
    if recorder is None:
        return
    try:
        recorder.flush()
    except Exception:
        logger.exception("Failed to flush step-shape log")


def reset_step_shape_recorder_for_tests() -> None:
    global _RECORDER, _DISABLED
    _RECORDER = None
    _DISABLED = False


def _csv_cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def main(argv: list[str] | None = None) -> int:
    """Write a CSV of per-step aggregates. ``--vectors`` adds the length lists."""
    import argparse

    parser = argparse.ArgumentParser(description="Decode an SGLSTEP1 step-shape log")
    parser.add_argument("path")
    parser.add_argument("-o", "--output", help="CSV path; default is stdout")
    parser.add_argument(
        "--vectors",
        action="store_true",
        help="Append comma-separated per-request q and k lengths",
    )
    args = parser.parse_args(argv)
    records = read_step_shape_records(args.path)
    header = [
        "t_ns",
        "stage",
        "draft",
        "bs",
        "forward_id",
        "qsum",
        "qmin",
        "qmed",
        "qmax",
        "ksum",
        "kmin",
        "kmed",
        "kmax",
    ]
    if args.vectors:
        header.extend(("q", "k"))
    lines = [",".join(header)]
    for record in records:
        row = [
            record.time_ns,
            record.stage,
            int(record.draft),
            record.batch_size,
            record.forward_id,
            record.q_sum,
            record.q_min,
            record.q_median,
            record.q_max,
            record.k_sum,
            record.k_min,
            record.k_median,
            record.k_max,
        ]
        cells = [_csv_cell(value) for value in row]
        if args.vectors:
            cells.append('"[' + ",".join(map(str, record.q.tolist())) + ']"')
            cells.append('"[' + ",".join(map(str, record.k.tolist())) + ']"')
        lines.append(",".join(cells))
    text = "\n".join(lines) + "\n"
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
