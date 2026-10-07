"""Shared trace contract for InferCast simulator and real-serving replays."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TraceRequest:
    timestamp_ms: float
    input_length: int
    output_length: int
    token_ids: list[int]
    metadata: dict[str, Any]


def token_ids_for_row(row: dict, input_length: int, request_index: int) -> list[int]:
    hash_ids = row.get("hash_ids")
    if not isinstance(hash_ids, list) or not hash_ids:
        return [1000 + request_index] * input_length

    block_size = int(row.get("block_size", 64))
    if block_size < 1:
        raise ValueError("block_size must be positive")
    tokens = [1000 + int(hash_id) for hash_id in hash_ids for _ in range(block_size)]
    if len(tokens) < input_length:
        tokens.extend([120000 + request_index] * (input_length - len(tokens)))
    return tokens[:input_length]


def load_trace_requests(path: Path) -> list[TraceRequest]:
    requests = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        try:
            timestamp_ms = float(row["timestamp_ms"])
            input_length = int(row["input_length"])
            output_length = int(row["output_length"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid trace row {line_number}: {error}") from error
        if timestamp_ms < 0 or input_length < 1 or output_length < 1:
            raise ValueError(f"invalid trace row {line_number}: values out of range")
        metadata = {
            key: value
            for key, value in row.items()
            if key not in {"timestamp_ms", "input_length", "output_length"}
        }
        requests.append(
            TraceRequest(
                timestamp_ms=timestamp_ms,
                input_length=input_length,
                output_length=output_length,
                token_ids=token_ids_for_row(row, input_length, len(requests)),
                metadata=metadata,
            )
        )
    if not requests:
        raise ValueError("trace must contain at least one request")
    return requests


def file_identity(path: Path) -> dict[str, str]:
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
