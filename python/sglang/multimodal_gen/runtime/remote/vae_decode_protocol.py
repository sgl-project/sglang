# SPDX-License-Identifier: Apache-2.0
"""Wire helpers for the experimental exact realtime VAE decoder."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import msgspec.msgpack
import torch

SCHEMA_VERSION = "sglang-realtime-vae/v1"
RAW_RGB_CONTENT_TYPE = "application/x-raw-rgb"
RAW_RGB_FRAMES_PER_TRANSPORT_BATCH = 16
SHARED_MEMORY_DIR_ENV = "SGLANG_REALTIME_VAE_SHM_DIR"
DEFAULT_SHARED_MEMORY_DIR = "/dev/shm/sglang-realtime-vae"

_NAME_TO_DTYPE = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}
_DTYPE_TO_NAME = {dtype: name for name, dtype in _NAME_TO_DTYPE.items()}


def tensor_to_payload(tensor: torch.Tensor) -> dict[str, Any]:
    cpu = tensor.detach().contiguous().cpu()
    try:
        dtype_name = _DTYPE_TO_NAME[cpu.dtype]
    except KeyError as exc:
        raise TypeError(
            f"unsupported tensor dtype for remote VAE: {cpu.dtype}"
        ) from exc
    return {
        "shape": list(cpu.shape),
        "dtype": dtype_name,
        "data": cpu.view(torch.uint8).numpy().tobytes(),
    }


def payload_to_tensor(payload: dict[str, Any]) -> torch.Tensor:
    shape = tuple(int(dim) for dim in payload["shape"])
    try:
        dtype = _NAME_TO_DTYPE[str(payload["dtype"])]
    except KeyError as exc:
        raise TypeError(
            f"unsupported tensor dtype from remote VAE: {payload['dtype']}"
        ) from exc
    flat = torch.frombuffer(bytearray(payload["data"]), dtype=dtype)
    return flat.reshape(shape).contiguous()


def packb(value: Any) -> bytes:
    return msgspec.msgpack.encode(value)


def unpackb(value: bytes) -> Any:
    return msgspec.msgpack.decode(value)


def build_raw_transport_batches(
    frame_batches: list[list[bytes]],
) -> list[list[dict[str, Any]]]:
    """Prejoin raw frames where the remote decoder can hide the copy behind DiT."""
    transport_batches = []
    for frames in frame_batches:
        batches = []
        if not frames:
            batches.append({"num_frames": 0, "payload": b""})
        for start in range(0, len(frames), RAW_RGB_FRAMES_PER_TRANSPORT_BATCH):
            split = frames[start : start + RAW_RGB_FRAMES_PER_TRANSPORT_BATCH]
            batches.append(
                {
                    "num_frames": len(split),
                    "payload": b"".join(split),
                }
            )
        transport_batches.append(batches)
    return transport_batches


def store_raw_transport_batches_in_shared_memory(
    transport_batches: list[list[dict[str, Any]]],
    *,
    root: str | Path | None = None,
) -> list[list[dict[str, Any]]]:
    shared_root = Path(
        root or os.environ.get(SHARED_MEMORY_DIR_ENV, DEFAULT_SHARED_MEMORY_DIR)
    ).resolve()
    shared_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    stored_batches = []
    for logical_batch in transport_batches:
        stored_logical_batch = []
        for batch in logical_batch:
            payload = batch["payload"]
            if not isinstance(payload, bytes):
                raise TypeError("raw transport payload must be bytes")
            path = shared_root / f"{os.getpid()}-{uuid4().hex}.bin"
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
            stored_logical_batch.append(
                {
                    "num_frames": int(batch["num_frames"]),
                    "num_bytes": len(payload),
                    "path": str(path),
                }
            )
        stored_batches.append(stored_logical_batch)
    return stored_batches


def materialize_raw_transport_batches_from_shared_memory(
    stored_batches: list[list[dict[str, Any]]],
    *,
    root: str | Path | None = None,
) -> list[list[dict[str, Any]]]:
    shared_root = Path(
        root or os.environ.get(SHARED_MEMORY_DIR_ENV, DEFAULT_SHARED_MEMORY_DIR)
    ).resolve()
    validated_paths: list[Path] = []
    materialized_batches = []
    try:
        for logical_batch in stored_batches:
            materialized_logical_batch = []
            for batch in logical_batch:
                path = Path(str(batch["path"])).resolve(strict=True)
                if path.parent != shared_root:
                    raise ValueError(
                        f"remote VAE shared-memory path escapes root: {path}"
                    )
                validated_paths.append(path)
                payload = path.read_bytes()
                expected_bytes = int(batch["num_bytes"])
                if len(payload) != expected_bytes:
                    raise RuntimeError(
                        "remote VAE shared-memory payload size mismatch: "
                        f"expected {expected_bytes}, got {len(payload)}"
                    )
                materialized_logical_batch.append(
                    {
                        "num_frames": int(batch["num_frames"]),
                        "payload": payload,
                    }
                )
            materialized_batches.append(materialized_logical_batch)
        return materialized_batches
    finally:
        for path in validated_paths:
            path.unlink(missing_ok=True)
