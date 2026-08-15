"""Shared-memory tensor transport: the shm rung of the TensorRef ladder.

A ref is a plain msgpack-safe dict: {"transport": "shm", "name": str,
"dtype": str, "shape": [int, ...]}. SRT only attaches, reads or writes, and
closes peer-owned segments; it never creates or unlinks them.
"""

from __future__ import annotations

import math
import re
from contextlib import suppress
from multiprocessing import resource_tracker, shared_memory
from typing import Any

import numpy as np
import torch

SHM_TRANSPORT = "shm"
_SHM_NAME_RE = re.compile(
    r"sgl_shm_[A-Za-z0-9][A-Za-z0-9_-]{0,31}_[1-9][0-9]{0,9}_[0-9a-f]{8}"
)
_SHM_DTYPES = frozenset(
    f"{kind}{size}" for kind in ("int", "uint") for size in (8, 16, 32, 64)
) | {"float16", "float32", "float64"}


def _untrack(segment: shared_memory.SharedMemory) -> None:
    """https://stackoverflow.com/q/62748654: SharedMemory registers every
    attachment with the process-local resource tracker, which may unlink it
    at process exit. Lifecycle here belongs to the peer, so drop the local
    attachment registration."""
    with suppress(Exception):
        resource_tracker.unregister(segment._name, "shared_memory")


def is_shm_ref(value: Any) -> bool:
    return isinstance(value, dict) and value.get("transport") == SHM_TRANSPORT


def validate_shm_tensor_ref(
    ref: dict[str, Any], *, ndim: int | None = None
) -> tuple[tuple[int, ...], np.dtype]:
    """Validate untrusted TensorRef metadata before opening its segment."""
    if (
        not is_shm_ref(ref)
        or not isinstance(ref.get("name"), str)
        or _SHM_NAME_RE.fullmatch(ref["name"]) is None
    ):
        raise ValueError(f"malformed shm tensor ref: {ref!r}")
    shape = ref.get("shape")
    if (
        not isinstance(shape, list)
        or not shape
        or (ndim is not None and len(shape) != ndim)
        or any(type(dim) is not int or dim <= 0 for dim in shape)
    ):
        raise ValueError(f"malformed shm tensor shape: {shape!r}")
    dtype_name = ref.get("dtype")
    if not isinstance(dtype_name, str):
        raise ValueError(f"unsupported shm tensor dtype: {dtype_name!r}")
    try:
        dtype = np.dtype(dtype_name)
    except (TypeError, ValueError) as e:
        raise ValueError(f"unsupported shm tensor dtype: {dtype_name!r}") from e
    if dtype.name not in _SHM_DTYPES:
        raise ValueError(f"unsupported shm tensor dtype: {dtype_name!r}")
    return tuple(shape), dtype


def _validate_segment_size(
    segment: shared_memory.SharedMemory,
    shape: tuple[int, ...],
    dtype: np.dtype,
    *,
    exact: bool = False,
) -> None:
    required_bytes = math.prod(shape) * dtype.itemsize
    if exact and required_bytes != segment.size:
        raise ValueError(
            f"shm tensor needs exactly {required_bytes} bytes, "
            f"segment has {segment.size}"
        )
    if not exact and required_bytes > segment.size:
        raise ValueError(
            f"shm tensor needs {required_bytes} bytes, segment has {segment.size}"
        )


def read_shm_tensor(ref: dict[str, Any]) -> np.ndarray:
    """Copy a peer-created segment out into process-local memory."""
    shape, dtype = validate_shm_tensor_ref(ref)
    segment = shared_memory.SharedMemory(name=ref["name"])
    _untrack(segment)
    try:
        _validate_segment_size(segment, shape, dtype)
        view = np.ndarray(shape, dtype=dtype, buffer=segment.buf)
        return view.copy()
    finally:
        segment.close()


def validate_shm_tensor_buffer(
    ref: dict[str, Any],
    *,
    shape: tuple[int, ...],
    dtype: str,
) -> None:
    """Validate and open a peer-owned output buffer without taking ownership."""
    actual_shape, actual_dtype = validate_shm_tensor_ref(ref, ndim=len(shape))
    expected_dtype = np.dtype(dtype)
    if actual_shape != shape:
        raise ValueError(
            f"shm tensor shape {actual_shape} does not match expected shape {shape}"
        )
    if actual_dtype != expected_dtype:
        raise ValueError(
            f"shm tensor dtype {actual_dtype.name!r} does not match "
            f"expected dtype {expected_dtype.name!r}"
        )
    segment = shared_memory.SharedMemory(name=ref["name"])
    _untrack(segment)
    try:
        _validate_segment_size(segment, actual_shape, actual_dtype, exact=True)
    finally:
        segment.close()


def write_shm_tensor_buffer(ref: dict[str, Any], tensor: np.ndarray) -> None:
    """Write a tensor into a peer-owned segment without unlinking it."""
    tensor = np.asarray(tensor)
    shape, dtype = validate_shm_tensor_ref(ref, ndim=tensor.ndim)
    if tensor.shape != shape:
        raise ValueError(
            f"tensor shape {tensor.shape} does not match shm buffer shape {shape}"
        )
    if tensor.dtype != dtype:
        raise ValueError(
            f"tensor dtype {tensor.dtype.name!r} does not match "
            f"shm buffer dtype {dtype.name!r}"
        )
    segment = shared_memory.SharedMemory(name=ref["name"])
    _untrack(segment)
    try:
        _validate_segment_size(segment, shape, dtype, exact=True)
        np.ndarray(shape, dtype=dtype, buffer=segment.buf)[...] = tensor
    finally:
        segment.close()


def package_hidden_states(
    chunks: list[torch.Tensor], *, output_buffer: dict[str, Any]
) -> dict[str, Any]:
    """Write hidden states to a caller-owned [rows, hidden] buffer."""
    rows = torch.cat(
        [chunk if chunk.dim() == 2 else chunk.unsqueeze(0) for chunk in chunks]
    )
    write_shm_tensor_buffer(output_buffer, rows.float().numpy())
    return output_buffer
