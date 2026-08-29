# SPDX-License-Identifier: Apache-2.0
"""Helpers for transferring large numpy arrays between local scheduler processes."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_MIN_FILE_REF_BYTES = 32 << 20


@dataclass
class NumpyArrayFileRef:
    path: str

    def materialize(self) -> np.ndarray:
        try:
            return np.load(self.path, allow_pickle=False)
        finally:
            try:
                os.unlink(self.path)
            except FileNotFoundError:
                pass


def is_local_endpoint(endpoint: str) -> bool:
    return endpoint.startswith(
        ("tcp://127.0.0.1:", "tcp://localhost:", "ipc://", "inproc://")
    )


def spill_large_arrays_to_file_refs(value: Any) -> Any:
    directory = _array_ipc_dir()
    if directory is None:
        return value
    return _spill_large_arrays_to_file_refs(value, directory)


def _spill_large_arrays_to_file_refs(value: Any, directory: str) -> Any:
    if isinstance(value, np.ndarray) and value.nbytes >= _MIN_FILE_REF_BYTES:
        # only spill if the array size is above the threshold. if not, it's not worth it
        return _spill_array(value, directory)
    if isinstance(value, list):
        return [_spill_large_arrays_to_file_refs(item, directory) for item in value]
    if isinstance(value, tuple):
        return tuple(
            _spill_large_arrays_to_file_refs(item, directory) for item in value
        )
    return value


def materialize_file_refs(value: Any) -> Any:
    if isinstance(value, NumpyArrayFileRef):
        return value.materialize()
    if isinstance(value, list):
        return [materialize_file_refs(item) for item in value]
    if isinstance(value, tuple):
        return tuple(materialize_file_refs(item) for item in value)
    return value


def _spill_array(array: np.ndarray, directory: str) -> NumpyArrayFileRef:
    if not array.flags.c_contiguous:
        array = np.ascontiguousarray(array)

    fd, path = tempfile.mkstemp(
        prefix="sgldiffusion-array-",
        suffix=".npy",
        dir=directory,
    )
    try:
        with os.fdopen(fd, "wb") as f:
            np.save(f, array, allow_pickle=False)
    except Exception:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
        raise
    return NumpyArrayFileRef(path=path)


def _array_ipc_dir() -> str | None:
    shm_path = Path("/dev/shm")
    if shm_path.is_dir() and os.access(shm_path, os.W_OK):
        return str(shm_path)
    return None
