# SPDX-License-Identifier: Apache-2.0
"""Helpers for transferring large numpy arrays between local scheduler processes.

Offline diffusion requests can return multi-frame numpy arrays that are much larger
than the rest of the response metadata.  When the scheduler and client share a local
filesystem, passing a small file reference keeps the ZMQ pickle payload small while
preserving the same in-memory value after the client materializes the reference.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_MIN_FILE_REF_BYTES = 1 << 20


@dataclass
class NumpyArrayFileRef:
    path: str

    def materialize(self) -> np.ndarray:
        # A file ref has a single owner: the first client that materializes it.
        # Removing the file here keeps repeated offline generations from leaking
        # large frame arrays under /dev/shm or the system temp directory.
        try:
            return np.load(self.path, allow_pickle=False)
        finally:
            try:
                os.unlink(self.path)
            except FileNotFoundError:
                pass


def is_local_endpoint(endpoint: str) -> bool:
    # File refs are only safe when both processes can see the same filesystem.
    # For non-loopback TCP endpoints we keep the old inline-pickle behavior.
    return endpoint.startswith(
        ("tcp://127.0.0.1:", "tcp://localhost:", "ipc://", "inproc://")
    )


def spill_large_arrays_to_file_refs(value: Any) -> Any:
    # Preserve small arrays inline: writing tiny payloads to disk is slower and
    # makes ordinary control responses harder to inspect.
    if isinstance(value, np.ndarray) and value.nbytes >= _MIN_FILE_REF_BYTES:
        return _spill_array(value)
    if isinstance(value, list):
        return [spill_large_arrays_to_file_refs(item) for item in value]
    if isinstance(value, tuple):
        return tuple(spill_large_arrays_to_file_refs(item) for item in value)
    return value


def materialize_file_refs(value: Any) -> Any:
    if isinstance(value, NumpyArrayFileRef):
        return value.materialize()
    if isinstance(value, list):
        return [materialize_file_refs(item) for item in value]
    if isinstance(value, tuple):
        return tuple(materialize_file_refs(item) for item in value)
    return value


def _spill_array(array: np.ndarray) -> NumpyArrayFileRef:
    if not array.flags.c_contiguous:
        array = np.ascontiguousarray(array)

    directory = _array_ipc_dir()
    fd, path = tempfile.mkstemp(
        prefix="sgldiffusion-array-",
        suffix=".npy",
        dir=directory,
    )
    with os.fdopen(fd, "wb") as f:
        np.save(f, array, allow_pickle=False)
    return NumpyArrayFileRef(path=path)


def _array_ipc_dir() -> str | None:
    shm_path = Path("/dev/shm")
    if shm_path.is_dir() and os.access(shm_path, os.W_OK):
        return str(shm_path)
    # Returning None lets tempfile use the platform default temp directory.
    return None
