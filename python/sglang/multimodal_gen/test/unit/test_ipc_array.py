# SPDX-License-Identifier: Apache-2.0

import tempfile
from pathlib import Path

import numpy as np
import pytest

from sglang.multimodal_gen.runtime import ipc_array
from sglang.multimodal_gen.runtime.ipc_array import (
    NumpyArrayFileRef,
    is_local_endpoint,
    materialize_file_refs,
    spill_large_arrays_to_file_refs,
)


def test_spill_large_arrays_round_trips_and_removes_file(monkeypatch, tmp_path):
    monkeypatch.setattr(ipc_array, "_array_ipc_dir", lambda: str(tmp_path))
    array = np.arange(ipc_array._MIN_FILE_REF_BYTES, dtype=np.uint8)

    spilled = spill_large_arrays_to_file_refs([array])

    assert isinstance(spilled[0], NumpyArrayFileRef)
    spilled_path = Path(spilled[0].path)
    assert spilled_path.exists()

    materialized = materialize_file_refs(spilled)

    assert np.array_equal(materialized[0], array)
    assert not spilled_path.exists()


def test_small_arrays_are_kept_inline():
    array = np.arange(16, dtype=np.uint8)

    spilled = spill_large_arrays_to_file_refs((array,))

    assert spilled[0] is array


def test_large_arrays_are_kept_inline_without_shm(monkeypatch):
    monkeypatch.setattr(ipc_array, "_array_ipc_dir", lambda: None)
    array = np.arange(ipc_array._MIN_FILE_REF_BYTES, dtype=np.uint8)

    spilled = spill_large_arrays_to_file_refs(array)

    assert spilled is array


def test_spill_removes_temp_file_when_save_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(ipc_array, "_array_ipc_dir", lambda: str(tmp_path))
    array = np.arange(ipc_array._MIN_FILE_REF_BYTES, dtype=np.uint8)
    created_paths = []

    def fail_save(*args, **kwargs):
        raise OSError("simulated write failure")

    original_mkstemp = tempfile.mkstemp

    def tracked_mkstemp(*args, **kwargs):
        fd, path = original_mkstemp(*args, **kwargs)
        created_paths.append(Path(path))
        return fd, path

    monkeypatch.setattr(tempfile, "mkstemp", tracked_mkstemp)
    monkeypatch.setattr(np, "save", fail_save)

    with pytest.raises(OSError, match="simulated write failure"):
        spill_large_arrays_to_file_refs(array)

    assert created_paths
    assert not created_paths[0].exists()


def test_local_endpoint_detection():
    assert is_local_endpoint("tcp://127.0.0.1:30000")
    assert is_local_endpoint("tcp://localhost:30000")
    assert is_local_endpoint("ipc:///tmp/sgl.sock")
    assert is_local_endpoint("inproc://scheduler")
    assert not is_local_endpoint("tcp://10.0.0.2:30000")
