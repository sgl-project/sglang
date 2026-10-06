# SPDX-License-Identifier: Apache-2.0

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from sglang.multimodal_gen.runtime import ipc_array
from sglang.multimodal_gen.runtime.ipc_array import (
    NumpyArrayFileRef,
    TorchTensorFileRef,
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

    # A failed spill falls back to sending the payload inline.
    assert spill_large_arrays_to_file_refs(array) is array

    assert created_paths
    assert not created_paths[0].exists()


def test_local_endpoint_detection():
    assert is_local_endpoint("tcp://127.0.0.1:30000")
    assert is_local_endpoint("tcp://localhost:30000")
    assert is_local_endpoint("ipc:///tmp/sgl.sock")
    assert is_local_endpoint("inproc://scheduler")
    assert not is_local_endpoint("tcp://10.0.0.2:30000")


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.uint8])
def test_spill_large_tensors_round_trips(monkeypatch, tmp_path, dtype):
    monkeypatch.setattr(ipc_array, "_array_ipc_dir", lambda: str(tmp_path))
    elements = (
        ipc_array._MIN_FILE_REF_BYTES // torch.empty((), dtype=dtype).element_size()
    )
    tensor = torch.arange(elements, dtype=torch.int64).to(dtype).reshape(2, -1)

    spilled = spill_large_arrays_to_file_refs([tensor])

    assert isinstance(spilled[0], TorchTensorFileRef)
    spilled_path = Path(spilled[0].ref.path)
    assert spilled_path.exists()

    materialized = materialize_file_refs(spilled)[0]

    assert materialized.dtype == tensor.dtype
    assert materialized.shape == tensor.shape
    assert torch.equal(materialized, tensor)
    assert not spilled_path.exists()


def test_non_contiguous_tensor_round_trips(monkeypatch, tmp_path):
    monkeypatch.setattr(ipc_array, "_array_ipc_dir", lambda: str(tmp_path))
    elements = ipc_array._MIN_FILE_REF_BYTES // 2
    tensor = torch.arange(elements, dtype=torch.float32).reshape(2, -1).T

    materialized = materialize_file_refs(spill_large_arrays_to_file_refs(tensor))

    assert torch.equal(materialized, tensor)


def test_small_tensors_are_kept_inline():
    tensor = torch.zeros(16)

    spilled = spill_large_arrays_to_file_refs((tensor,))

    assert spilled[0] is tensor


def test_tensor_spill_falls_back_inline_when_shm_is_full(monkeypatch, tmp_path):
    monkeypatch.setattr(ipc_array, "_array_ipc_dir", lambda: str(tmp_path))
    tensor = torch.zeros(ipc_array._MIN_FILE_REF_BYTES, dtype=torch.uint8)

    def fail_save(*args, **kwargs):
        raise OSError("No space left on device")

    monkeypatch.setattr(np, "save", fail_save)

    assert spill_large_arrays_to_file_refs(tensor) is tensor
