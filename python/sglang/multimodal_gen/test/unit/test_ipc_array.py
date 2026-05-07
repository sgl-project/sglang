# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np

from sglang.multimodal_gen.runtime.ipc_array import (
    NumpyArrayFileRef,
    is_local_endpoint,
    materialize_file_refs,
    spill_large_arrays_to_file_refs,
)


def test_spill_large_arrays_round_trips_and_removes_file():
    array = np.arange(1024 * 1024, dtype=np.uint8).reshape(1024, 1024)

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


def test_local_endpoint_detection():
    assert is_local_endpoint("tcp://127.0.0.1:30000")
    assert is_local_endpoint("tcp://localhost:30000")
    assert is_local_endpoint("ipc:///tmp/sgl.sock")
    assert is_local_endpoint("inproc://scheduler")
    assert not is_local_endpoint("tcp://10.0.0.2:30000")
