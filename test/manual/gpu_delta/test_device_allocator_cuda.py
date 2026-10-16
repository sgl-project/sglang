"""Scoped DE allocation under the caller's unchanged Torch allocator.

Run this and test_codec.py in separate fresh processes for each ambient mode:
  PYTORCH_ALLOC_CONF=backend:native,expandable_segments:True
  PYTORCH_ALLOC_CONF=backend:cudaMallocAsync,expandable_segments:False
No allocator configuration is changed by these tests or the receiver.
"""

import gc
import weakref

import pytest
import torch

from sglang.srt.weight_sync.gpu_delta.memory import (
    _device_owners,
    device_empty,
    record_device_stream,
    require_de_capable,
)


def _allocator_state():
    backend = torch.cuda.memory.get_allocator_backend()
    settings = (
        dict(torch.cuda.memory._snapshot()["allocator_settings"])
        if backend == "native"
        else None
    )
    return backend, settings


def test_foreign_storage_leases_and_ambient_allocator_are_independent():
    device = torch.device("cuda:0")
    # Retain a normal large tensor before DE allocation and another afterwards.
    # The expandable mode must already have an actual expandable Torch segment.
    ordinary_before = torch.full((8 << 20,), 17, dtype=torch.uint8, device=device)
    before_pointer = ordinary_before.data_ptr()
    state = _allocator_state()
    if state[0] == "native" and state[1]["expandable_segments"]:
        assert any(
            entry["device"] == device.index and entry["is_expandable"]
            for entry in torch.cuda.memory._snapshot()["segments"]
        )
    allocated = torch.cuda.memory_allocated(device)
    tensor = device_empty((1024, 1024), dtype=torch.uint8, device=device)
    pointer = tensor.data_ptr()
    assert pointer % 256 == 0
    require_de_capable(pointer)
    assert torch.cuda.memory_allocated(device) == allocated
    owner = weakref.ref(_device_owners[pointer])
    view = tensor.view(-1)[16:]
    del tensor
    gc.collect()
    assert owner() is not None
    stream = torch.cuda.Stream(device=device)
    record_device_stream(view, stream)
    with torch.cuda.stream(stream):
        torch.cuda._sleep(1000000)
        view.fill_(91)
        completed = torch.cuda.Event()
        completed.record(stream)
    del view
    gc.collect()
    assert owner() is None and completed.query()
    # Zero-sized views use Torch's empty storage; retain this native contract
    # alongside the foreign-storage lifetime and allocator-isolation checks.
    for dtype in (torch.uint8, torch.int32, torch.int64):
        empty = device_empty((0, 13), dtype=dtype, device=device)
        assert empty.shape == (0, 13) and empty.numel() == 0
        assert empty.dtype == dtype and empty.device == device
    ordinary_after = torch.full_like(ordinary_before, 23)
    assert _allocator_state() == state
    assert ordinary_before.data_ptr() == before_pointer
    assert bool((ordinary_before == 17).all())
    assert bool((ordinary_after == 23).all())
    if state[0] == "native" and state[1]["expandable_segments"]:
        snapshot = torch.cuda.memory._snapshot()
        for pointer in (before_pointer, ordinary_after.data_ptr()):
            assert any(
                entry["device"] == device.index
                and entry["is_expandable"]
                and entry["address"] <= pointer < entry["address"] + entry["total_size"]
                for entry in snapshot["segments"]
            )


def test_allocation_and_finalization_restore_callers_cuda_context():
    from sglang.srt.weight_sync.gpu_delta.memory import _driver

    assert torch.cuda.device_count() >= 2, "context restoration test requires two GPUs"
    torch.cuda.set_device(0)
    torch.cuda.current_stream(0)
    driver = _driver()
    status, original = driver.cuCtxGetCurrent()
    assert not status and int(original)
    value = device_empty(1024, dtype=torch.uint8, device="cuda:1")
    assert torch.cuda.current_device() == 0
    assert int(driver.cuCtxGetCurrent()[1]) == int(original)
    stream = torch.cuda.Stream(device=1)
    with pytest.raises(ValueError, match="stream/device mismatch"):
        record_device_stream(value, torch.cuda.current_stream(0))
    record_device_stream(value, stream)
    with torch.cuda.stream(stream):
        value.fill_(31)
        completed = torch.cuda.Event()
        completed.record(stream)
    owner = weakref.ref(_device_owners[value.data_ptr()])
    del value
    gc.collect()
    assert owner() is None and completed.query()
    assert torch.cuda.current_device() == 0
    assert int(driver.cuCtxGetCurrent()[1]) == int(original)
