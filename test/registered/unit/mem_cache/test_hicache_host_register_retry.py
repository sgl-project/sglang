"""Unit tests for cudaHostRegister chunk degradation in pool_host/common.py.

Large single-shot cudaHostRegister calls were observed to fail intermittently
with cudaErrorInvalidValue in processes under heavy GPU memory usage, while
smaller registrations succeed. _register_chunk_with_retry halves the chunk
until the driver accepts it instead of crashing the scheduler at startup.
"""

from sglang.srt.mem_cache.pool_host.common import (
    _cuda_host_register,
    _register_chunk_with_retry,
)

import pytest
import torch

MiB = 1024**2


class _FakeCudart:
    """Mimics the pybind cudart surface; fails chunks larger than fail_above."""

    def __init__(self, fail_above: int):
        self.fail_above = fail_above
        self.register_calls = []
        self.unregister_calls = []

    def cudaHostRegister(self, ptr, size, flags):
        self.register_calls.append((ptr, size))
        return 0 if size <= self.fail_above else 1

    def cudaHostUnregister(self, ptr):
        self.unregister_calls.append(ptr)
        return 0

    def cudaGetErrorString(self, rc):
        return "invalid argument"


class _FakeBuffer:
    def __init__(self, total: int):
        self.total = total

    def data_ptr(self) -> int:
        return 1 * 1024**3

    def numel(self) -> int:
        return self.total

    def element_size(self) -> int:
        return 1


def test_retry_succeeds_on_first_try():
    cudart = _FakeCudart(fail_above=64 * MiB)
    size = _register_chunk_with_retry(
        cudart, ptr=0x1000, size=32 * MiB, offset=0, total=32 * MiB
    )
    assert size == 32 * MiB
    assert cudart.register_calls == [(0x1000, 32 * MiB)]


def test_retry_halves_until_driver_accepts():
    cudart = _FakeCudart(fail_above=4 * MiB)
    size = _register_chunk_with_retry(
        cudart, ptr=0x1000, size=32 * MiB, offset=0, total=32 * MiB
    )
    assert size == 4 * MiB
    assert [size for _, size in cudart.register_calls] == [
        32 * MiB,
        16 * MiB,
        8 * MiB,
        4 * MiB,
    ]


def test_retry_raises_with_context_when_nothing_works():
    cudart = _FakeCudart(fail_above=0)
    with pytest.raises(RuntimeError) as exc_info:
        _register_chunk_with_retry(
            cudart, ptr=0x1000, size=32 * MiB, offset=64, total=128 * MiB
        )
    message = str(exc_info.value)
    assert "rc=1" in message
    assert "offset=64" in message
    assert f"total={128 * MiB}" in message


def test_cuda_host_register_degrades_and_covers_whole_buffer(monkeypatch):
    # 256 GB default chunk limit does not chunk this buffer up front; the
    # driver rejects anything above 8 MiB, so chunks must degrade to 8 MiB.
    cudart = _FakeCudart(fail_above=8 * MiB)
    monkeypatch.setattr(torch.cuda, "cudart", lambda: cudart)

    buffer = _FakeBuffer(total=64 * MiB)
    _cuda_host_register(buffer)

    ranges = getattr(buffer, "_sglang_cuda_host_registered_ranges")
    assert ranges[0][0] == buffer.data_ptr()
    assert sum(size for _, size in ranges) == 64 * MiB
    assert all(size <= 8 * MiB for _, size in ranges)
    # Ranges are contiguous: each starts where the previous one ended.
    for (prev_ptr, prev_size), (next_ptr, _) in zip(ranges, ranges[1:]):
        assert prev_ptr + prev_size == next_ptr
