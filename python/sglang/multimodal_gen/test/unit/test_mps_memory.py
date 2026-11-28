from types import SimpleNamespace
from unittest.mock import Mock

import psutil
import pytest
import torch

from sglang.multimodal_gen.runtime.platforms.mps import MpsPlatform

GIB = 1 << 30


def test_total_memory_uses_metal_working_set(monkeypatch):
    get_total = MpsPlatform.get_device_total_memory
    get_total.cache_clear()
    try:
        monkeypatch.setattr(torch.mps, "recommended_max_memory", lambda: 0)
        with pytest.raises(RuntimeError, match="working-set limit"):
            get_total()
        monkeypatch.setattr(torch.mps, "recommended_max_memory", lambda: 16 * GIB)
        assert get_total() == 16 * GIB
    finally:
        get_total.cache_clear()


@pytest.mark.parametrize(
    "host_free,limit,allocated,empty_cache,expected",
    [
        (20, 16, 10, True, 6),
        (4, 16, 10, False, 4),
        (20, 16, 18, False, 0),
        (20, 0, 0, False, 0),
    ],
)
def test_available_memory_in_gib(
    monkeypatch, host_free, limit, allocated, empty_cache, expected
):
    clear = Mock()
    monkeypatch.setattr(torch.mps, "empty_cache", clear)
    monkeypatch.setattr(torch.mps, "recommended_max_memory", lambda: limit * GIB)
    monkeypatch.setattr(torch.mps, "driver_allocated_memory", lambda: allocated * GIB)
    monkeypatch.setattr(
        psutil, "virtual_memory", lambda: SimpleNamespace(available=host_free * GIB)
    )

    assert MpsPlatform.get_available_gpu_memory(empty_cache=empty_cache) == expected
    assert clear.call_count == int(empty_cache)
