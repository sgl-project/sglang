import sys
from types import ModuleType, SimpleNamespace

from sglang_simulator.simulation.sglang.mem_pool_host import (
    _SIMULATED_AVAILABLE_HOST_MEMORY_BYTES,
    _call_with_meta_host_memory,
)


def test_meta_host_memory_supports_legacy_core_without_budget_scope(monkeypatch):
    for name in (
        "sglang",
        "sglang.srt",
        "sglang.srt.mem_cache",
        "sglang.srt.mem_cache.pool_host",
    ):
        package = ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.mem_cache.pool_host.base",
        ModuleType("sglang.srt.mem_cache.pool_host.base"),
    )

    fake_psutil = ModuleType("psutil")
    fake_psutil.virtual_memory = lambda: SimpleNamespace(available=1)
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    original_virtual_memory = fake_psutil.virtual_memory
    observed_available = []

    def legacy_init(_self):
        observed_available.append(fake_psutil.virtual_memory().available)
        return "initialized"

    assert _call_with_meta_host_memory(legacy_init, object()) == "initialized"
    assert observed_available == [_SIMULATED_AVAILABLE_HOST_MEMORY_BYTES]
    assert fake_psutil.virtual_memory is original_virtual_memory
