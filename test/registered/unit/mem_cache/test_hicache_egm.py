import gc
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.mem_cache.pool_host import base, common, egm
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-c", runner_config="2-gpu-gb300")


def test_egm_flag_selects_host_allocator(monkeypatch):
    monkeypatch.setattr(
        common, "get_memory", lambda: SimpleNamespace(hicache_storage_backend=None)
    )
    with envs.SGLANG_HICACHE_EGM.override(True):
        assert common.get_allocator_type() == "egm"
        assert common.get_allocator_from_storage("egm").supports_hugetlb() is False


def test_egm_flag_rejects_storage_backend(monkeypatch):
    monkeypatch.setattr(
        common,
        "get_memory",
        lambda: SimpleNamespace(hicache_storage_backend="mooncake"),
    )
    with envs.SGLANG_HICACHE_EGM.override(True):
        with pytest.raises(ValueError, match="storage backends"):
            common.get_allocator_type()


def test_preflight_accounts_for_vmm_granularity(monkeypatch):
    monkeypatch.setattr(egm, "_allocation_properties", lambda device: (None, 0, 1024))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    with base.host_memory_budget_scope(4096):
        first = base.host_memory_preflight(
            [1, 1025], egm.EgmHostTensorAllocator(), "cuda"
        )
        second = base.host_memory_preflight(
            [1025], egm.EgmHostTensorAllocator(), "cuda"
        )
    assert first.allocation_bytes == 3072 and first.available_bytes == 4096
    assert second.allocation_bytes == 2048 and second.available_bytes == 1024


def test_runtime_storage_attach_rejects_egm_pool():
    controller = HybridCacheController.__new__(HybridCacheController)
    pool = SimpleNamespace(allocator=egm.EgmHostTensorAllocator())
    controller.mem_pool_host = SimpleNamespace(
        entry_map={"kv": SimpleNamespace(host_pool=pool)}
    )
    with pytest.raises(ValueError, match="EGM host pools"):
        controller.attach_storage_backend("mooncake")


@pytest.mark.parametrize(
    "allocator_type,io_backend",
    [("egm", "direct"), ("egm", "kernel"), ("default", "kernel")],
)
def test_egm_hicache_pool_round_trip(allocator_type, io_backend):
    device = MHATokenToKVPool(
        size=16,
        page_size=1,
        dtype=torch.float16,
        head_num=2,
        head_dim=64,
        layer_num=2,
        device="cuda:0",
        enable_memory_saver=False,
    )
    host = MHATokenToKVPoolHost(
        buffer_info=device.get_device_pool_infos()[0].buffer_info,
        host_to_device_ratio=2.0,
        host_size=0,
        page_size=1,
        layout="layer_first",
        pin_memory=True,
        allocator_type=allocator_type,
        device_capacity_tokens=device.size,
        start_layer=device.start_layer,
        end_layer=device.end_layer,
    )
    allocator = host.allocator
    try:
        for layer, (k, v) in enumerate(zip(device.k_buffer, device.v_buffer)):
            k.fill_(layer + 3)
            v.fill_(layer + 7)
        ids = torch.arange(4, device="cuda:0", dtype=torch.int64)
        host.backup_from_device_all_layer(device, ids, ids, io_backend)
        torch.cuda.synchronize()
        for k, v in zip(device.k_buffer, device.v_buffer):
            k.zero_()
            v.zero_()
        for layer in range(2):
            host.load_to_device_per_layer(device, ids, ids, layer, io_backend)
        torch.cuda.synchronize()
        for layer, (k, v) in enumerate(zip(device.k_buffer, device.v_buffer)):
            assert torch.all(k[ids] == layer + 3)
            assert torch.all(v[ids] == layer + 7)
    finally:
        host.destroy()
    del host
    gc.collect()
    if allocator_type == "egm":
        assert not allocator.allocations
