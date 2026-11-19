"""A HiSparse swap-in must reach the buffers of the layer that asked for it.

Attention backends pass a model layer id, while the device and host buffers
hold only the layers of this pipeline-parallel stage, or only the
full-attention layers of a hybrid model.
"""

import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.managers import hisparse_coordinator
from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
from sglang.srt.mem_cache.allocator.hisparse import HiSparseTokenToKVPoolAllocator
from sglang.srt.mem_cache.deepseek_v4_memory_pool import HiSparseC4DevicePool
from sglang.srt.mem_cache.hisparse_memory_pool import HiSparseDSATokenToKVPool
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _device_pool(pool_type, *, start_layer, layer_num):
    pool = object.__new__(pool_type)
    pool.start_layer = start_layer
    pool.layer_num = layer_num
    pool.kv_buffer = [torch.empty(1) for _ in range(layer_num)]
    return pool


def _coordinator(kvcache, *, is_dsv4=False):
    allocator = object.__new__(HiSparseTokenToKVPoolAllocator)
    allocator._kvcache = kvcache
    device_pool = kvcache if is_dsv4 else allocator.get_hisparse_kvcache()
    layer_num = device_pool.layer_num
    # Each per-layer row holds its own buffer index, so a routed slice names its layer.
    per_layer = torch.arange(layer_num).view(layer_num, 1, 1)
    coordinator = object.__new__(HiSparseCoordinator)
    coordinator.is_dsv4_hisparse = is_dsv4
    coordinator.is_m3_hisparse = False
    coordinator.mem_pool_device = device_pool
    coordinator.mem_pool_host = SimpleNamespace(
        kv_buffer=[torch.empty(1) for _ in range(layer_num)]
    )
    coordinator._pool_layer_index = hisparse_coordinator._resolve_pool_layer_index(
        kvcache=kvcache,
        device_pool=device_pool,
        callers_pass_model_layer_ids=not is_dsv4,
    )
    coordinator.enable_prefetch = False
    coordinator.skip_io = False
    coordinator.top_k = 2
    coordinator.device_buffer_size = 4
    coordinator.swap_in_block_size = 1
    coordinator.item_size_bytes = 1
    coordinator.num_real_reqs = torch.ones(1, dtype=torch.int32)
    coordinator.top_k_device_locs_buffer = torch.zeros((1, 2), dtype=torch.int32)
    coordinator.req_to_host_pool = torch.zeros((1, 4), dtype=torch.int64)
    coordinator.req_device_buffer_tokens = per_layer.clone()
    coordinator.req_device_buffer_token_locs = per_layer.clone()
    coordinator.lru_slots = per_layer.clone()
    return coordinator


def _swapped_in_buffer_layer(coordinator, layer_id):
    """Run one swap-in up to the kernel and return the buffer layer it was handed."""
    kernel_name = (
        "load_cache_to_device_buffer_dsv4_mla"
        if coordinator.is_dsv4_hisparse
        else "load_cache_to_device_buffer_mla"
    )
    with patch.object(hisparse_coordinator, kernel_name) as kernel:
        coordinator.swap_in_selected_pages(
            req_pool_indices=torch.zeros(1, dtype=torch.int64),
            compressed_seq_lens=torch.ones(1, dtype=torch.int64),
            top_k_result=torch.zeros((1, 2), dtype=torch.int32),
            layer_id=layer_id,
        )
    buffers = kernel.call_args.kwargs
    device_layers = [
        i
        for i, buffer in enumerate(coordinator.mem_pool_device.kv_buffer)
        if buffer is buffers["device_buffer"]
    ]
    host_layers = [
        i
        for i, buffer in enumerate(coordinator.mem_pool_host.kv_buffer)
        if buffer is buffers["host_cache"]
    ]
    state_layers = {
        int(buffers[name].item())
        for name in ("device_buffer_tokens", "device_buffer_locs", "lru_slots")
    }
    assert device_layers == host_layers and state_layers == set(device_layers)
    return device_layers[0]


@pytest.mark.parametrize("start_layer,layer_num", [(0, 10), (10, 10), (70, 8)])
def test_pp_stage_swaps_into_its_own_layers(start_layer, layer_num):
    """A stage past the first one indexed its buffers by model layer id: out of
    range, or another layer's KV."""
    coordinator = _coordinator(
        _device_pool(
            HiSparseDSATokenToKVPool, start_layer=start_layer, layer_num=layer_num
        )
    )
    for buffer_layer in range(layer_num):
        assert (
            _swapped_in_buffer_layer(coordinator, start_layer + buffer_layer)
            == buffer_layer
        )
    outside = [start_layer + layer_num] + ([start_layer - 1] if start_layer else [])
    for layer_id in outside:
        with pytest.raises(LookupError):
            _swapped_in_buffer_layer(coordinator, layer_id)


@pytest.mark.parametrize(
    "stage_start,full_attention_layer_ids",
    [(0, [3, 7, 11, 15]), (40, [41, 45, 49])],
)
def test_hybrid_pool_swaps_into_its_dense_full_pool(
    stage_start, full_attention_layer_ids
):
    """The dense full pool starts at 0 whatever the stage, so a stage offset
    applied on top of the hybrid mapping rejects or misroutes every layer."""
    hybrid = object.__new__(HybridLinearKVPool)
    hybrid.start_layer = stage_start
    hybrid.full_kv_pool = _device_pool(
        HiSparseDSATokenToKVPool,
        start_layer=0,
        layer_num=len(full_attention_layer_ids),
    )
    hybrid.full_attention_layer_id_mapping = {
        layer_id: i for i, layer_id in enumerate(full_attention_layer_ids)
    }
    coordinator = _coordinator(hybrid)
    for buffer_layer, layer_id in enumerate(full_attention_layer_ids):
        assert _swapped_in_buffer_layer(coordinator, layer_id) == buffer_layer
    with pytest.raises(LookupError):
        _swapped_in_buffer_layer(coordinator, full_attention_layer_ids[0] + 1)


def test_dsv4_compressed_layer_index_is_not_shifted():
    """DeepSeek-V4 callers already pass the compressed pool's layer index."""
    coordinator = _coordinator(
        _device_pool(HiSparseC4DevicePool, start_layer=10, layer_num=4), is_dsv4=True
    )
    for compress_layer_id in range(4):
        assert (
            _swapped_in_buffer_layer(coordinator, compress_layer_id)
            == compress_layer_id
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
