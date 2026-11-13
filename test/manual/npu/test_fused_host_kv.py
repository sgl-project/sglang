import pytest
import torch

pytest.importorskip("torch_npu")

from sgl_kernel_npu.sparsity_driven_kv_offload.fused_shm import (
    free_fused_shm as free_shm,
)

from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.fused_manager import (
    FusedSparseKVCacheManager,
    ShmSplitKVPool,
)


def test_paged_host_overlap_move_real_npu():
    from types import SimpleNamespace

    from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMLATokenToKVPool

    if not torch.npu.is_available():
        pytest.skip("NPU is required")
    device_id = torch.npu.current_device()
    manager = FusedSparseKVCacheManager.__new__(FusedSparseKVCacheManager)
    manager.device = torch.device("npu", device_id)
    manager.head_num, manager.head_dim = 1, 576
    manager.store_dtype = torch.bfloat16
    manager.kv_lora_rank, manager.qk_rope_head_dim = 512, 64
    manager.host_pool = ShmSplitKVPool(
        shape=(1, 512, 1, 576),
        widths=(512, 64),
        layers=1,
        dtype=torch.bfloat16,
        device_id=device_id,
        name="prefix_shared_overlap_move_test",
    )
    manager.host_k_buffer, manager.host_v_buffer = manager.host_pool.buffers

    def write_host(value):
        manager.host_k_buffer[0][0].copy_(value[..., :512])
        manager.host_v_buffer[0][0].copy_(value[..., 512:])

    def read_host():
        return torch.cat(
            (manager.host_k_buffer[0][0], manager.host_v_buffer[0][0]), dim=-1
        )

    pool = SimpleNamespace(size=512, page_size=0, sparse_kv_manager=manager)
    original = (
        torch.arange(512, dtype=torch.float32)
        .remainder(127)
        .to(torch.bfloat16)[:, None, None]
        .expand(512, 1, 576)
        .clone()
    )
    write_host(original)
    source = torch.tensor([40, 41, 42], device=manager.device)
    target = torch.tensor([41, 42, 40], device=manager.device)
    try:
        NPUMLATokenToKVPool.move_kv_cache(pool, target, source)
        torch.npu.synchronize()
        torch.testing.assert_close(
            read_host()[[41, 42, 40]], original[[40, 41, 42]], rtol=0, atol=0
        )
        write_host(original)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            NPUMLATokenToKVPool.move_kv_cache(pool, target, source)
        torch.npu.synchronize()
        write_host(original)
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(
            read_host()[[41, 42, 40]], original[[40, 41, 42]], rtol=0, atol=0
        )
    finally:
        torch.npu.synchronize()
        free_shm(device_id)
