import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import torch
import torch.multiprocessing as mp

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-c", runner_config="4-gpu-b200")

PORT = 29721
POOL_PORT = 29722
SLAB_PORT = 29723


def _destroy_distributed() -> None:
    from sglang.srt.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
    )

    destroy_model_parallel()
    destroy_distributed_environment()


def _run_rank_major_vmm(rank: int, world_size: int, port: int):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    torch.cuda.set_device(rank)

    from sglang.srt.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from sglang.srt.mem_cache.dsa_cache_shared import (
        create_rank_major_shared_tensor,
    )
    from sglang.srt.runtime_context import get_parallel

    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=rank,
        distributed_init_method=f"tcp://127.0.0.1:{port}",
        backend="nccl",
    )
    initialize_model_parallel(
        tensor_model_parallel_size=world_size,
        attention_context_model_parallel_size=world_size,
    )
    cpu_group = get_parallel().attn_cp_group.cpu_group

    for iteration in range(2):
        allocation = create_rank_major_shared_tensor(
            (64, 1, 8),
            dtype=torch.uint8,
            cpu_group=cpu_group,
            first_dim_multiple=64,
        )
        assert allocation.global_view.device.index == rank
        assert allocation.local_view.device.index == rank
        assert allocation.rank_local_view.device.index == rank
        allocation.local_view.fill_(rank + iteration * world_size + 1)
        torch.cuda.synchronize()
        torch.distributed.barrier(group=cpu_group)

        for owner_rank in range(world_size):
            start = owner_rank * allocation.local_rows
            segment = allocation.global_view.narrow(0, start, allocation.local_rows)
            expected = owner_rank + iteration * world_size + 1
            assert torch.all(segment == expected).item()

            rank_local_segment = (owner_rank - rank) % world_size
            start = rank_local_segment * allocation.local_rows
            segment = allocation.rank_local_view.narrow(0, start, allocation.local_rows)
            assert torch.all(segment == expected).item()

        allocation.close()
        torch.distributed.barrier(group=cpu_group)
    _destroy_distributed()


def _run_rank_major_slab(rank: int, world_size: int, port: int):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    torch.cuda.set_device(rank)

    from sglang.srt.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from sglang.srt.mem_cache.dsa_cache_shared import (
        create_rank_major_shared_slab,
    )
    from sglang.srt.runtime_context import get_parallel

    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=rank,
        distributed_init_method=f"tcp://127.0.0.1:{port}",
        backend="nccl",
    )
    initialize_model_parallel(
        tensor_model_parallel_size=world_size,
        attention_context_model_parallel_size=world_size,
    )
    cpu_group = get_parallel().attn_cp_group.cpu_group

    slab = create_rank_major_shared_slab(
        (64, 1, 8),
        layer_num=3,
        dtype=torch.uint8,
        cpu_group=cpu_group,
        first_dim_multiple=64,
    )
    assert len(slab.global_views) == 3
    assert len(slab.rank_local_views) == 3
    assert len(slab.local_views) == 3
    assert slab.layer_rows == 64
    assert slab.global_views[1].data_ptr() - slab.global_views[0].data_ptr() == 512

    for layer_id, local_view in enumerate(slab.local_views):
        local_view.fill_(100 * layer_id + rank + 1)
    torch.cuda.synchronize()
    torch.distributed.barrier(group=cpu_group)

    rank_stride = slab.allocation.local_rows
    for layer_id in range(3):
        for owner_rank in range(world_size):
            expected = 100 * layer_id + owner_rank + 1
            assert torch.all(
                slab.global_views[layer_id][owner_rank * rank_stride] == expected
            ).item()
            rank_local_owner = (owner_rank - rank) % world_size
            assert torch.all(
                slab.rank_local_views[layer_id][rank_local_owner * rank_stride]
                == expected
            ).item()

    slab.close()
    torch.distributed.barrier(group=cpu_group)
    _destroy_distributed()


def _run_shared_pool(rank: int, world_size: int, port: int):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    torch.cuda.set_device(rank)

    from sglang.jit_kernel.fused_store_index_cache import (
        fused_store_index_k_cache,
    )
    from sglang.kernels.ops.attention.dsa import index_buf_accessor
    from sglang.srt.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from sglang.srt.mem_cache.dsa_cache_shared import SharedDSATokenToKVPool

    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=rank,
        distributed_init_method=f"tcp://127.0.0.1:{port}",
        backend="nccl",
    )
    initialize_model_parallel(
        tensor_model_parallel_size=world_size,
        attention_context_model_parallel_size=world_size,
    )

    pool = SharedDSATokenToKVPool(
        128,
        page_size=64,
        kv_lora_rank=512,
        dtype=torch.bfloat16,
        qk_rope_head_dim=64,
        layer_num=2,
        device=f"cuda:{rank}",
        index_head_dim=128,
        enable_memory_saver=False,
        kv_cache_dim=576,
        shared_rank=rank,
        shared_size=world_size,
    )
    assert len(pool.kv_buffer) == 2
    assert len(pool.index_k_with_scale_buffer) == 2
    assert pool.shared_kv_slab.rank_local_views == []
    assert len(pool.shared_index_slab.rank_local_views) == 2
    assert pool.main_layout.cp_size == world_size
    assert pool.index_layout.cp_size == world_size
    assert pool.kv_buffer[0].device.index == rank
    assert pool.index_k_with_scale_buffer[0].device.index == rank

    for layer_id in range(2):
        pool.local_kv_buffer[layer_id].fill_(100 * layer_id + rank + 1)
        pool.local_index_k_with_scale_buffer[layer_id].fill_(100 * layer_id + rank + 11)
    torch.cuda.synchronize()
    torch.distributed.barrier(group=pool.shared_cp_group.cpu_group)

    from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager

    manager = MooncakeKVManager.__new__(MooncakeKVManager)
    captured = []
    manager._transfer_data = MagicMock(
        side_effect=lambda _session, _blocks: captured.append(staging.buffer.clone())
        or 0
    )

    def check_staged_bytes(buffer, item_len):
        nonlocal captured, staging
        captured = []
        staging = SimpleNamespace(
            buffer=torch.empty(item_len * 2, dtype=torch.uint8, device=buffer.device),
            get_ptr=lambda: 1000,
            get_size=lambda: item_len * 2,
        )
        indices = np.array([1, 0], dtype=np.int32)
        MooncakeKVManager._send_dsa_shared_staged(
            manager,
            "session",
            [buffer],
            [item_len],
            indices,
            [2000],
            np.array([3, 5], dtype=np.int32),
            staging,
        )
        expected = (
            buffer.view(torch.uint8)
            .reshape(-1, item_len)
            .index_select(
                0, torch.tensor(indices, dtype=torch.long, device=buffer.device)
            )
        )
        assert len(captured) == 1
        assert torch.equal(captured[0].view_as(expected), expected)

    staging = None
    check_staged_bytes(
        pool.local_kv_buffer[0], pool.local_kv_buffer[0][0].nbytes * pool.page_size
    )
    check_staged_bytes(
        pool.local_index_k_with_scale_buffer[0],
        pool.local_index_k_with_scale_buffer[0][0].nbytes,
    )

    for layer_id in range(2):
        main_buffer = pool.get_key_buffer(layer_id)
        index_buffer = pool.get_paged_index_k_with_scale_buffer(layer_id)
        for owner_rank in range(world_size):
            logical_page = torch.tensor(
                [owner_rank], dtype=torch.int64, device=f"cuda:{rank}"
            )
            logical_slot = logical_page * pool.page_size
            main_slot = pool.translate_main_slots(logical_slot)
            index_page = pool.prepare_paged_index_page_table(logical_page)
            assert torch.all(
                main_buffer[main_slot] == 100 * layer_id + owner_rank + 1
            ).item()
            assert torch.all(
                index_buffer[index_page] == 100 * layer_id + owner_rank + 11
            ).item()

    logical_slots = torch.arange(128, dtype=torch.int64, device=f"cuda:{rank}")
    index_k_bytes = (
        torch.arange(128 * 128, dtype=torch.int64, device=f"cuda:{rank}") % 120
    ).to(torch.uint8)
    index_k = index_k_bytes.view(128, 128).view(torch.float8_e4m3fn)
    index_scale = torch.arange(1, 129, dtype=torch.float32, device=f"cuda:{rank}")
    pool.set_index_k_scale_buffer(0, logical_slots, index_k, index_scale)
    pool.synchronize_shared_writes()

    actual_k, actual_scale = pool.get_index_k_scale_buffer(
        0,
        torch.tensor([128], dtype=torch.int64, device=f"cuda:{rank}"),
        torch.tensor([[0, 1]], dtype=torch.int32, device=f"cuda:{rank}"),
        128,
        128,
    )
    assert torch.equal(actual_k, index_k_bytes.view(128, 128))
    assert torch.equal(actual_scale, index_scale.view(torch.uint8).view(128, 4))

    logical_slots = torch.arange(64, 192, dtype=torch.int64, device=f"cuda:{rank}")
    key = (
        torch.arange(128 * 128, dtype=torch.float32, device=f"cuda:{rank}")
        .view(128, 128)
        .remainder(97)
        .sub_(48)
        .to(torch.bfloat16)
    )
    full_index = torch.zeros(
        pool._index_buffer_shape(4), dtype=torch.uint8, device=f"cuda:{rank}"
    )
    fused_store_index_k_cache(key, full_index, logical_slots, pool.page_size)

    local_slots, (local_key,) = pool.prepare_index_k_write(logical_slots, key)
    if local_slots.numel() > 0:
        fused_store_index_k_cache(
            local_key,
            pool.local_index_k_with_scale_buffer[1],
            local_slots,
            pool.page_size,
        )
    pool.synchronize_shared_writes()
    logical_pages = torch.tensor([[1, 2]], dtype=torch.int32, device=f"cuda:{rank}")
    seq_len = torch.tensor([128], dtype=torch.int64, device=f"cuda:{rank}")
    expected_k, expected_scale = index_buf_accessor.GetKAndS.execute(
        pool,
        full_index,
        page_indices=logical_pages,
        seq_len_tensor=seq_len,
        seq_len_sum=128,
        max_seq_len=128,
    )
    actual_k, actual_scale = pool.get_index_k_scale_buffer(
        1, seq_len, logical_pages, 128, 128
    )
    assert torch.equal(actual_k, expected_k)
    assert torch.equal(actual_scale, expected_scale)

    pool._clear_buffers()
    torch.distributed.barrier()
    _destroy_distributed()


class TestRankMajorSharedTensor(CustomTestCase):
    def test_peer_read_and_repeated_cleanup(self):
        if torch.cuda.device_count() < 2:
            self.skipTest("rank-major VMM test needs at least two GPUs")
        mp.spawn(_run_rank_major_vmm, args=(2, PORT), nprocs=2, join=True)

    def test_shared_pool_allocates_main_and_indexer_shards(self):
        if torch.cuda.device_count() < 2:
            self.skipTest("shared DSA pool test needs at least two GPUs")
        mp.spawn(_run_shared_pool, args=(2, POOL_PORT), nprocs=2, join=True)

    def test_slab_exposes_cross_rank_layer_views(self):
        if torch.cuda.device_count() < 2:
            self.skipTest("rank-major VMM slab test needs at least two GPUs")
        mp.spawn(_run_rank_major_slab, args=(2, SLAB_PORT), nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()
