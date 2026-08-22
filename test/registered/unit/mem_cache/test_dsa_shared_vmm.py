import os
import time
import unittest

import torch
import torch.multiprocessing as mp

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-c", runner_config="4-gpu-b200")

POOL_PORT = 29722
MAIN_ONLY_POOL_PORT = 29723
STATUS_POOL_PORT = 29724


def _destroy_distributed() -> None:
    from sglang.srt.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
    )

    destroy_model_parallel()
    destroy_distributed_environment()


def _run_shared_pool(rank: int, world_size: int, port: int):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    torch.cuda.set_device(rank)

    from sglang.kernels.ops.attention.dsa import index_buf_accessor
    from sglang.kernels.ops.attention.fused_store_index_cache import (
        fused_store_index_k_cache,
    )
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
        kv_cache_dim=656,
        shared_rank=rank,
        shared_size=world_size,
        indexer_cache_layer_ids=(0, 1),
    )
    assert len(pool.kv_buffer) == 2
    assert len(pool.index_key_cache.buffer) == 2
    assert pool.main_family.slab.rank_local_views == []
    assert len(pool.index_key_cache.family.slab.rank_local_views) == 2
    assert pool.main_layout.cp_size == world_size
    assert pool.index_layout.cp_size == world_size
    assert pool.kv_buffer[0].device.index == rank
    assert pool.index_key_cache.buffer[0].device.index == rank

    for layer_id in range(2):
        pool.local_kv_buffer[layer_id].fill_(100 * layer_id + rank + 1)
        pool.index_key_cache.local_buffer[layer_id].fill_(100 * layer_id + rank + 11)
    torch.cuda.synchronize()
    torch.distributed.barrier(group=pool.shared_cp_group.cpu_group)

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
    full_index = torch.zeros_like(pool.index_key_cache.buffer[1])
    fused_store_index_k_cache(key, full_index, logical_slots, pool.page_size)

    for buffer, owner_rank, owner_size in pool.get_index_k_write_targets(1):
        fused_store_index_k_cache(
            key,
            buffer,
            logical_slots,
            pool.page_size,
            owner_rank=owner_rank,
            owner_size=owner_size,
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

    prepared_pages = pool.prepare_paged_index_page_table(logical_pages)
    pool_cache, pool_page_table = pool.materialize_index_pages(
        1,
        prepared_pages,
        logical_pages,
        seq_len,
    )
    torch.cuda.synchronize()
    assert torch.equal(pool_page_table, logical_pages)
    assert torch.equal(
        pool_cache[logical_pages.view(-1).long()],
        full_index[logical_pages.view(-1).long()],
    )

    # Exercise a cross-owner swap. Delaying rank 1 makes the old one-phase
    # implementation deterministically overwrite rank 1's source before it is
    # read; a correct two-phase move snapshots every source before any write.
    swap_targets = torch.tensor([1, 65], dtype=torch.int64, device=f"cuda:{rank}")
    swap_sources = swap_targets.flip(0)
    translated_swap = pool.translate_main_slots(swap_targets)
    main_before = [buf[translated_swap].clone() for buf in pool.kv_buffer]
    translated_index_swap = pool.translate_index_slots(swap_targets)
    from sglang.srt.mem_cache.dsa_cache_shared import gather_shared_index_rows

    index_before = [
        gather_shared_index_rows(pool, buf, translated_index_swap)
        for buf in pool.index_key_cache.buffer
    ]
    torch.distributed.barrier(group=pool.shared_cp_group.cpu_group)
    if rank == 1:
        time.sleep(0.5)
    pool.move_kv_cache(swap_targets, swap_sources)

    for layer_id, expected_rows in enumerate(main_before):
        actual_rows = pool.kv_buffer[layer_id][translated_swap]
        assert torch.equal(actual_rows, expected_rows.flip(0))
    for layer_id, (expected_k, expected_scale) in enumerate(index_before):
        actual_k, actual_scale = gather_shared_index_rows(
            pool,
            pool.index_key_cache.buffer[layer_id],
            translated_index_swap,
        )
        assert torch.equal(actual_k, expected_k.flip(0))
        assert torch.equal(actual_scale, expected_scale.flip(0))

    offload_slots = swap_targets
    expected_main = [
        buf[pool.translate_main_slots(offload_slots)].clone() for buf in pool.kv_buffer
    ]
    expected_index = [
        gather_shared_index_rows(pool, buf, pool.translate_index_slots(offload_slots))
        for buf in pool.index_key_cache.buffer
    ]
    cpu_copy = pool.get_cpu_copy(offload_slots)
    owned = pool.main_layout.owned_slot_mask(offload_slots, owner_rank=pool.shared_rank)
    local_rows = pool.main_layout.translate_local_slots(offload_slots[owned])
    for local_buffer in pool.local_kv_buffer:
        local_buffer[local_rows].zero_()
    pool.set_index_k_scale_buffer(
        0,
        offload_slots,
        torch.zeros((2, 128), dtype=torch.float8_e4m3fn, device=f"cuda:{rank}"),
        torch.zeros((2,), dtype=torch.float32, device=f"cuda:{rank}"),
    )
    pool.set_index_k_scale_buffer(
        1,
        offload_slots,
        torch.zeros((2, 128), dtype=torch.float8_e4m3fn, device=f"cuda:{rank}"),
        torch.zeros((2,), dtype=torch.float32, device=f"cuda:{rank}"),
    )
    pool.synchronize_shared_writes()
    pool.load_cpu_copy(cpu_copy, offload_slots)

    for layer_id, expected_rows in enumerate(expected_main):
        actual_rows = pool.kv_buffer[layer_id][pool.translate_main_slots(offload_slots)]
        assert torch.equal(actual_rows, expected_rows)
    for layer_id, (expected_k, expected_scale) in enumerate(expected_index):
        actual_k, actual_scale = gather_shared_index_rows(
            pool,
            pool.index_key_cache.buffer[layer_id],
            pool.translate_index_slots(offload_slots),
        )
        assert torch.equal(actual_k, expected_k)
        assert torch.equal(actual_scale, expected_scale)

    pool._clear_buffers()
    torch.distributed.barrier()
    _destroy_distributed()


def _run_shared_status(rank: int, world_size: int, port: int):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    torch.cuda.set_device(rank)

    from sglang.srt.distributed.parallel_state import (
        get_attn_cp_group,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from sglang.srt.mem_cache.shared_kv.synchronization import SharedWritePublisher

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
    publisher = SharedWritePublisher(get_attn_cp_group())

    # Rank 1 injects the local L2-copy failure. Every rank must observe the
    # aggregate false result, and a later successful round must recover.
    assert not publisher.publish_status(rank != 1)
    assert publisher.publish_status(True)

    publisher.close()
    torch.distributed.barrier()
    _destroy_distributed()


def _run_main_only_shared_pool(rank: int, world_size: int, port: int):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    torch.cuda.set_device(rank)

    from sglang.kernels.ops.attention.fused_store_index_cache import (
        fused_store_index_k_cache,
    )
    from sglang.srt.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from sglang.srt.mem_cache.dsa_cache_shared import (
        ReplicatedIndexKeyCache,
        SharedDSATokenToKVPool,
        gather_shared_index_rows,
    )

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
        kv_cache_dim=656,
        shared_rank=rank,
        shared_size=world_size,
        share_indexer=False,
        indexer_cache_layer_ids=(),
    )
    assert isinstance(pool.index_key_cache, ReplicatedIndexKeyCache)
    assert pool.index_layout is None
    assert pool.prepare_paged_index_page_table(
        torch.tensor([[0, 1]], dtype=torch.int32, device=f"cuda:{rank}")
    ).tolist() == [[0, 1]]
    assert pool.shared_cache_access.uses_shared_indexer is False

    slots = torch.arange(128, dtype=torch.int64, device=f"cuda:{rank}")
    key = (
        torch.arange(128 * 128, dtype=torch.float32, device=f"cuda:{rank}")
        .view(128, 128)
        .remainder(97)
        .sub_(48)
        .to(torch.bfloat16)
    )
    for layer_id in range(2):
        for buffer, owner_rank, owner_size in pool.get_index_k_write_targets(layer_id):
            assert (owner_rank, owner_size) == (0, 1)
            fused_store_index_k_cache(
                key,
                buffer,
                slots,
                pool.page_size,
                owner_rank=owner_rank,
                owner_size=owner_size,
            )

    swap_targets = torch.tensor([1, 65], dtype=torch.int64, device=f"cuda:{rank}")
    swap_sources = swap_targets.flip(0)
    before = [
        gather_shared_index_rows(pool, buffer, swap_targets)
        for buffer in pool.index_key_cache.buffer
    ]
    pool.move_kv_cache(swap_targets, swap_sources)
    for layer_id, (expected_k, expected_scale) in enumerate(before):
        actual_k, actual_scale = gather_shared_index_rows(
            pool, pool.index_key_cache.buffer[layer_id], swap_targets
        )
        assert torch.equal(actual_k, expected_k.flip(0))
        assert torch.equal(actual_scale, expected_scale.flip(0))

    pool._clear_buffers()
    torch.distributed.barrier()
    _destroy_distributed()


class TestSharedDSAPool(CustomTestCase):
    def test_shared_status_propagates_an_injected_rank_failure(self):
        if torch.cuda.device_count() < 2:
            self.skipTest("shared DSA status test needs at least two GPUs")
        mp.spawn(_run_shared_status, args=(2, STATUS_POOL_PORT), nprocs=2, join=True)

    def test_shared_pool_shards_main_and_indexer(self):
        if torch.cuda.device_count() < 2:
            self.skipTest("shared DSA pool test needs at least two GPUs")
        mp.spawn(_run_shared_pool, args=(2, POOL_PORT), nprocs=2, join=True)

    def test_shared_pool_can_keep_replicated_indexer(self):
        if torch.cuda.device_count() < 2:
            self.skipTest("shared DSA pool test needs at least two GPUs")
        mp.spawn(
            _run_main_only_shared_pool,
            args=(2, MAIN_ONLY_POOL_PORT),
            nprocs=2,
            join=True,
        )


if __name__ == "__main__":
    unittest.main()
