"""CPU contracts for NVFP4 layer ownership, remote reads, and cache mutation."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.kernels.ops.attention.dsa import index_buf_accessor
from sglang.srt.layers.attention.dsa.nvfp4_k_cache import (
    dequantize_nvfp4_k_cache_paged_reference,
)
from sglang.srt.mem_cache.dsa_cache_layer_split import LayerSplitDSATokenToKVPool
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

PAGE_SIZE = 64
SIZE = PAGE_SIZE * 3
LAYER_NUM = 5  # Uneven ownership for both two and four CP ranks.
ROW_BYTES = 416
INDEX_ROW_BYTES = 132


def _make_pool(rank=None, world=1):
    kwargs = dict(
        size=SIZE,
        page_size=PAGE_SIZE,
        kv_lora_rank=512,
        dtype=torch.float4_e2m1fn_x2,
        qk_rope_head_dim=64,
        layer_num=LAYER_NUM,
        device="cpu",
        index_head_dim=128,
        enable_memory_saver=False,
        kv_cache_dim=ROW_BYTES,
    )
    if rank is not None:
        kwargs.update(layer_shard_rank=rank, layer_shard_size=world)
    cls = DSATokenToKVPool if rank is None else LayerSplitDSATokenToKVPool
    with (
        patch(
            "sglang.srt.mem_cache.memory_pool.maybe_init_custom_mem_pool",
            return_value=(False, None, None),
        ),
        # Recipe selection normally also requires a CUDA device. Exercise its
        # real byte layout on CPU with the codec's CPU reference implementation.
        patch(
            "sglang.srt.mem_cache.memory_pool.is_float4_e2m1fn_x2",
            side_effect=lambda dtype: dtype == torch.float4_e2m1fn_x2,
        ),
        patch.object(LayerSplitDSATokenToKVPool, "_init_layer_broadcast_comm"),
    ):
        pool = cls(**kwargs)
    for layer_id in range(LAYER_NUM):
        pool.set_mla_kv_global_scale(layer_id, 0.5 + layer_id * 0.375)
    return pool


def _make_pools(world):
    pools = [_make_pool(rank, world) for rank in range(world)]

    # Only replace transport. Allocation, ownership, codec, getters, prefetch,
    # cache mutation, and scale management all use the production methods.
    def broadcast(tensor, layer_id, **_kwargs):
        owner = pools[pools[0]._get_layer_owner_rank(layer_id)]
        if tensor.shape[-1] == ROW_BYTES:
            source = owner.kv_buffer[layer_id]
        else:
            source = owner.index_k_with_scale_buffer[layer_id]
        tensor.copy_(source)
        return tensor

    for pool in pools:
        pool._broadcast_tensor_from_owner = broadcast
    return pools


def _inputs(layer_id, count):
    generator = torch.Generator().manual_seed(100 + layer_id)
    return (
        torch.randn(count, 1, 512, generator=generator).to(torch.bfloat16),
        torch.randn(count, 1, 64, generator=generator).to(torch.bfloat16),
    )


@pytest.mark.parametrize("world", [2, 4])
def test_chunked_writes_and_prefetched_rows_match_replicated_nvfp4(world):
    reference = _make_pool()
    pools = _make_pools(world)
    # Cross page boundaries and keep physical token order nontrivial.
    loc = torch.arange(1, 132).roll(37)
    with (
        get_parallel().override(attn_dcp_size=1),
        patch(
            "sglang.srt.mem_cache.memory_pool.dequantize_nvfp4_k_cache_paged",
            side_effect=dequantize_nvfp4_k_cache_paged_reference,
        ),
    ):
        for layer_id in range(LAYER_NUM):
            layer = SimpleNamespace(layer_id=layer_id)
            nope, rope = _inputs(layer_id, loc.numel())
            for begin, end in [(0, 65), (65, loc.numel())]:
                # Prefetch before a write also tests updating an existing remote
                # copy instead of reading stale prefix bytes on the next chunk.
                for pool in pools:
                    pool.prefetch_kv_buffer(layer_id)
                for pool in [reference, *pools]:
                    pool.set_mla_kv_buffer(
                        layer, loc[begin:end], nope[begin:end], rope[begin:end]
                    )
                for pool in pools:
                    raw = pool.get_raw_mla_kv_buffer(layer_id)
                    assert raw["kv"].dtype == torch.uint8
                    assert raw["kv"].shape == (SIZE + PAGE_SIZE, 1, ROW_BYTES)
                    assert torch.equal(raw["kv"], reference.get_key_buffer(layer_id))
                    assert (
                        pool.get_value_buffer(layer_id).data_ptr()
                        == raw["kv"].data_ptr()
                    )
                    actual = pool.get_mla_kv_buffer(layer, loc[:end])
                    expected = reference.get_mla_kv_buffer(layer, loc[:end])
                    for actual_part, expected_part in zip(actual, expected):
                        torch.testing.assert_close(
                            actual_part, expected_part, rtol=0, atol=0
                        )


@pytest.mark.parametrize("sharded", [False, True])
def test_global_scale_values_and_addresses_survive_buffer_rebuild(sharded):
    pool = _make_pool(0, 2) if sharded else _make_pool()
    scales = pool.mla_kv_global_scale.clone()
    address = pool.mla_kv_global_scale.data_ptr()
    pool._clear_buffers()
    with patch.object(LayerSplitDSATokenToKVPool, "_init_layer_broadcast_comm"):
        pool._create_buffers()
    pool.index_key_cache = pool._create_index_key_cache()
    assert pool.mla_kv_global_scale.data_ptr() == address
    assert torch.equal(pool.mla_kv_global_scale, scales)


@pytest.mark.parametrize("world", [2, 4])
def test_pool_bytes_and_pd_descriptors_include_only_owned_layers(world):
    reference = _make_pool()
    for pool in _make_pools(world):
        owned = [i for i in range(LAYER_NUM) if pool._is_layer_owned(i)]
        ptrs, lengths, items = pool.get_contiguous_buf_infos()
        assert ptrs == [pool.kv_buffer[i].data_ptr() for i in owned]
        assert lengths == [(SIZE + PAGE_SIZE) * ROW_BYTES] * len(owned)
        assert items == [PAGE_SIZE * ROW_BYTES] * len(owned)
        idx_ptrs, idx_lengths, idx_items = pool.get_state_buf_infos()
        assert idx_ptrs == [pool.index_k_with_scale_buffer[i].data_ptr() for i in owned]
        assert idx_lengths == [pool.index_k_with_scale_buffer[i].nbytes for i in owned]
        assert idx_items == [PAGE_SIZE * INDEX_ROW_BYTES] * len(owned)
        allocations = [
            *pool.kv_buffer,
            *pool.index_k_with_scale_buffer,
            pool.remote_kv_buffer,
            pool.index_key_cache.remote_buffer,
            pool.mla_kv_global_scale,
        ]
        assert pool.get_kv_size_bytes() == sum(tensor.nbytes for tensor in allocations)
        assert pool.get_kv_size_bytes() < reference.get_kv_size_bytes()


@pytest.mark.parametrize("world", [2, 4])
def test_move_and_restore_refresh_remote_kv_and_page_packed_indexer(world):
    pools = _make_pools(world)
    all_indices = torch.arange(PAGE_SIZE * 2)
    with get_parallel().override(attn_dcp_size=1):
        for layer_id in range(LAYER_NUM):
            layer = SimpleNamespace(layer_id=layer_id)
            nope, rope = _inputs(layer_id, all_indices.numel())
            for pool in pools:
                pool.set_mla_kv_buffer(layer, all_indices, nope, rope)
                if pool._is_layer_owned(layer_id):
                    buffer = pool.index_k_with_scale_buffer[layer_id]
                    for page in range(buffer.shape[0]):
                        buffer[page, : PAGE_SIZE * 128] = (
                            torch.arange(PAGE_SIZE * 128) + page * 17 + layer_id
                        ).to(torch.uint8)
                        buffer[page, PAGE_SIZE * 128 :] = (
                            torch.arange(PAGE_SIZE, dtype=torch.float32)
                            + 0.25 * layer_id
                            + page * PAGE_SIZE
                            + 1
                        ).view(torch.uint8)

    def read_index(pool):
        buffer = pool.get_broadcastable_index_k_with_scale_buffer(layer_id)
        pages = torch.arange(2)
        return (
            index_buf_accessor.GetK.slow(pool, buffer, 128, pages),
            index_buf_accessor.GetS.slow(pool, buffer, 128, pages),
        )

    # These are token locations larger than the number of indexer pages.
    # The third pair also checks overlapping source and target sets.
    source, target = torch.tensor([65, 70, 1]), torch.tensor([1, 2, 65])
    layer_id = LAYER_NUM - 1
    expected_kv = pools[0].get_key_buffer(layer_id)[:128].clone()
    expected_kv[target] = expected_kv[source]
    expected_keys, expected_scales = read_index(pools[0])
    expected_keys[target] = expected_keys[source]
    expected_scales[target] = expected_scales[source]
    for pool in pools:
        pool.prefetch_kv_buffer(layer_id)
        pool.get_broadcastable_index_k_with_scale_buffer(layer_id)
        pool.move_kv_cache(target, source)
        assert pool.remote_kv_layer_id is None
        assert pool.index_key_cache.remote_layer_id is None
    for pool in pools:
        assert torch.equal(pool.get_key_buffer(layer_id)[:128], expected_kv)
        keys, scales = read_index(pool)
        assert torch.equal(keys, expected_keys)
        assert torch.equal(scales, expected_scales)

    with (
        patch("torch.cuda.synchronize"),
        patch("sglang.srt.utils.current_platform.synchronize"),
    ):
        saved = [pool.get_cpu_copy(all_indices) for pool in pools]
        # CPU offload must own its copy just as a GPU-to-CPU transfer does.
        for snapshot in saved:
            for field in ("kv", "index_k"):
                snapshot[field] = [
                    [chunk.clone() for chunk in layer] for layer in snapshot[field]
                ]
        for pool in pools:
            for tensor in [*pool.kv_buffer, *pool.index_k_with_scale_buffer]:
                tensor.zero_()
        for pool, snapshot in zip(pools, saved):
            pool.load_cpu_copy(snapshot, all_indices)
            assert pool.remote_kv_layer_id is None
            assert pool.index_key_cache.remote_layer_id is None
    for pool in pools:
        assert torch.equal(pool.get_key_buffer(layer_id)[:128], expected_kv)
        keys, scales = read_index(pool)
        assert torch.equal(keys, expected_keys)
        assert torch.equal(scales, expected_scales)
