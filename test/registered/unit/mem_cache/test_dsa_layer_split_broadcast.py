"""Multi-GPU integration test for LayerSplitDSATokenToKVPool owner-broadcast.

Spawns ``world`` processes forming a single attention-CP group, builds a tiny
``LayerSplitDSATokenToKVPool`` on each rank, writes a rank-distinct value into
every owned layer, then verifies that reading ANY layer (owned or not) returns
the *owning* rank's bytes -- i.e. the owner-broadcast in
``_get_broadcastable_kv_buffer`` / ``prefetch_kv_buffer`` surfaces correct
contents. Also exercises the DSA indexer broadcast and the async prefetch path.

Registered as a base-c 4-gpu-b200 unit test; uses up to 4 GPUs and skips when
fewer than 2 are visible. Run directly on 2+ GPUs:
    CUDA_VISIBLE_DEVICES=0,1 python -m pytest \
        test/registered/unit/mem_cache/test_dsa_layer_split_broadcast.py
"""

import os
import unittest
from types import SimpleNamespace

import torch
import torch.multiprocessing as mp

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase, publish_build_topology

register_cuda_ci(est_time=180, stage="base-c", runner_config="4-gpu-b200")

LAYER_NUM = 4
PAGE_SIZE = 64
KV_LORA_RANK = 512
QK_ROPE = 64
INDEX_HEAD_DIM = 128
SIZE = PAGE_SIZE * 3  # a few pages
PORT = 29711


def _run(rank: int, world: int, port: int, nvfp4: bool = False, fp8: bool = False):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    os.environ.setdefault("no_proxy", "127.0.0.1,localhost")
    torch.cuda.set_device(rank)

    from sglang.srt.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from sglang.srt.runtime_context import get_parallel

    init_distributed_environment(
        world_size=world,
        rank=rank,
        local_rank=rank,
        distributed_init_method=f"tcp://127.0.0.1:{port}",
        backend="nccl",
    )
    publish_build_topology(tp_size=world, attn_cp_size=world, world_rank=rank)
    initialize_model_parallel()

    if nvfp4:
        _run_nvfp4(rank, world)
        return
    if fp8:
        _run_fp8(rank, world)
        return

    from sglang.srt.mem_cache.dsa_cache_layer_split import (
        LayerSplitDSATokenToKVPool,
    )

    # This worker builds the groups but never publishes, so the scope declares
    # the size it runs at.
    with get_parallel().override(attn_cp_size=world):
        cp_rank = get_parallel().attn_cp_rank
        cp_size = get_parallel().attn_cp_size
        assert cp_size == world

        pool = LayerSplitDSATokenToKVPool(
            SIZE,
            page_size=PAGE_SIZE,
            kv_lora_rank=KV_LORA_RANK,
            dtype=torch.bfloat16,
            qk_rope_head_dim=QK_ROPE,
            layer_num=LAYER_NUM,
            device=f"cuda:{rank}",
            index_head_dim=INDEX_HEAD_DIM,
            enable_memory_saver=False,
            kv_cache_dim=KV_LORA_RANK + QK_ROPE,
            layer_shard_rank=cp_rank,
            layer_shard_size=cp_size,
        )

        # Owner writes a layer-distinct constant into each owned kv_buffer layer.
        for layer_id in range(LAYER_NUM):
            if pool._is_layer_owned(layer_id):
                pool.kv_buffer[layer_id].fill_(float(layer_id + 1))

        torch.cuda.synchronize()
        torch.distributed.barrier()

        # Every rank reads every layer; broadcast must surface the owner's value.
        ok = True
        for layer_id in range(LAYER_NUM):
            buf = pool._get_broadcastable_kv_buffer(layer_id)
            expected = float(layer_id + 1)
            got = buf.float().mean().item()
            if abs(got - expected) > 1e-3:
                print(f"[rank {rank}] layer {layer_id}: expected {expected}, got {got}")
                ok = False
        assert ok, f"rank {rank} read stale/incorrect broadcast contents"

        # Indexer buffer owner-broadcast: owner writes a layer-distinct value, then
        # every rank must read it back for every layer.
        for layer_id in range(LAYER_NUM):
            store_buf = pool.get_index_k_with_scale_buffer(layer_id)
            assert (
                store_buf.data_ptr()
                == pool.index_k_with_scale_buffer[layer_id].data_ptr()
            )
            if pool._is_layer_owned(layer_id):
                store_buf.fill_(layer_id + 10)
        torch.cuda.synchronize()
        torch.distributed.barrier()
        for layer_id in range(LAYER_NUM):
            # invalidate any cached remote copy so the read forces a fresh broadcast
            pool.invalidate_index_buffer_for_layer(layer_id)
            buf = pool._get_broadcastable_index_buffer(layer_id)
            expected = layer_id + 10
            got = buf.float().mean().item()
            if abs(got - expected) > 1e-3:
                print(
                    f"[rank {rank}] index layer {layer_id}: exp {expected}, got {got}"
                )
                ok = False
        assert ok, f"rank {rank} read stale/incorrect index broadcast contents"

        # Async prefetch path: prefetch layer, then read must return owner value.
        for layer_id in range(LAYER_NUM):
            pool.remote_kv_layer_id = None  # force a fresh broadcast
            pool.prefetch_kv_buffer(layer_id)
            buf = pool._get_broadcastable_kv_buffer(layer_id)
            got = buf.float().mean().item()
            if abs(got - float(layer_id + 1)) > 1e-3:
                print(f"[rank {rank}] prefetch layer {layer_id}: got {got}")
                ok = False
        assert ok, f"rank {rank} prefetch path returned incorrect contents"

        print(f"[rank {rank}] OK: all {LAYER_NUM} layers read correct owner contents")
        torch.distributed.barrier()


@torch.inference_mode()
def _run_fp8(rank: int, world: int):
    from sglang.srt.mem_cache.dsa_cache_layer_split import LayerSplitDSATokenToKVPool
    from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool

    device = f"cuda:{rank}"
    kwargs = dict(
        size=SIZE,
        page_size=PAGE_SIZE,
        kv_lora_rank=KV_LORA_RANK,
        dtype=torch.float8_e4m3fn,
        qk_rope_head_dim=QK_ROPE,
        layer_num=5,
        device=device,
        index_head_dim=INDEX_HEAD_DIM,
        enable_memory_saver=False,
        kv_cache_dim=656,
    )
    reference = DSATokenToKVPool(**kwargs)
    pool = LayerSplitDSATokenToKVPool(
        **kwargs, layer_shard_rank=rank, layer_shard_size=world
    )
    assert pool.dsa_kv_cache_store_fp8 and not pool.dsa_kv_cache_store_nvfp4
    assert not hasattr(pool, "mla_kv_global_scale")
    loc = torch.arange(1, 132, device=device)
    for layer_id in range(5):
        generator = torch.Generator(device=device).manual_seed(500 + layer_id)
        nope = torch.randn(
            (loc.numel(), 1, KV_LORA_RANK),
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )
        rope = torch.randn(
            (loc.numel(), 1, QK_ROPE),
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )
        layer = SimpleNamespace(layer_id=layer_id)
        for part in (loc[:67], loc[67:]):
            offsets = part - 1
            pool.prefetch_kv_buffer(layer_id)
            reference.set_mla_kv_buffer(layer, part, nope[offsets], rope[offsets])
            pool.set_mla_kv_buffer(layer, part, nope[offsets], rope[offsets])
            keys = pool.get_key_buffer(layer_id)
            values = pool.get_value_buffer(layer_id)
            assert keys.dtype == values.dtype == torch.float8_e4m3fn
            assert keys.shape[-1] == 656
            assert keys.data_ptr() == values.data_ptr()
            assert torch.equal(
                keys.view(torch.uint8),
                reference.get_key_buffer(layer_id).view(torch.uint8),
            )
    assert pool.get_kv_size_bytes() < reference.get_kv_size_bytes()
    torch.distributed.barrier()
    print(f"[rank {rank}] FP8 CP={world}: packed rows, prefetch writes and reads OK")


@torch.inference_mode()
def _run_nvfp4(rank: int, world: int):
    from sgl_kernel import flashmla_ops  # noqa: F401

    from sglang.kernels.ops.attention.dsa import index_buf_accessor
    from sglang.srt.mem_cache.dsa_cache_layer_split import LayerSplitDSATokenToKVPool
    from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool

    assert hasattr(torch.ops.sgl_kernel, "sparse_decode_fwd_nvfp4")
    device = f"cuda:{rank}"
    kwargs = dict(
        size=SIZE,
        page_size=PAGE_SIZE,
        kv_lora_rank=KV_LORA_RANK,
        dtype=torch.float4_e2m1fn_x2,
        qk_rope_head_dim=QK_ROPE,
        layer_num=5,
        device=device,
        index_head_dim=INDEX_HEAD_DIM,
        enable_memory_saver=False,
        kv_cache_dim=416,
    )
    reference = DSATokenToKVPool(**kwargs)
    pool = LayerSplitDSATokenToKVPool(
        **kwargs, layer_shard_rank=rank, layer_shard_size=world
    )
    loc = torch.arange(1, 132, device=device).roll(37)
    for layer_id in range(5):
        layer = SimpleNamespace(layer_id=layer_id)
        for current in (reference, pool):
            current.set_mla_kv_global_scale(layer_id, 0.5 + layer_id * 0.375)
        generator = torch.Generator(device=device).manual_seed(100 + layer_id)
        nope = torch.randn(131, 1, 512, generator=generator, device=device).bfloat16()
        rope = torch.randn(131, 1, 64, generator=generator, device=device).bfloat16()
        index_keys = (
            (torch.arange(131 * 128, device=device).view(131, 128) + layer_id) % 13 - 6
        ).to(torch.float8_e4m3fn)
        index_scales = (
            torch.arange(131, device=device, dtype=torch.float32).view(-1, 1) / 128
            + 0.5
            + layer_id / 16
        )
        for begin, end in [(0, 65), (65, 131)]:
            pool.prefetch_kv_buffer(layer_id)
            for current in (reference, pool):
                current.set_mla_kv_buffer(
                    layer, loc[begin:end], nope[begin:end], rope[begin:end]
                )
                current.index_key_cache.store_quantized(
                    layer_id,
                    loc[begin:end],
                    index_keys[begin:end],
                    index_scales[begin:end],
                )
            raw = pool.get_key_buffer(layer_id)
            assert raw.dtype == torch.uint8 and raw.shape[-1] == 416
            assert torch.equal(raw, reference.get_key_buffer(layer_id))
            assert pool.get_value_buffer(layer_id).data_ptr() == raw.data_ptr()
            assert torch.equal(
                pool.index_key_cache.get_buffer(layer_id),
                reference.index_key_cache.get_buffer(layer_id),
            )
            for actual, expected in zip(
                pool.get_mla_kv_buffer(layer, loc[:end]),
                reference.get_mla_kv_buffer(layer, loc[:end]),
            ):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

            # Exercise the native H64 consumer with both the ordinary query and
            # the six-query EAGLE-flat shape. Use the existing native tolerance.
            for query_count in (1, 6):
                q = torch.randn(
                    1,
                    query_count,
                    64,
                    576,
                    generator=generator,
                    device=device,
                    dtype=torch.bfloat16,
                )
                indices = torch.full(
                    (1, query_count, 2048), -1, dtype=torch.int32, device=device
                )
                indices[..., :end] = loc[:end].to(torch.int32)

                def decode(current, layer_id, q, indices):
                    return torch.ops.sgl_kernel.sparse_decode_fwd_nvfp4.default(
                        q,
                        current.get_key_buffer(layer_id).view(-1, 64, 1, 416),
                        current.get_mla_kv_global_scale(layer_id),
                        indices,
                        None,
                        None,
                        None,
                        None,
                        512,
                        576**-0.5,
                    )[0]

                torch.testing.assert_close(
                    decode(pool, layer_id, q, indices),
                    decode(reference, layer_id, q, indices),
                    atol=8e-4,
                    rtol=2.01 / 128,
                )

    # Check real paged indexer K/scale bytes, including overlapping cross-page
    # moves, invalidation of an in-flight prefetch, and CPU offload/restore.
    layer_id = 4
    all_indices = torch.arange(128, device=device)
    source = torch.tensor([65, 70, 1], device=device)
    target = torch.tensor([1, 2, 65], device=device)
    pages = torch.arange(2, device=device)
    expected_kv = reference.get_key_buffer(layer_id)[:128].clone()
    expected_kv[target] = expected_kv[source]
    reference_index = reference.index_key_cache.get_buffer(layer_id)
    expected_keys = index_buf_accessor.GetK.slow(reference, reference_index, 128, pages)
    expected_scales = index_buf_accessor.GetS.slow(
        reference, reference_index, 128, pages
    )
    expected_keys[target] = expected_keys[source]
    expected_scales[target] = expected_scales[source]
    pool.prefetch_kv_buffer(0)
    pool.move_kv_cache(target, source)
    assert pool.remote_kv_layer_id is None
    assert pool.index_key_cache.remote_layer_id is None

    def check_moved_rows():
        assert torch.equal(pool.get_key_buffer(layer_id)[:128], expected_kv)
        buffer = pool.index_key_cache.get_buffer(layer_id)
        assert torch.equal(
            index_buf_accessor.GetK.slow(pool, buffer, 128, pages), expected_keys
        )
        assert torch.equal(
            index_buf_accessor.GetS.slow(pool, buffer, 128, pages), expected_scales
        )

    check_moved_rows()
    saved = pool.get_cpu_copy(all_indices)
    for tensor in [*pool.kv_buffer, *pool.index_k_with_scale_buffer]:
        tensor.zero_()
    pool.prefetch_kv_buffer(0)
    pool.load_cpu_copy(saved, all_indices)
    assert pool.remote_kv_layer_id is None
    assert pool.index_key_cache.remote_layer_id is None
    check_moved_rows()

    expected_bytes = sum(
        t.nbytes
        for t in [
            *pool.kv_buffer,
            *pool.index_k_with_scale_buffer,
            pool.remote_kv_buffer,
            pool.index_key_cache.remote_buffer,
            pool.mla_kv_global_scale,
        ]
    )
    assert pool.get_kv_size_bytes() == expected_bytes
    assert expected_bytes < reference.get_kv_size_bytes()
    torch.distributed.barrier()
    print(
        f"[rank {rank}] NVFP4 CP={world}: bytes, prefix reads, H64/EAGLE decode, "
        "indexer moves, restore OK"
    )


class TestLayerSplitDSABroadcast(CustomTestCase):
    def test_owner_broadcast(self):
        world = min(4, torch.cuda.device_count())
        if world < 2:
            self.skipTest("LayerSplitDSATokenToKVPool broadcast test needs >= 2 GPUs")
        mp.spawn(_run, args=(world, PORT), nprocs=world, join=True)

    def test_fp8_owner_broadcast_and_prefetched_writes(self):
        if torch.cuda.device_count() < 2:
            self.skipTest("FP8 layer-split broadcast test needs >= 2 GPUs")
        for world in (2, 4):
            if torch.cuda.device_count() >= world:
                with self.subTest(world=world):
                    mp.spawn(
                        _run,
                        args=(world, PORT + 10 + world, False, True),
                        nprocs=world,
                        join=True,
                    )

    def test_nvfp4_owner_broadcast_and_native_decode(self):
        if torch.cuda.device_count() < 2:
            self.skipTest("NVFP4 layer-split broadcast test needs >= 2 GPUs")
        if torch.cuda.get_device_capability()[0] != 10:
            self.skipTest("NVFP4 DSA requires SM100")
        for world in (2, 4):
            if torch.cuda.device_count() >= world:
                with self.subTest(world=world):
                    mp.spawn(
                        _run, args=(world, PORT + world, True), nprocs=world, join=True
                    )


if __name__ == "__main__":
    unittest.main()
