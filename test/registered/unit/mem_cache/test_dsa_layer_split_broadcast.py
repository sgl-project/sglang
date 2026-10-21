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

register_cuda_ci(est_time=300, stage="base-c", runner_config="4-gpu-b200")

LAYER_NUM = 4
PAGE_SIZE = 64
KV_LORA_RANK = 512
QK_ROPE = 64
INDEX_HEAD_DIM = 128
SIZE = PAGE_SIZE * 3  # a few pages
PORT = 29711


def _run(
    rank: int,
    world: int,
    port: int,
    nvfp4: bool = False,
    fp8: bool = False,
    owner_graph: bool = False,
):
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

    if owner_graph:
        _run_nvfp4_owner_graph(rank, world)
        return
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
    kwargs = {
        "size": SIZE,
        "page_size": PAGE_SIZE,
        "kv_lora_rank": KV_LORA_RANK,
        "dtype": torch.float8_e4m3fn,
        "qk_rope_head_dim": QK_ROPE,
        "layer_num": 5,
        "device": device,
        "index_head_dim": INDEX_HEAD_DIM,
        "enable_memory_saver": False,
        "kv_cache_dim": 656,
    }
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
    from sgl_kernel import flashmla_nvfp4_ops  # noqa: F401
    from sglang.srt.mem_cache.dsa_cache_layer_split import LayerSplitDSATokenToKVPool
    from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool

    assert hasattr(torch.ops.sgl_kernel, "sparse_decode_fwd_nvfp4")
    device = f"cuda:{rank}"
    kwargs = {
        "size": SIZE,
        "page_size": PAGE_SIZE,
        "kv_lora_rank": KV_LORA_RANK,
        "dtype": torch.float4_e2m1fn_x2,
        "qk_rope_head_dim": QK_ROPE,
        "layer_num": 5,
        "device": device,
        "index_head_dim": INDEX_HEAD_DIM,
        "enable_memory_saver": False,
        "kv_cache_dim": 416,
    }
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

    def read_index(buffer):
        # Independent byte reference for the page-packed key/scale layout.
        selected = buffer[pages]
        key_bytes = PAGE_SIZE * INDEX_HEAD_DIM
        return (
            selected[:, :key_bytes].reshape(-1, INDEX_HEAD_DIM),
            selected[:, key_bytes:].reshape(-1, 4),
        )

    expected_keys, expected_scales = read_index(reference_index)
    expected_keys[target] = expected_keys[source]
    expected_scales[target] = expected_scales[source]
    pool.prefetch_kv_buffer(0)
    pool.move_kv_cache(target, source)
    assert pool.remote_kv_layer_id is None
    assert pool.index_key_cache.remote_layer_id is None

    def check_moved_rows():
        assert torch.equal(pool.get_key_buffer(layer_id)[:128], expected_kv)
        buffer = pool.index_key_cache.get_buffer(layer_id)
        keys, scales = read_index(buffer)
        assert torch.equal(keys, expected_keys)
        assert torch.equal(scales, expected_scales)

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


@torch.inference_mode()
def _run_nvfp4_owner_graph_case(rank: int, world: int, query_count: int, mode):
    """An eager prefix changes scratch state between graph replays.

    The next decode must use the new owner-local KV and index bytes. This
    protects the non-PD owner route from stale scratch tags, capture-time
    tensor values, and a broadcast whose root differs from layer ownership.
    """
    from sgl_kernel import flashmla_nvfp4_ops, flashmla_ops  # noqa: F401
    from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend
    from sglang.srt.mem_cache.dsa_cache_layer_split import LayerSplitDSATokenToKVPool
    from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    device = f"cuda:{rank}"
    kwargs = {
        "size": SIZE,
        "page_size": PAGE_SIZE,
        "kv_lora_rank": KV_LORA_RANK,
        "dtype": torch.float4_e2m1fn_x2,
        "qk_rope_head_dim": QK_ROPE,
        "layer_num": 5,
        "device": device,
        "index_head_dim": INDEX_HEAD_DIM,
        "enable_memory_saver": False,
        "kv_cache_dim": 416,
    }
    reference = DSATokenToKVPool(**kwargs)
    pool = LayerSplitDSATokenToKVPool(
        **kwargs,
        layer_shard_rank=rank,
        layer_shard_size=world,
        unified_decode=True,
    )
    assert pool.use_owner_compute(mode)
    assert not pool.use_owner_compute(ForwardMode.EXTEND)
    generator = torch.Generator(device=device).manual_seed(9100 + query_count)
    layers = [
        SimpleNamespace(
            layer_id=layer_id, head_dim=576, v_head_dim=512, tp_q_head_num=64
        )
        for layer_id in range(5)
    ]
    backend = SimpleNamespace(token_to_kv_pool=pool)
    for layer in layers:
        for current in (reference, pool):
            current.set_mla_kv_global_scale(layer.layer_id, 0.5 + layer.layer_id / 4)

    loc = torch.arange(65, 65 + query_count, device=device)
    nope = torch.empty(5, query_count, 1, 512, device=device, dtype=torch.bfloat16)
    rope = torch.empty(5, query_count, 1, 64, device=device, dtype=torch.bfloat16)
    index_keys = torch.empty(
        5, query_count, 128, device=device, dtype=torch.float8_e4m3fn
    )
    index_scales = torch.empty(5, query_count, 1, device=device)
    q = torch.empty(5, query_count, 1, 64, 576, device=device, dtype=torch.bfloat16)
    index_q = torch.empty(5, query_count, 128, device=device)
    valid = torch.zeros(SIZE + PAGE_SIZE, device=device, dtype=torch.bool)
    valid[1:65] = True
    metadata, num_splits = torch.ops.sgl_kernel.get_mla_decoding_metadata.default(
        torch.ones(query_count, device=device, dtype=torch.int32),
        64,
        1,
        64,
        True,
        2048,
    )

    def random_values(count):
        return (
            torch.randn(count, 1, 512, device=device, generator=generator).bfloat16(),
            torch.randn(count, 1, 64, device=device, generator=generator).bfloat16(),
            torch.randn(count, 128, device=device, generator=generator).to(
                torch.float8_e4m3fn
            ),
            torch.rand(count, 1, device=device, generator=generator) + 0.5,
        )

    def prefill_write(write_loc):
        for layer in layers:
            values = random_values(write_loc.numel())
            for current in (reference, pool):
                current.set_mla_kv_buffer(layer, write_loc, values[0], values[1])
                current.index_key_cache.store_quantized(
                    layer.layer_id, write_loc, values[2], values[3]
                )
        # Leave the eager scratch tagged layer 0, unlike the graph's final
        # layer 4. The production forward boundary must invalidate it.
        pool.prefetch_kv_buffer(0)
        torch.testing.assert_close(
            pool.get_key_buffer(0), reference.get_key_buffer(0), rtol=0, atol=0
        )
        torch.testing.assert_close(
            pool.index_key_cache.get_buffer(0),
            reference.index_key_cache.get_buffer(0),
            rtol=0,
            atol=0,
        )

    def fill_step(step):
        loc.copy_(torch.arange(query_count, device=device) + 65 + step * query_count)
        valid[loc] = True
        for layer in layers:
            layer_id = layer.layer_id
            values = random_values(query_count)
            nope[layer_id].copy_(values[0])
            rope[layer_id].copy_(values[1])
            index_keys[layer_id].copy_(values[2])
            index_scales[layer_id].copy_(values[3])
        q.copy_(torch.randn(q.shape, device=device, generator=generator).bfloat16())
        index_q.copy_(torch.randn(index_q.shape, device=device, generator=generator))

    def select_indices(current, layer_id):
        # Independently unpack the production page layout. This small
        # indexer uses 16 candidates so changing the stored K/scales changes
        # the selected tokens. Transport still uses the native 2048 width.
        raw = (
            current.get_owned_index_buffer(layer_id)
            if current is pool
            else current.index_key_cache.get_buffer(layer_id)
        )
        keys = raw[:, : PAGE_SIZE * 128].reshape(-1, 128)
        keys = keys.view(torch.float8_e4m3fn).float()
        scales = raw[:, PAGE_SIZE * 128 :].contiguous().view(torch.float32)
        keys = keys * scales.reshape(-1, 1)
        logits = index_q[layer_id] @ keys.T
        logits.masked_fill_(~valid.unsqueeze(0), -torch.inf)
        selected = torch.topk(logits, 16, dim=-1).indices.to(torch.int32)
        indices = torch.full((query_count, 2048), -1, device=device, dtype=torch.int32)
        indices[:, :16] = selected
        return indices

    def decode(current, layer_id, indices):
        packed = (
            current.get_owned_key_buffer(layer_id)
            if current is pool
            else current.get_key_buffer(layer_id)
        )
        return torch.ops.sgl_kernel.sparse_decode_fwd_nvfp4.default(
            q[layer_id],
            packed.view(-1, 64, 1, 416),
            current.get_mla_kv_global_scale(layer_id),
            indices.unsqueeze(1),
            None,
            None,
            metadata,
            num_splits,
            512,
            576**-0.5,
        )[0]

    def forward(current):
        result = []
        for layer in layers:
            layer_id = layer.layer_id
            if current is pool:
                current.run_on_layer_owner(
                    layer_id,
                    lambda layer=layer, layer_id=layer_id: (
                        current.set_owned_mla_kv_buffer(
                            layer, loc, nope[layer_id], rope[layer_id]
                        )
                    ),
                    shape=None,
                    dtype=torch.uint8,
                    device=device,
                )
            else:
                current.set_mla_kv_buffer(layer, loc, nope[layer_id], rope[layer_id])
            current.index_key_cache.store_quantized(
                layer_id, loc, index_keys[layer_id], index_scales[layer_id]
            )
            if current is pool:
                indices = current.run_on_layer_owner(
                    layer_id,
                    lambda layer_id=layer_id: select_indices(current, layer_id),
                    shape=(query_count, 2048),
                    dtype=torch.int32,
                    device=device,
                )
                output = DeepseekSparseAttnBackend._forward_on_layer_owner(
                    backend,
                    q[layer_id],
                    None,
                    layer,
                    lambda layer_id=layer_id, indices=indices: decode(
                        current, layer_id, indices
                    ),
                )
            else:
                indices = select_indices(current, layer_id)
                output = decode(current, layer_id, indices)
            # If the helper reuses a result buffer, keep each layer's own
            # output before the next layer broadcasts into that buffer.
            result.append((indices.clone(), output.clone()))
        return result

    prefill_write(torch.arange(1, 65, device=device))
    fill_step(0)
    warmup_stream = torch.cuda.Stream(device=device)
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(2):
            pool.prepare_owner_compute()
            forward(pool)
            forward(reference)
    torch.cuda.current_stream().wait_stream(warmup_stream)
    torch.cuda.synchronize()
    torch.distributed.barrier()
    pool.prepare_owner_compute()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = forward(pool)

    for step in range(4):
        # Interleave a prefill-like mutation and scratch materialization
        # before every replay, including an existing prefix token.
        prefill_write(torch.tensor([31 + step], device=device))
        fill_step(step)
        pool.prepare_owner_compute()
        graph.replay()
        expected = forward(reference)
        torch.cuda.synchronize()
        for layer_id, (actual_pair, expected_pair) in enumerate(
            zip(captured, expected)
        ):
            torch.testing.assert_close(actual_pair[0], expected_pair[0], rtol=0, atol=0)
            torch.testing.assert_close(
                actual_pair[1], expected_pair[1], atol=8e-4, rtol=2.01 / 128
            )
            if pool._is_layer_owned(layer_id):
                assert torch.equal(
                    pool.get_owned_key_buffer(layer_id),
                    reference.get_key_buffer(layer_id),
                )
                assert torch.equal(
                    pool.get_owned_index_buffer(layer_id),
                    reference.index_key_cache.get_buffer(layer_id),
                )
        # Eager reads after graph execution must also materialize the
        # current owner bytes rather than reuse a capture-time scratch tag.
        pool.prepare_owner_compute()
        for layer in layers:
            assert torch.equal(
                pool.get_key_buffer(layer.layer_id),
                reference.get_key_buffer(layer.layer_id),
            )
            assert torch.equal(
                pool.index_key_cache.get_buffer(layer.layer_id),
                reference.index_key_cache.get_buffer(layer.layer_id),
            )
    torch.distributed.barrier()
    print(
        f"[rank {rank}] NVFP4 owner graph CP={world} Q={query_count}: "
        "four replays, eager prefixes, index/KV bytes and native outputs OK",
        flush=True,
    )


def _run_nvfp4_owner_graph(rank: int, world: int):
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    for query_count, mode in ((1, ForwardMode.DECODE), (6, ForwardMode.TARGET_VERIFY)):
        _run_nvfp4_owner_graph_case(rank, world, query_count, mode)


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

    def test_nvfp4_owner_compute_decode_and_verify_graph_replay(self):
        if torch.cuda.device_count() < 2:
            self.skipTest("NVFP4 owner graph test needs >= 2 GPUs")
        if torch.cuda.get_device_capability()[0] != 10:
            self.skipTest("NVFP4 DSA requires SM100")
        for world in (2, 4):
            if torch.cuda.device_count() >= world:
                with self.subTest(world=world):
                    mp.spawn(
                        _run,
                        args=(world, PORT + 20 + world, False, False, True),
                        nprocs=world,
                        join=True,
                    )


if __name__ == "__main__":
    unittest.main()
