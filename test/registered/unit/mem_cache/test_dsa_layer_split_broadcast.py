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
    prefill_pages: bool = False,
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

    if prefill_pages:
        _run_nvfp4_prefill_pages(rank, world)
        return
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
        # Enter the real partial-plan boundary after an owner graph replay.
        # Page zero used by the legacy codec fixture is not a request page.
        table = torch.arange(64, 128, device=device).view(1, -1)
        pool.begin_prefill_page_plan(
            table, torch.tensor([64]), [1], table[:, -1:].flatten()
        )
        pool.prefetch_prefill_kv_buffer(0)
        torch.testing.assert_close(
            pool.get_prefill_key_buffer(0)[64:128],
            reference.get_key_buffer(0)[64:128],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            pool.index_key_cache.get_prefill_buffer(0)[1],
            reference.index_key_cache.get_buffer(0)[1],
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


def _run_nvfp4_prefill_pages(rank: int, world: int):
    """Actual metadata, FP8 indexer and BF16 FlashMLA over partial NVFP4 pages.

    The fixture supplies final projected/rotated Q and globally gathered K.
    It does not test model projections, RoPE/rotation or the CP key gather.
    Cache writes, page-plan construction, CP request filtering, quantization,
    logits/topk, dequantization and attention use production entry points and
    kernels.
    """
    import json
    from unittest.mock import patch

    import sgl_kernel
    from sglang.srt.layers.attention.dsa import dsa_indexer, dsa_topk_backend
    from sglang.srt.layers.attention.dsa.dsa_indexer import Indexer
    from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend
    from sglang.srt.layers.cp import get_cp_strategy, init_cp_strategy
    from sglang.srt.mem_cache.dsa_cache_layer_split import LayerSplitDSATokenToKVPool
    from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
    from sglang.srt.model_executor.forward_context import (
        ForwardContext,
        forward_context,
    )
    from sglang.srt.runtime_context import get_context, get_parallel

    device = torch.device("cuda", rank)
    size, layer_num = 64 * 64, 5
    skip = [False, True, False, True, False]
    kwargs = {
        "size": size,
        "page_size": 64,
        "kv_lora_rank": 512,
        "dtype": torch.float4_e2m1fn_x2,
        "qk_rope_head_dim": 64,
        "layer_num": layer_num,
        "device": str(device),
        "index_head_dim": 128,
        "enable_memory_saver": False,
        "kv_cache_dim": 416,
        "skip_topk_layers": skip,
    }
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            architectures=["GlmMoeDsaForCausalLM"], index_topk=2048
        ),
        context_len=2304,
        num_attention_heads=64,
        qk_nope_head_dim=192,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
    )
    logits_calls, topk_routes = [], []
    resolve_logits = dsa_indexer.resolve_fp8_mqa_logits_fn
    v2_topk = dsa_topk_backend._topk_transform_v2_ragged
    legacy_topk = sgl_kernel.fast_topk_transform_ragged_fused

    def traced_logits_resolver():
        kernel = resolve_logits()

        def call(query, cache, weights, starts, ends, **kwargs):
            logits = kernel(query, cache, weights, starts, ends, **kwargs)
            # DeepGEMM leaves columns outside [ks, ke) uninitialized. Snapshot
            # only valid windows, before top-k masks alignment padding in-place.
            windows = list(zip(starts.cpu().tolist(), ends.cpu().tolist()))
            logits_calls.append(
                {
                    "kernel": kernel,
                    "query": query.clone(),
                    "key": cache[0].clone(),
                    "scale": cache[1].clone(),
                    "weights": weights.clone(),
                    "starts": starts.clone(),
                    "ends": ends.clone(),
                    "windows": windows,
                    "logits": [
                        logits[row, start:end].clone()
                        for row, (start, end) in enumerate(windows)
                    ],
                    "kwargs": kwargs,
                }
            )
            return logits

        return call

    def traced_v2_topk(*args, **kwargs):
        topk_routes.append("v2_ragged")
        return v2_topk(*args, **kwargs)

    def traced_legacy_topk(*args, **kwargs):
        topk_routes.append("legacy_ragged_fused")
        return legacy_topk(*args, **kwargs)

    def same_bytes(left, right):
        return torch.equal(
            left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8)
        )

    with (
        get_context().override_server_args(
            # This publishes a fresh config, while initialized rank/group
            # handles survive. Preserve their topology instead of publishing
            # the dummy config's default TP=1 underneath the live CP group.
            tp_size=world,
            attn_cp_size=world,
            dp_size=1,
            pp_size=1,
            ep_size=1,
            moe_dp_size=1,
            enable_prefill_cp=True,
            cp_strategy="interleave",
            dsa_prefill_backend="flashmla_sparse",
            dsa_decode_backend="flashmla_kv",
            speculative_num_draft_tokens=6,
            speculative_eagle_topk=1,
        ),
        get_parallel().override(
            tp_size=world,
            enable_prefill_cp=True,
            cp_strategy="interleave",
            attn_cp_size=world,
            attn_tp_size=1,
            attn_dp_size=1,
            moe_tp_size=world,
            moe_ep_size=1,
            moe_dp_size=1,
            attn_dcp_size=1,
            attn_dcp_rank=0,
        ),
        # The fixture has no checkpoint. Publish its architecture at the model
        # config boundary; leave every CP eligibility and metadata check live.
        patch(
            "sglang.srt.arg_groups.model_override_base.model_config_of",
            return_value=model_config,
        ),
        patch(
            "sglang.srt.arg_groups.overrides.model_config_of", return_value=model_config
        ),
        patch.object(dsa_indexer, "resolve_fp8_mqa_logits_fn", traced_logits_resolver),
        patch.object(dsa_topk_backend, "_topk_transform_v2_ragged", traced_v2_topk),
        patch.object(
            sgl_kernel, "fast_topk_transform_ragged_fused", traced_legacy_topk
        ),
    ):
        init_cp_strategy(
            enable_prefill_cp=True, cp_size=world, cp_strategy="interleave"
        )
        strategy = get_cp_strategy()
        reference = DSATokenToKVPool(**kwargs)
        pool = LayerSplitDSATokenToKVPool(
            **kwargs, layer_shard_rank=rank, layer_shard_size=world, unified_decode=True
        )
        req_table = torch.zeros((4, 2304), dtype=torch.int32, device=device)

        def backend_for(current):
            return DeepseekSparseAttnBackend(
                SimpleNamespace(
                    device=str(device),
                    page_size=64,
                    model_config=model_config,
                    token_to_kv_pool=current,
                    req_to_token_pool=SimpleNamespace(size=4, req_to_token=req_table),
                    hisparse_coordinator=None,
                    is_draft_worker=False,
                    kv_cache_dtype=torch.float4_e2m1fn_x2,
                    max_running_requests=4,
                )
            )

        backends = [backend_for(reference), backend_for(pool)]
        layers = [
            SimpleNamespace(
                layer_id=i,
                head_dim=576,
                v_head_dim=512,
                tp_q_head_num=64,
                is_cross_attention=False,
                scaling=576**-0.5,
            )
            for i in range(layer_num)
        ]
        indexer = object.__new__(Indexer)
        indexer.index_topk = 2048
        indexer.alt_stream = None
        indexer.dsa_enable_prefill_cp = True
        indexer.use_gfx950_fused_indexer = False
        indexer.use_dsa_indexer_fusion = False
        indexer.use_aiter_fused_fp8_writer = False
        indexer.weights_proj = SimpleNamespace()
        indexer.block_size = 128
        indexer.scale_fmt = None
        indexer.n_heads = 32
        indexer.softmax_scale = 128**-0.5
        indexer.logits_with_pp_recv = False
        indexer.num_init_tokens = indexer.num_local_tokens = 0
        indexer._mqa_logits_budget_bytes = {}
        indexer._aiter_fused_fp8_active = lambda batch: False
        generator = torch.Generator(device=device).manual_seed(1207)
        physical = torch.arange(64, size + 64, device=device)
        # Seed all allocated valid pages, including pages never in a plan. A
        # same-size ordinary pool is an independent persistent-byte oracle.
        for layer in layers:
            nope = torch.randn(
                size, 1, 512, device=device, generator=generator
            ).bfloat16()
            rope = torch.randn(
                size, 1, 64, device=device, generator=generator
            ).bfloat16()
            index_key = torch.randn(
                size, 128, device=device, generator=generator
            ).bfloat16()
            for backend in backends:
                current = backend.token_to_kv_pool
                current.set_mla_kv_global_scale(
                    layer.layer_id, 0.5 + layer.layer_id * 0.375
                )
                current.set_mla_kv_buffer(layer, physical, nope, rope)
                if not skip[layer.layer_id]:
                    with forward_context(ForwardContext(attn_backend=backend)):
                        indexer._store_index_k_cache(
                            SimpleNamespace(out_cache_loc=physical),
                            layer.layer_id,
                            index_key,
                        )

        # A shared prefix >2048 makes topk depend on index bytes. Short requests
        # assign different request subsets to CP ranks; no rank invents its own
        # plan from those filtered tables. The last step extends existing pages.
        cases = [(0, [1, 2, 5, 8]), (2112, [1, 2, 5, 8]), (2112, [4, 3, 13, 12])]
        for case_id, (prefix, lengths) in enumerate(cases):
            previous = [0] * 4 if case_id < 2 else [1, 2, 5, 8]
            extend = [n - old for n, old in zip(lengths, previous)]
            req_table.zero_()
            for request, tail_page in enumerate((40, 42, 44, 46)):
                if prefix:
                    req_table[request, :prefix] = torch.arange(
                        64, prefix + 64, device=device
                    )
                req_table[request, prefix : prefix + 64] = torch.arange(
                    tail_page * 64, (tail_page + 1) * 64, device=device
                )
            seq_cpu = torch.tensor([prefix + n for n in lengths])
            out_loc = torch.cat(
                [
                    req_table[i, prefix + old : prefix + n]
                    for i, (old, n) in enumerate(zip(previous, lengths))
                ]
            ).long()
            total = sum(extend)
            batch = ForwardBatch(
                forward_mode=ForwardMode.EXTEND,
                batch_size=4,
                input_ids=torch.zeros(total, device=device, dtype=torch.int64),
                req_pool_indices=torch.arange(4, device=device),
                seq_lens=seq_cpu.to(device),
                out_cache_loc=out_loc,
                seq_lens_sum=int(seq_cpu.sum()),
            )
            batch.seq_lens_cpu = seq_cpu
            batch.extend_seq_lens_cpu = extend
            batch.extend_seq_lens = torch.tensor(
                extend, device=device, dtype=torch.int32
            )
            batch.extend_prefix_lens_cpu = [prefix + old for old in previous]
            batch.extend_num_tokens = total
            batch.attn_cp_metadata = strategy.build_metadata(
                total, seq_cpu.tolist(), extend
            )
            for backend in backends:
                backend.init_forward_metadata(batch)
            expected_pages = list(range(1, 34)) if prefix else []
            expected_pages += [40, 42, 44, 46]
            assert (
                pool.prefill_page_ids[: pool.prefill_num_pages].cpu().tolist()
                == expected_pages
            )
            assert pool.prefill_page_plan_active
            ids = pool.prefill_page_ids[: pool.prefill_num_pages]
            # Explicitly prove at least one request is absent on some rank.
            local_requests = backends[1].forward_metadata.page_table_1.shape[0]
            if case_id == 0:
                counts = [None] * world
                torch.distributed.all_gather_object(counts, local_requests)
                assert min(counts) < 4
            previous_topk = [None, None]
            for layer in layers:
                layer_id = layer.layer_id
                nope = torch.randn(
                    total, 1, 512, device=device, generator=generator
                ).bfloat16()
                rope = torch.randn(
                    total, 1, 64, device=device, generator=generator
                ).bfloat16()
                key = torch.randn(
                    total, 128, device=device, generator=generator
                ).bfloat16()
                q = strategy.shard_local_tokens(
                    torch.randn(
                        total, 64, 576, device=device, generator=generator
                    ).bfloat16()
                )
                index_q = strategy.shard_local_tokens(
                    torch.randn(
                        total, 32, 128, device=device, generator=generator
                    ).bfloat16()
                )
                indexer._get_q_k_bf16 = lambda *args, query=index_q, keys=key, **kw: (
                    query,
                    keys,
                    None,
                )
                # Substitute the learned head projection only. Quantization of
                # Q and the real FP8 logits/topk kernels still run below.
                indexer._get_logits_head_gate = (
                    lambda x, scale: scale * indexer.softmax_scale
                )
                x = torch.zeros(q.shape[0], 4, device=device, dtype=torch.bfloat16)
                pool.remote_kv_buffer.fill_(197)
                pool.index_key_cache.remote_buffer.fill_(211)
                outputs, topks = [], []
                logits_calls.clear()
                topk_routes.clear()
                for which, backend in enumerate(backends):
                    with forward_context(ForwardContext(attn_backend=backend)):
                        topk = (
                            previous_topk[which]
                            if skip[layer_id]
                            else indexer.forward_cuda(
                                x,
                                x,
                                torch.arange(x.shape[0], device=device),
                                batch,
                                layer_id,
                            )
                        )
                        previous_topk[which] = topk
                        topks.append(topk)
                        outputs.append(
                            backend.forward_extend(
                                q,
                                nope,
                                nope,
                                layer,
                                batch,
                                k_rope=rope,
                                topk_indices=topk,
                            )
                        )
                context = f"case={case_id} prefix={prefix} layer={layer_id} CP={world} rank={rank}"
                evidence = {
                    "context": context,
                    "topk_backend": backends[0].dsa_topk_backend.value,
                    "topk_routes": list(topk_routes),
                    "logits_call_count": len(logits_calls),
                    "shared_topk": skip[layer_id],
                }
                actual = pool.get_prefill_key_buffer(layer_id).view(-1, 64, 416)
                expected = reference.get_key_buffer(layer_id).view(-1, 64, 416)
                evidence["kv_pages_exact"] = torch.equal(actual[ids], expected[ids])
                evidence["untransferred_kv_untouched"] = bool(
                    torch.all(actual[63] == 197)
                )
                if pool._is_layer_owned(layer_id):
                    evidence["owned_kv_exact"] = torch.equal(
                        pool.get_owned_key_buffer(layer_id),
                        reference.get_key_buffer(layer_id),
                    )
                if not skip[layer_id]:
                    idx = pool.index_key_cache.get_prefill_buffer(layer_id)
                    evidence["index_pages_exact"] = torch.equal(
                        idx[ids], reference.index_key_cache.get_buffer(layer_id)[ids]
                    )
                    evidence["untransferred_index_untouched"] = bool(
                        torch.all(idx[63] == 211)
                    )
                    if pool._is_layer_owned(layer_id):
                        evidence["owned_index_exact"] = torch.equal(
                            pool.get_owned_index_buffer(layer_id),
                            reference.index_key_cache.get_buffer(layer_id),
                        )
                    assert len(logits_calls) == 2, (context, len(logits_calls))
                    left, right = logits_calls
                    # These K/scale tensors are the real GetKAndS results fed
                    # to the same logits kernel, not independent reconstruction.
                    for field in ("query", "key", "scale", "weights", "starts", "ends"):
                        evidence[f"logits_{field}_exact"] = same_bytes(
                            left[field], right[field]
                        )
                    kernel_type = type(left["kernel"])
                    evidence["logits_kernel"] = (
                        f"{getattr(left['kernel'], '__module__', kernel_type.__module__)}.{getattr(left['kernel'], '__name__', kernel_type.__name__)}"
                    )
                    evidence["valid_logits_exact"] = all(
                        torch.equal(a, b)
                        for a, b in zip(left["logits"], right["logits"])
                    )
                    evidence["valid_logits_max_abs"] = max(
                        (
                            float((a - b).abs().max())
                            for a, b in zip(left["logits"], right["logits"])
                            if a.numel()
                        ),
                        default=0.0,
                    )
                    if not evidence["valid_logits_exact"]:
                        # Do not invent a DeepGEMM determinism requirement. Give
                        # the reviewer a same-input repeat to localize any drift.
                        repeat = left["kernel"](
                            left["query"],
                            (left["key"], left["scale"]),
                            left["weights"],
                            left["starts"],
                            left["ends"],
                            **left["kwargs"],
                        )
                        repeated_windows = [
                            repeat[row, start:end]
                            for row, (start, end) in enumerate(left["windows"])
                        ]
                        evidence["ordinary_repeat_valid_logits_exact"] = all(
                            torch.equal(a, b)
                            for a, b in zip(left["logits"], repeated_windows)
                        )
                        evidence["ordinary_repeat_valid_logits_max_abs"] = max(
                            (
                                float((a - b).abs().max())
                                for a, b in zip(left["logits"], repeated_windows)
                                if a.numel()
                            ),
                            default=0.0,
                        )
                ordered = [topk.sort(dim=-1).values for topk in topks]
                evidence["topk_raw_order_diff_count"] = int(
                    torch.count_nonzero(topks[0] != topks[1])
                )
                evidence["topk_sorted_multiset_exact"] = torch.equal(*ordered)
                print(
                    f"[rank {rank}] NVFP4 prefill evidence {json.dumps(evidence, sort_keys=True)}",
                    flush=True,
                )
                for name, value in evidence.items():
                    if (
                        name
                        in (
                            "kv_pages_exact",
                            "owned_kv_exact",
                            "index_pages_exact",
                            "owned_index_exact",
                            "untransferred_kv_untouched",
                            "untransferred_index_untouched",
                        )
                        or name.startswith("logits_")
                        and name.endswith("_exact")
                    ):
                        assert value, (context, name)
                # SGL top-k emits an unordered collection. Sort only the oracle
                # comparison, preserving duplicate entries and -1 padding.
                # Attention above receives each kernel's original order.
                torch.testing.assert_close(
                    ordered[0], ordered[1], rtol=0, atol=0, msg=context
                )
                torch.testing.assert_close(
                    outputs[0], outputs[1], atol=8e-4, rtol=2.01 / 128, msg=context
                )
                # Full public reads must still work after partial materialization.
                assert torch.equal(
                    pool.get_key_buffer(layer_id), reference.get_key_buffer(layer_id)
                )
                if not skip[layer_id]:
                    assert torch.equal(
                        pool.index_key_cache.get_buffer(layer_id),
                        reference.index_key_cache.get_buffer(layer_id),
                    )
        # Move and restore after a partial prefetch. Ordinary reads below are
        # deliberate full-cache checks, after verifying that each mutation
        # ended the plan rather than retaining stale scope tags.
        backends[1].init_forward_metadata(batch)
        pool.prefetch_prefill_kv_buffer(4)
        source = torch.tensor([64, 65, 40 * 64], device=device)
        target = torch.tensor([65, 66, 42 * 64], device=device)
        for current in (reference, pool):
            current.move_kv_cache(target, source)
        assert not pool.prefill_page_plan_active
        saved_loc = torch.arange(64, 192, device=device)
        snapshots = [current.get_cpu_copy(saved_loc) for current in (reference, pool)]
        backends[1].init_forward_metadata(batch)
        pool.prefetch_prefill_kv_buffer(4)
        for current, snapshot in zip((reference, pool), snapshots):
            for raw in current.kv_buffer:
                if raw.numel():
                    raw[saved_loc] = 0
            for raw in current.index_k_with_scale_buffer:
                if raw.numel():
                    raw[1:3] = 0
            current.load_cpu_copy(snapshot, saved_loc)
        assert not pool.prefill_page_plan_active
        for layer in layers:
            assert torch.equal(
                pool.get_key_buffer(layer.layer_id),
                reference.get_key_buffer(layer.layer_id),
            )
            if not skip[layer.layer_id]:
                assert torch.equal(
                    pool.index_key_cache.get_buffer(layer.layer_id),
                    reference.index_key_cache.get_buffer(layer.layer_id),
                )

        # Exercise the real two-chunk NCCL transfer and both stream workspaces.
        # CPU tests cover C-1/C/C+1; this C+1 case also reuses the main staging
        # buffer for a new plan while the previous prefetch is still pending.
        chunked = LayerSplitDSATokenToKVPool(
            **{**kwargs, "size": 1026 * 64},
            layer_shard_rank=rank,
            layer_shard_size=world,
            unified_decode=True,
        )
        for layer in layers:
            layer_id = layer.layer_id
            if chunked._is_layer_owned(layer_id):
                chunked.kv_buffer[layer_id].fill_(layer_id + 19)
                if not skip[layer_id]:
                    chunked.index_k_with_scale_buffer[layer_id].fill_(layer_id + 39)
        large_table = torch.arange(64, 1026 * 64, device=device).view(1, -1)
        for layer_id in (0, 2, 4):
            chunked.begin_prefill_page_plan(
                large_table,
                torch.tensor([large_table.numel()]),
                [1],
                large_table[:, -1:].flatten(),
            )
            chunked.remote_kv_buffer.fill_(197)
            chunked.index_key_cache.remote_buffer.fill_(211)
            chunked.prefetch_prefill_kv_buffer(layer_id)
            chunked.index_key_cache.prepare_prefill_buffer(layer_id)
            kv = chunked.get_prefill_key_buffer(layer_id).view(-1, 64, 416)
            idx = chunked.index_key_cache.get_prefill_buffer(layer_id)
            assert torch.all(kv[1:1026] == layer_id + 19)
            assert torch.all(idx[1:1026] == layer_id + 39)
            assert torch.all(kv[[0, 1026]] == 197)
            assert torch.all(idx[[0, 1026]] == 211)
        chunked.begin_prefill_page_plan(
            large_table,
            torch.tensor([large_table.numel()]),
            [1],
            large_table[:, -1:].flatten(),
        )
        chunked.prefetch_prefill_kv_buffer(0)
        small_table = large_table[:, :64]
        chunked.begin_prefill_page_plan(
            small_table, torch.tensor([64]), [1], small_table[:, -1:].flatten()
        )
        assert chunked.prefill_num_pages == 1
        assert torch.all(chunked.get_prefill_key_buffer(4)[64:128] == 23)
        chunked.prefetch_prefill_kv_buffer(0)
        scales = chunked.mla_kv_global_scale.clone()
        scale_ptr = chunked.mla_kv_global_scale.data_ptr()
        chunked._clear_buffers()
        chunked._create_buffers()
        chunked.index_key_cache = chunked._create_index_key_cache()
        assert not chunked.prefill_page_plan_active
        assert chunked.mla_kv_global_scale.data_ptr() == scale_ptr
        assert torch.equal(chunked.mla_kv_global_scale, scales)
        torch.cuda.synchronize()
        torch.distributed.barrier()
        print(
            f"[rank {rank}] NVFP4 prefill pages CP={world}: metadata, indexer, FlashMLA and lifecycle OK",
            flush=True,
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

    def test_nvfp4_prefill_page_plan_and_real_consumers(self):
        if torch.cuda.device_count() < 2:
            self.skipTest("NVFP4 prefill page test needs >= 2 GPUs")
        if torch.cuda.get_device_capability()[0] != 10:
            self.skipTest("NVFP4 DSA requires SM100")
        for world in (2, 4):
            if torch.cuda.device_count() >= world:
                with self.subTest(world=world):
                    mp.spawn(
                        _run,
                        args=(world, PORT + 30 + world, False, False, False, True),
                        nprocs=world,
                        join=True,
                    )


if __name__ == "__main__":
    unittest.main()
