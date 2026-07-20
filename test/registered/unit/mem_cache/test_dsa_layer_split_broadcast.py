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

import torch
import torch.multiprocessing as mp

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=25, stage="base-c", runner_config="4-gpu-b200")

LAYER_NUM = 4
PAGE_SIZE = 64
KV_LORA_RANK = 512
QK_ROPE = 64
INDEX_HEAD_DIM = 128
SIZE = PAGE_SIZE * 3  # a few pages
PORT = 29711


def _run(rank: int, world: int, port: int):
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
    initialize_model_parallel(
        tensor_model_parallel_size=world,
        attention_context_model_parallel_size=world,
    )

    from sglang.srt.mem_cache.dsa_cache_layer_split import (
        LayerSplitDSATokenToKVPool,
    )

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
        if pool._is_layer_owned(layer_id):
            pool.index_k_with_scale_buffer[layer_id].fill_(layer_id + 10)
    torch.cuda.synchronize()
    torch.distributed.barrier()
    for layer_id in range(LAYER_NUM):
        # invalidate any cached remote copy so the read forces a fresh broadcast
        pool.invalidate_index_buffer_for_layer(layer_id)
        buf = pool._get_broadcastable_index_buffer(layer_id)
        expected = layer_id + 10
        got = buf.float().mean().item()
        if abs(got - expected) > 1e-3:
            print(f"[rank {rank}] index layer {layer_id}: exp {expected}, got {got}")
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


class TestLayerSplitDSABroadcast(CustomTestCase):
    def test_owner_broadcast(self):
        world = min(4, torch.cuda.device_count())
        if world < 2:
            self.skipTest("LayerSplitDSATokenToKVPool broadcast test needs >= 2 GPUs")
        mp.spawn(_run, args=(world, PORT), nprocs=world, join=True)


if __name__ == "__main__":
    unittest.main()
