"""Multi-GPU isolation test for DecodeKVBroadcaster (SGLANG_ENABLE_DISAGG_MLA_DECODE_KV_BROADCAST).

Spawns ``world`` processes forming a single attention-TP group, builds a tiny
``DSATokenToKVPool`` on each rank, writes distinct KV/indexer-K values only on
the source rank (attn TP rank 0), fills the peer ranks with sentinels, then
calls ``DecodeKVBroadcaster.broadcast`` directly -- no scheduler, no server,
no transfer engine -- and verifies every peer rank ends up with the source
rank's bytes.

Registered as a base-b 2-gpu-large unit test; skips when fewer than 2 GPUs
are visible. Run directly on 2 GPUs:
    CUDA_VISIBLE_DEVICES=0,1 python -m pytest \
        test/registered/unit/disaggregation/test_decode_kv_broadcast_multi_gpu.py
"""

import os
import unittest

import torch
import torch.multiprocessing as mp

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b", runner_config="2-gpu-large")

LAYER_NUM = 2
PAGE_SIZE = 64
KV_LORA_RANK = 512
QK_ROPE = 64
INDEX_HEAD_DIM = 128
SIZE = 256  # kv token rows
NUM_ROWS = 8
NUM_PAGES = 4
SENTINEL = -1.0
PORT = 29723


def _run(rank: int, world: int, port: int):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    os.environ.setdefault("no_proxy", "127.0.0.1,localhost")
    torch.cuda.set_device(rank)

    from sglang.srt.disaggregation.decode_kv_broadcast import DecodeKVBroadcaster
    from sglang.srt.distributed.device_communicators.pynccl import PyNcclCommunicator
    from sglang.srt.distributed.parallel_state import (
        get_attn_tp_group,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool

    init_distributed_environment(
        world_size=world,
        rank=rank,
        local_rank=rank,
        distributed_init_method=f"tcp://127.0.0.1:{port}",
        backend="nccl",
    )
    initialize_model_parallel(tensor_model_parallel_size=world)

    device = f"cuda:{rank}"
    pool = DSATokenToKVPool(
        SIZE,
        page_size=PAGE_SIZE,
        kv_lora_rank=KV_LORA_RANK,
        dtype=torch.bfloat16,
        qk_rope_head_dim=QK_ROPE,
        layer_num=LAYER_NUM,
        device=device,
        index_head_dim=INDEX_HEAD_DIM,
        enable_memory_saver=False,
        kv_cache_dim=KV_LORA_RANK + QK_ROPE,
        index_buf_size=SIZE,
    )

    attn_tp_group = get_attn_tp_group()
    relay_comm = PyNcclCommunicator(
        group=attn_tp_group.cpu_group, device=attn_tp_group.device
    )
    broadcaster = DecodeKVBroadcaster(
        token_to_kv_pool=pool,
        draft_token_to_kv_pool=None,
        relay_comm=relay_comm,
        attn_tp_rank=attn_tp_group.rank_in_group,
        attn_tp_size=attn_tp_group.world_size,
        forward_stream=torch.cuda.current_stream(),
    )

    # Row 0 of kv_buffer is a padding sink, so valid rows start at 1.
    kv_indices = torch.arange(1, 1 + NUM_ROWS, device=device, dtype=torch.int64)
    state_indices = torch.arange(0, NUM_PAGES, device=device, dtype=torch.int64)

    for layer_id in range(LAYER_NUM):
        if rank == 0:
            pool.kv_buffer[layer_id][kv_indices] = float(layer_id + 1)
            pool.index_k_with_scale_buffer[layer_id][state_indices] = layer_id + 10
        else:
            pool.kv_buffer[layer_id][kv_indices] = SENTINEL
            pool.index_k_with_scale_buffer[layer_id][state_indices] = 0

    torch.cuda.synchronize()
    torch.distributed.barrier()

    # The collective every attn TP rank must call with matching indices.
    broadcaster.broadcast([kv_indices], [state_indices])
    torch.cuda.synchronize()

    ok = True
    for layer_id in range(LAYER_NUM):
        got_kv = pool.kv_buffer[layer_id][kv_indices].float().mean().item()
        expected_kv = float(layer_id + 1)
        if abs(got_kv - expected_kv) > 1e-3:
            print(
                f"[rank{rank}] kv layer {layer_id}: expected {expected_kv}, got {got_kv}"
            )
            ok = False

        got_idx = (
            pool.index_k_with_scale_buffer[layer_id][state_indices]
            .float()
            .mean()
            .item()
        )
        expected_idx = float(layer_id + 10)
        if abs(got_idx - expected_idx) > 1e-3:
            print(
                f"[rank{rank}] idx layer {layer_id}: expected {expected_idx}, got {got_idx}"
            )
            ok = False

    assert ok, f"rank {rank} did not receive the source rank's broadcast contents"
    print(f"[rank{rank}] OK: received source rank's KV and indexer-K contents")
    torch.distributed.barrier()


class TestDecodeKVBroadcast(CustomTestCase):
    def test_broadcast(self):
        world = min(2, torch.cuda.device_count())
        if world < 2:
            self.skipTest("DecodeKVBroadcaster test needs >= 2 GPUs")
        mp.spawn(_run, args=(world, PORT), nprocs=world, join=True)


if __name__ == "__main__":
    unittest.main()
