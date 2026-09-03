"""Multi-NPU integration test for LayerSplitDSV4NPUTokenToKVPool owner-broadcast.

Spawns ``world`` processes forming a single attention-CP group, builds a tiny
layer-split DSV4 NPU pool on each rank, writes the owner's bytes into every
owned buffer family, then verifies that reading ANY layer (owned or not)
surfaces the *owning* rank's contents -- i.e. the per-family owner broadcast
over the CP group works for the SWA / c4 / c128 KV buffers and the c4 indexer
K/scale buffers. Also checks the allocation shape contract (owned layers full,
non-owned 0-row placeholders) and that the PD buffer reports list owned layers
only.

Run directly on 2+ NPUs:
    ASCEND_RT_VISIBLE_DEVICES=0,1 python -m pytest \\
        test/registered/unit/hardware_backend/npu/test_dsv4_layer_split_broadcast.py
"""

import os
import unittest

import torch
import torch.multiprocessing as mp

from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=120, suite="per-commit-2-npu-a2")

LAYER_NUM = 4
RATIOS = [4, 0, 4, 128]
PAGE_SIZE = 128
SWA_PAGE = 128
C128_KERNEL_PAGE = 16
NOPE_DIM = 448
ROPE_DIM = 64
INDEX_HEAD_DIM = 128
SWA_SIZE = SWA_PAGE * 2
C4_SIZE = 8
C128_SIZE = 8
PORT = int(os.environ.get("DSV4_LS_PORT", "29811"))


def _setup_context_bags():
    """Publish the minimal config leaves the DSV4 pool constructors read."""
    from sglang.srt.runtime_context import _ConfigBag, get_context

    schedule = _ConfigBag("schedule")
    schedule._set("page_size", PAGE_SIZE)
    schedule._set("c128_page_size", C128_KERNEL_PAGE)
    spec = _ConfigBag("spec")
    spec._set("speculative_algorithm", None)
    exec_bag = _ConfigBag("exec")
    kernel = _ConfigBag("kernel")
    kernel._set("enable_deepseek_v4_fp4_indexer", False)
    exec_bag._set_sub("kernel", kernel)
    get_context()._config_bags = {
        "schedule": schedule,
        "spec": spec,
        "exec": exec_bag,
    }


def _build_pool(rank, cp_size, device):
    from sglang.srt.hardware_backend.npu.dsv4.dsv4_cache_layer_split import (
        LayerSplitDSV4NPUTokenToKVPool,
    )

    return LayerSplitDSV4NPUTokenToKVPool(
        max_num_reqs=4,
        num_req_slots=5,
        swa_size=SWA_SIZE,
        c4_size=C4_SIZE,
        c128_size=C128_SIZE,
        c4_state_pool_size=16,
        c128_state_pool_size=256,
        page_size=PAGE_SIZE,
        swa_page_size=SWA_PAGE,
        sliding_window=128,
        dtype=torch.bfloat16,
        c4_state_dtype=torch.float32,
        c128_state_dtype=torch.float32,
        qk_nope_head_dim=NOPE_DIM,
        qk_rope_head_dim=ROPE_DIM,
        indexer_head_dim=INDEX_HEAD_DIM,
        layer_num=LAYER_NUM,
        device=device,
        enable_memory_saver=False,
        compression_ratios=RATIOS,
        layer_shard_rank=rank,
        layer_shard_size=cp_size,
    )


def _fill_owned(pool, rank):
    """Write rank-distinct constants into every owned buffer family."""
    for layer_id in range(LAYER_NUM):
        if pool._is_layer_owned(layer_id):
            pool.swa_kv_pool.kv_buffer[layer_id].fill_(float(layer_id + 1))
        item = pool.layer_mapping[layer_id]
        if item.compress_ratio == 4 and pool._is_layer_owned(layer_id):
            pool.c4_kv_pool.kv_buffer[item.compress_layer_id].fill_(10.0 + layer_id)
            pool.c4_indexer_kv_pool.index_k_buffer[item.compress_layer_id].fill_(
                20.0 + layer_id
            )
            pool.c4_indexer_kv_pool.index_scale_buffer[item.compress_layer_id].fill_(
                30.0 + layer_id
            )
        if item.compress_ratio == 128 and pool._is_layer_owned(layer_id):
            pool.c128_kv_pool.kv_buffer[item.compress_layer_id].fill_(10.0 + layer_id)


def _run(rank: int, world: int, port: int):
    import torch_npu  # noqa: F401

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    os.environ.setdefault("no_proxy", "127.0.0.1,localhost")
    torch.npu.set_device(rank)

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
        backend="hccl",
    )
    initialize_model_parallel(
        tensor_model_parallel_size=world,
        attention_context_model_parallel_size=world,
    )
    _setup_context_bags()

    cp_rank = get_parallel().attn_cp_rank
    cp_size = get_parallel().attn_cp_size
    assert cp_size == world, (cp_size, world)

    pool = _build_pool(cp_rank, cp_size, f"npu:{rank}")

    # --- allocation shape checks: owned layers full, others 0-row ----------
    owned = [pool._is_layer_owned(i) for i in range(LAYER_NUM)]
    start = owned.index(True) if any(owned) else LAYER_NUM
    end = LAYER_NUM - owned[::-1].index(True) if any(owned) else 0
    assert owned == [start <= i < end for i in range(LAYER_NUM)], owned
    assert sum(owned) in (LAYER_NUM // world, LAYER_NUM // world + 1)
    swa_pages = (SWA_SIZE + SWA_PAGE + 1) // SWA_PAGE
    for layer_id in range(LAYER_NUM):
        expect = swa_pages if owned[layer_id] else 0
        actual = pool.swa_kv_pool.kv_buffer[layer_id].shape[0]
        assert actual == expect, (layer_id, actual, expect)
    for layer_id in (0, 2):  # c4 layers
        item = pool.layer_mapping[layer_id]
        expect = (C4_SIZE + 32 + 1) // 32 if pool._is_layer_owned(layer_id) else 0
        assert pool.c4_kv_pool.kv_buffer[item.compress_layer_id].shape[0] == expect
        # The indexer pool is sized by c4_logical_size (= c128_size * 32).
        indexer_pages = (C128_SIZE * 32 + 32 + 1) // 32
        expect_k = indexer_pages if pool._is_layer_owned(layer_id) else 0
        assert (
            pool.c4_indexer_kv_pool.index_k_buffer[item.compress_layer_id].shape[0]
            == expect_k
        )
        assert (
            pool.c4_indexer_kv_pool.index_scale_buffer[item.compress_layer_id].shape[0]
            == expect_k
        )
    c128_item = pool.layer_mapping[3]
    expect128 = (
        (C128_SIZE + C128_KERNEL_PAGE + 1) // C128_KERNEL_PAGE if owned[3] else 0
    )
    assert (
        pool.c128_kv_pool.kv_buffer[c128_item.compress_layer_id].shape[0] == expect128
    )

    # --- owned-only PD buffer reports ---------------------------------------
    kv_ptrs, _, _ = pool.get_contiguous_buf_infos()
    owned_c4 = sum(1 for i in (0, 2) if pool._is_layer_owned(i))
    assert len(kv_ptrs) == 3 * owned_c4, (len(kv_ptrs), owned_c4)
    state_ptrs, _, _ = pool.get_state_buf_infos()
    owned_swa = sum(owned)
    assert len(state_ptrs) == owned_swa + 2 * owned_c4
    c128_ptrs, _, _ = pool.get_c128_kv_buf_infos()
    assert len(c128_ptrs) == (1 if owned[3] else 0)
    c128_state_ptrs, _, _ = pool.get_c128_state_buf_infos()
    assert len(c128_state_ptrs) == (1 if owned[3] else 0)
    assert (pool.layer_shard_start, pool.layer_shard_end) == (start, end)

    # --- owner broadcast: every rank reads the owner's bytes ---------------
    _fill_owned(pool, cp_rank)
    torch.npu.synchronize()
    torch.distributed.barrier()

    ok = True
    for layer_id in range(LAYER_NUM):
        got = pool.get_swa_buffer(layer_id).float().mean().item()
        expected = float(layer_id + 1)
        if abs(got - expected) > 1e-3:
            print(f"[rank {rank}] swa layer {layer_id}: exp {expected}, got {got}")
            ok = False
        item = pool.layer_mapping[layer_id]
        if item.compress_ratio == 4:
            got = pool.get_compress_buffer(layer_id).float().mean().item()
            if abs(got - (10.0 + layer_id)) > 1e-3:
                print(f"[rank {rank}] c4 layer {layer_id}: got {got}")
                ok = False
            got_k = pool.get_compress_buffer(layer_id, True).float().mean().item()
            got_s = pool.get_compress_dequant_scale_buffer(layer_id, True)
            got_s = got_s.float().mean().item()
            if abs(got_k - (20.0 + layer_id)) > 1e-3 or abs(
                got_s - (30.0 + layer_id)
            ) > 1e-3:
                print(f"[rank {rank}] index layer {layer_id}: k {got_k} s {got_s}")
                ok = False
        if item.compress_ratio == 128:
            got = pool.get_compress_buffer(layer_id).float().mean().item()
            if abs(got - (10.0 + layer_id)) > 1e-3:
                print(f"[rank {rank}] c128 layer {layer_id}: got {got}")
                ok = False
    assert ok, f"rank {rank} read stale/incorrect broadcast contents"

    # --- non-owned writes are no-ops; owner writes re-broadcast ------------
    for layer_id in range(LAYER_NUM):
        pool.set_swa_buffer(
            layer_id=layer_id,
            loc=torch.arange(1, device=f"npu:{rank}"),
            cache=torch.full(
                (1, 1, NOPE_DIM + ROPE_DIM), 99.0, dtype=torch.bfloat16,
                device=f"npu:{rank}",
            ),
        )
    torch.npu.synchronize()
    torch.distributed.barrier()
    for layer_id in range(LAYER_NUM):
        buf = pool.get_swa_buffer(layer_id)
        got = buf.float().mean().item()
        # The owner's write (99 into row 1) must be visible on every rank: the
        # owner reads its local buffer, a non-owner re-broadcasts after its own
        # (no-op) write call invalidated the cached copy.
        rows = buf.shape[0] * buf.shape[1]
        expected = (float(layer_id + 1) * (rows - 1) + 99.0) / rows
        if abs(got - expected) > 1e-2:
            print(f"[rank {rank}] post-write layer {layer_id}: exp {expected}, got {got}")
            ok = False
    assert ok, f"rank {rank} post-write contents incorrect"

    print(f"[rank {cp_rank}] OK: allocation, PD reports and owner broadcast verified")
    torch.distributed.barrier()


class TestLayerSplitDSV4Broadcast(CustomTestCase):
    def test_owner_broadcast(self):
        world = min(
            int(os.environ.get("DSV4_LS_WORLD", "2")), torch.npu.device_count()
        )
        if world < 2:
            self.skipTest("LayerSplitDSV4NPUTokenToKVPool broadcast test needs >= 2 NPUs")
        mp.spawn(_run, args=(world, PORT), nprocs=world, join=True)


if __name__ == "__main__":
    unittest.main()
