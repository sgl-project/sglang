# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Two-GPU MTP KV-pool correctness, without model weights or a serving stack.

Target and single-layer MTP pools share logical token locations but own their
storage, prefix gathers, and scratch plans. Check every persisted physical row
and every assembled prefix/chunk row against canonical, exactly representable
BF16 values. Exercise MLA over TP and MHA over CP, local and nonzero draft layer
IDs, uneven owner counts, fragmented pages, and a partially filled final page.
Both MLA write APIs, repeated reads, and plan changes must preserve independent
target/draft state across batches. Draft pools use the real configurator and
share the target's allocator. This does not run a model forward or a P/D transfer.

Run: CUDA_VISIBLE_DEVICES=0,1 python test/manual/test_page_interleave_mtp.py
"""

import argparse
import os
from types import SimpleNamespace

import torch
import torch.multiprocessing as mp

WORLD = 2
PAGE_SIZE = 16
SIZE = 32 * PAGE_SIZE  # Physical token capacity of each pool on each rank.
TARGET_LAYERS = 4
KV_LORA_RANK = 128
QK_ROPE = 32
HEAD_NUM = 2
HEAD_DIM = 32
DTYPE = torch.bfloat16
UNWRITTEN = -256


def _dist_init(rank, port, kind):
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        WORLD_SIZE=str(WORLD),
    )
    os.environ.setdefault("no_proxy", "127.0.0.1,localhost")
    torch.cuda.set_device(rank)

    from sglang.srt.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from sglang.srt.runtime_context import publish
    from sglang.srt.server_args import ServerArgs
    from sglang.test.test_utils import publish_build_topology

    init_distributed_environment(
        world_size=WORLD,
        rank=rank,
        local_rank=rank,
        distributed_init_method=f"tcp://127.0.0.1:{port}",
        backend="nccl",
    )
    cp_size = WORLD if kind == "mha" else 1
    publish(
        ServerArgs(model_path="dummy", tp_size=WORLD, attn_cp_size=cp_size),
        role="scheduler",
    )
    publish_build_topology(tp_size=WORLD, attn_cp_size=cp_size, world_rank=rank)
    initialize_model_parallel()


def _make_pool(rank, group, kind, start_layer, layer_num):
    from sglang.srt.mem_cache.page_interleave import PageShardSpec
    from sglang.srt.mem_cache.page_interleave_pool import (
        PageInterleaveMHATokenToKVPool,
        PageInterleaveMLATokenToKVPool,
    )

    kwargs = dict(
        size=SIZE,
        page_size=PAGE_SIZE,
        dtype=DTYPE,
        layer_num=layer_num,
        device=f"cuda:{rank}",
        enable_memory_saver=False,
        start_layer=start_layer,
        end_layer=start_layer + layer_num - 1,
        shard_spec=PageShardSpec(
            shard_rank=group.rank_in_group,
            shard_size=WORLD,
            page_size=PAGE_SIZE,
            max_prefix_tokens=16 * WORLD * PAGE_SIZE,
            chunk_tokens=4 * PAGE_SIZE,
        ),
        shard_group=group,
    )
    if kind == "mla":
        pool = PageInterleaveMLATokenToKVPool(
            **kwargs, kv_lora_rank=KV_LORA_RANK, qk_rope_head_dim=QK_ROPE
        )
    else:
        pool = PageInterleaveMHATokenToKVPool(
            **kwargs,
            head_num=HEAD_NUM,
            head_dim=HEAD_DIM,
            enable_alt_stream=False,
        )
    for local_layer in range(layer_num):
        for buffer in _buffers(pool, kind, local_layer):
            buffer.fill_(UNWRITTEN)
    return pool


def _buffers(pool, kind, local_layer):
    if kind == "mla":
        return (pool.kv_buffer[local_layer],)
    return (pool.k_buffer[local_layer], pool.v_buffer[local_layer])


def _make_draft_pool(target, allocator, kind, start_layer):
    from sglang.srt.mem_cache.kv_cache_configurator import (
        KVCacheConfigurator,
        _PoolSizes,
    )
    from sglang.srt.runtime_context import get_context
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    # Skip model loading while using the production pool dispatch and sharing
    # the target allocator. A replicated draft pool must fail this test.
    kvc = KVCacheConfigurator.__new__(KVCacheConfigurator)
    kvc.device = target.device
    kvc.is_draft_worker = True
    kvc.page_size = PAGE_SIZE
    kvc.use_mla_backend = kind == "mla"
    kvc.kv_cache_dtype = DTYPE
    kvc.kv_cache_dtype_str = "bfloat16"
    kvc.post_capture_kv_active = False
    kvc.is_hybrid_swa = False
    kvc.sliding_window_size = None
    kvc.mambaish_config = None
    kvc.token_to_kv_pool_allocator = allocator
    kvc.spec_algorithm = SpeculativeAlgorithm.EAGLE
    kvc.model_config = SimpleNamespace(
        num_nextn_predict_layers=1,
        hf_config=SimpleNamespace(architectures=["DeepseekV3ForCausalLM"]),
        kv_lora_rank=KV_LORA_RANK,
        qk_rope_head_dim=QK_ROPE,
        head_dim=HEAD_DIM,
        v_head_dim=HEAD_DIM,
        get_num_kv_heads=lambda *_args: HEAD_NUM,
    )
    kvc.layer_info = SimpleNamespace(
        start_layer=start_layer, end_layer=start_layer + 1, num_effective_layers=1
    )
    sizes = _PoolSizes(
        max_total_num_tokens=target.size,
        max_running_requests=1,
        full_max_total_num_tokens=None,
        swa_max_total_num_tokens=None,
        c4_max_total_num_tokens=0,
        c128_max_total_num_tokens=0,
        c4_state_pool_size=0,
        c128_state_pool_size=0,
        c4_state_dtype=None,
        c128_state_dtype=None,
    )
    with get_context().override_server_args(
        page_size=PAGE_SIZE,
        tp_size=WORLD,
        attn_cp_size=WORLD if kind == "mha" else 1,
        speculative_algorithm="EAGLE",
        enable_multi_layer_eagle=False,
    ):
        pool = kvc._build_token_to_kv_pool(
            sizes=sizes,
            is_dsa_model=False,
            is_dsv4_model=False,
            req_to_token_pool=None,
        )
    assert pool.shard_spec is target.shard_spec
    assert pool.shard_group is target.shard_group
    for buffer in _buffers(pool, kind, 0):
        buffer.fill_(UNWRITTEN)
    return pool


def _values(locs, width, tag, local_layer, is_v=False):
    """Each location, feature, layer, and pool has distinguishable BF16 bytes.

    Small integers survive BF16 exactly; high-valued logical slot IDs plus tiny
    feature offsets would round away the differences this test must detect.
    The first two columns encode the full logical slot, avoiding hash aliases.
    """
    features = torch.arange(width, device=locs.device)
    values = (
        locs[:, None] * 37
        + features[None, :] * 17
        + tag * 73
        + local_layer * 11
        + int(is_v) * 97
    ) % 251 - 125
    values[:, 0] = locs % 128
    values[:, 1] = locs // 128
    values[:, 2] = tag
    values[:, 3] = local_layer
    values[:, 4] = int(is_v)
    return values.to(DTYPE)


def _canonical(kind, locs, tag, local_layer):
    if kind == "mla":
        return (_values(locs, KV_LORA_RANK + QK_ROPE, tag, local_layer).unsqueeze(1),)
    return tuple(
        _values(locs, HEAD_NUM * HEAD_DIM, tag, local_layer, is_v).reshape(
            -1, HEAD_NUM, HEAD_DIM
        )
        for is_v in (False, True)
    )


def _assert_bytes(got, expected, label):
    assert got.shape == expected.shape, label
    assert got.dtype == expected.dtype == DTYPE, label
    assert torch.equal(
        got.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
    ), f"{label}: KV bytes differ"


def _logical_chain(device):
    # Owners alternate from rank 1, while local pages run out of order. Some
    # logical slots exceed SIZE, catching a replicated draft indexing them raw.
    local_pages = torch.tensor([25, 3, 20, 6, 28, 4, 30, 8, 22], device=device)
    owners = (1 + torch.arange(local_pages.numel(), device=device)) % WORLD
    pages = local_pages * WORLD + owners
    return (
        pages[:, None] * PAGE_SIZE + torch.arange(PAGE_SIZE, device=device)
    ).reshape(-1)


def _write(pool, kind, locs, tag, local_layer, epoch):
    layer = SimpleNamespace(layer_id=pool.start_layer + local_layer)
    values = _canonical(kind, locs, tag, local_layer)
    if kind == "mha":
        pool.set_kv_buffer(layer, locs, *values)
    elif epoch % 2:
        # Both MLA write entry points must owner-filter and stage MTP rows.
        pool.set_mla_kv_buffer(
            layer, locs, values[0][..., :KV_LORA_RANK], values[0][..., KV_LORA_RANK:]
        )
    else:
        pool.set_kv_buffer(layer, locs, values[0], values[0][..., :KV_LORA_RANK])


def _check_persisted(pool, kind, locs, tag, rank):
    owned_locs = locs[(locs // PAGE_SIZE) % WORLD == rank]
    local_rows = owned_locs // (WORLD * PAGE_SIZE) * PAGE_SIZE + owned_locs % PAGE_SIZE
    for local_layer in range(pool.layer_num):
        canonical = _canonical(kind, owned_locs, tag, local_layer)
        for buffer, values in zip(_buffers(pool, kind, local_layer), canonical):
            expected = torch.full_like(buffer, UNWRITTEN)
            expected[local_rows] = values
            _assert_bytes(
                buffer, expected, f"{kind} pool {tag} owned layer {local_layer}"
            )


def _check_scratch(pool, kind, locs, prefix_len, tag, local_layer):
    layer_id = pool.start_layer + local_layer
    rows = pool.translate_loc_to_scratch(locs)
    expected = _canonical(kind, locs, tag, local_layer)
    _assert_bytes(
        pool.get_key_buffer(layer_id)[rows], expected[0], f"{kind} pool {tag} key"
    )
    _assert_bytes(
        pool.get_value_buffer(layer_id)[rows],
        expected[0][..., :KV_LORA_RANK] if kind == "mla" else expected[1],
        f"{kind} pool {tag} value",
    )
    if kind == "mla" and prefix_len:
        # Arbitrary prefix indices also exercise the chunked MLA consumer.
        subset = locs[3:prefix_len:7].clone()
        k_nope, k_rope = pool.get_mla_kv_buffer(
            SimpleNamespace(layer_id=layer_id), subset, DTYPE
        )
        reference = _canonical(kind, subset, tag, local_layer)[0]
        _assert_bytes(k_nope, reference[..., :KV_LORA_RANK], f"pool {tag} latent")
        _assert_bytes(k_rope, reference[..., KV_LORA_RANK:], f"pool {tag} rope")


def _run(rank, port, kind):
    _dist_init(rank, port, kind)
    from sglang.srt.mem_cache.allocator.page_interleave import (
        PageInterleavePoolAllocator,
    )
    from sglang.srt.mem_cache.page_interleave import get_kv_shard_group

    group = get_kv_shard_group(use_mla_backend=kind == "mla")
    assert group.world_size == WORLD
    target = _make_pool(rank, group, kind, start_layer=0, layer_num=TARGET_LAYERS)
    allocator = PageInterleavePoolAllocator(
        size=target.size,
        physical_page_size=PAGE_SIZE,
        shard_size=WORLD,
        dtype=DTYPE,
        device=target.device,
        kvcache=target,
        need_sort=False,
        shard_spec=target.shard_spec,
    )
    # DeepSeek's NextN uses local layer 0. Also cover an absolute/offset layer ID
    # so the one-layer gather terminates correctly on either scratch-slot parity.
    drafts = [
        _make_draft_pool(target, allocator, kind, start_layer=start)
        for start in (0, 61)
    ]
    pools = [target, *drafts]
    for other in drafts:
        assert other._page_pos.data_ptr() != target._page_pos.data_ptr()
        assert other.kv_gather_comm is not target.kv_gather_comm
        for slot, target_slot in zip(other._slots, target._slots):
            for name in slot.tensors:
                assert (
                    slot.tensors[name].data_ptr()
                    != target_slot.tensors[name].data_ptr()
                )

    locs = _logical_chain(target.device)
    req_to_token = locs.to(torch.int32).unsqueeze(0)
    req_indices = torch.tensor([0], dtype=torch.int64, device=locs.device)
    # Each cached prefix is page aligned. The third batch writes only five
    # tokens of its tail page; the fourth recomputes and completes that page.
    batches = [(0, 48), (48, 96), (96, 133), (128, 144)]
    for epoch, (prefix_len, seq_len) in enumerate(batches, start=1):
        active_locs = locs[:seq_len]
        chunk_locs = locs[prefix_len:seq_len]
        target_epoch = target._epoch
        for tag, pool in enumerate(pools, start=1):
            pool.begin_shard_extend(req_to_token, req_indices, [prefix_len], [seq_len])
            assert pool._epoch == epoch
            for local_layer in range(pool.layer_num):
                _write(pool, kind, chunk_locs, tag, local_layer, epoch)
                _check_scratch(pool, kind, active_locs, prefix_len, tag, local_layer)
            _check_persisted(pool, kind, active_locs, tag, rank)
            # MTP metadata/writes/reads must leave the target's last resident
            # layer and gather plan usable, despite identical logical locations.
            assert target._epoch == target_epoch + 1
            _check_scratch(target, kind, active_locs, prefix_len, 1, TARGET_LAYERS - 1)
            if pool.layer_num == 1 and prefix_len:
                resident = pool._slots[pool.start_layer % 2]
                assert resident.resident_key == (pool.start_layer, epoch)
                assert pool._slots[(pool.start_layer + 1) % 2].resident_key is None
        if rank == 0:
            print(f"PASS: {kind.upper()} target/MTP prefix={prefix_len}, seq={seq_len}")

    # Deactivating one draft's plan must not deactivate the other pools.
    drafts[0].end_shard_extend()
    assert not drafts[0]._shard_extend_active
    assert target._shard_extend_active and drafts[1]._shard_extend_active
    _check_scratch(drafts[1], kind, locs, 128, 3, 0)
    _check_scratch(target, kind, locs, 128, 1, TARGET_LAYERS - 1)
    torch.cuda.synchronize()
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", choices=("all", "mla", "mha"), default="all")
    parser.add_argument("--port", type=int, default=29821)
    args = parser.parse_args()
    if torch.cuda.device_count() < WORLD:
        raise RuntimeError("MTP KV sharding correctness requires two visible GPUs")
    kinds = ("mla", "mha") if args.pool == "all" else (args.pool,)
    for offset, kind in enumerate(kinds):
        mp.spawn(_run, args=(args.port + offset, kind), nprocs=WORLD, join=True)
    print("PASS: MTP page-interleave KV cache correctness")


if __name__ == "__main__":
    main()
