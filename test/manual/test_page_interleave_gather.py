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
"""Multi-GPU test for logical-page KV sharding pools (2 GPUs, real NCCL).

Two phases, run in separate process groups (one mp.spawn each):

1. MLA pool sharded across the attention-TP group: replicated writes are
   owner-filtered into disjoint pool stripes; a later batch's chunked-prefix
   read (get_mla_kv_buffer) assembles the full prefix from all ranks via the
   layer-ahead NCCL allgather and must return the canonical bytes.
2. MHA pool sharded across the attention-CP group: the post-allgather full
   chunk is staged into the scratch chunk region and owner-persisted; a later
   batch reads prefix+chunk through the translated page table (the scratch),
   and the assembled rows must match the canonical bytes.

Run: CUDA_VISIBLE_DEVICES=0,1 python test/manual/test_page_interleave_gather.py

Modeled on test/manual/test_dsa_layer_split_broadcast.py.
"""

import os
from types import SimpleNamespace

import torch
import torch.multiprocessing as mp

WORLD = 2
LAYER_NUM = 4
PAGE_SIZE = 16
GRANULE = WORLD * PAGE_SIZE
SIZE = PAGE_SIZE * 64  # physical token slots per rank
KV_LORA_RANK = 128
QK_ROPE = 32
HEAD_NUM = 2
HEAD_DIM = 32
DTYPE = torch.bfloat16


def _mla_value(loc, dim):
    """Deterministic canonical latent value for logical slot ``loc``."""
    loc = loc.to(torch.float32)
    return (loc.unsqueeze(-1) + torch.arange(dim, device=loc.device) * 0.001).to(DTYPE)


def _dist_init(rank, world, port, attn_cp_size):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    os.environ.setdefault("no_proxy", "127.0.0.1,localhost")
    torch.cuda.set_device(rank)

    from sglang.srt.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    init_distributed_environment(
        world_size=world,
        rank=rank,
        local_rank=rank,
        distributed_init_method=f"tcp://127.0.0.1:{port}",
        backend="nccl",
    )
    initialize_model_parallel(
        tensor_model_parallel_size=world,
        attention_context_model_parallel_size=attn_cp_size,
    )


def _make_spec(shard_rank, max_prefix_groups=16, chunk_groups=4):
    from sglang.srt.mem_cache.page_interleave import PageShardSpec

    return PageShardSpec(
        shard_rank=shard_rank,
        shard_size=WORLD,
        page_size=PAGE_SIZE,
        max_prefix_tokens=max_prefix_groups * GRANULE,
        chunk_tokens=chunk_groups * GRANULE,
    )


def _fake_req_to_token(groups, seq_len, device):
    """req_to_token row where sequence group j is allocator group groups[j]."""
    row = torch.zeros((1, len(groups) * GRANULE), dtype=torch.int32, device=device)
    for j, q in enumerate(groups):
        row[0, j * GRANULE : (j + 1) * GRANULE] = torch.arange(
            q * GRANULE, (q + 1) * GRANULE, dtype=torch.int32, device=device
        )
    return row[:, :seq_len] if seq_len < row.shape[1] else row


def _check(rank, name, got, expect, atol=0.0):
    ok = torch.allclose(got.float(), expect.float(), atol=atol, rtol=0)
    max_err = (got.float() - expect.float()).abs().max().item()
    print(f"[rank {rank}] {name}: max_err={max_err:.6f} {'OK' if ok else 'FAIL'}")
    assert ok, f"[rank {rank}] {name} mismatch (max_err={max_err})"


def _run_mla(rank, world, port):
    _dist_init(rank, world, port, attn_cp_size=1)

    from sglang.srt.layers.dp_attention import get_attention_tp_group
    from sglang.srt.mem_cache.page_interleave import get_kv_shard_group
    from sglang.srt.mem_cache.page_interleave_pool import (
        PageInterleaveMLATokenToKVPool,
    )

    group = get_attention_tp_group()
    assert group.world_size == world
    # Topology-first shard-group selection: no CP here, so MLA falls back to
    # the attn-TP axis, while GQA has no replicated axis (world_size 1).
    assert get_kv_shard_group(use_mla_backend=True) is group
    assert get_kv_shard_group(use_mla_backend=False).world_size == 1
    spec = _make_spec(shard_rank=group.rank_in_group)

    pool = PageInterleaveMLATokenToKVPool(
        SIZE,
        page_size=PAGE_SIZE,
        dtype=DTYPE,
        kv_lora_rank=KV_LORA_RANK,
        qk_rope_head_dim=QK_ROPE,
        layer_num=LAYER_NUM,
        device=f"cuda:{rank}",
        enable_memory_saver=False,
        start_layer=0,
        end_layer=LAYER_NUM - 1,
        shard_spec=spec,
        shard_group=group,
    )
    device = pool.kv_buffer[0].device

    # ---- chunk 1: replicated write, owner-filtered persist -----------------
    # "Allocator" hands out fragmented groups (identical on every rank).
    chunk1_groups = [5, 2, 9]
    chunk1_locs = _fake_req_to_token(chunk1_groups, 3 * GRANULE, device)[0].long()
    for layer_id in range(LAYER_NUM):
        layer = SimpleNamespace(layer_id=layer_id)
        vals = _mla_value(chunk1_locs + layer_id * 1000, KV_LORA_RANK + QK_ROPE)
        pool.set_mla_kv_buffer(
            layer,
            chunk1_locs,
            vals[:, :KV_LORA_RANK].unsqueeze(1),
            vals[:, KV_LORA_RANK:].unsqueeze(1),
        )
    torch.cuda.synchronize()
    torch.distributed.barrier()

    # Pool holds only the owned stripe: group Q sits at local rows [Q*ps,(Q+1)*ps)
    # on every rank, holding that rank's page of the group.
    for q in chunk1_groups:
        local_rows = torch.arange(q * PAGE_SIZE, (q + 1) * PAGE_SIZE, device=device)
        owned_locs = (
            q * GRANULE
            + group.rank_in_group * PAGE_SIZE
            + torch.arange(PAGE_SIZE, device=device)
        )
        got = pool.kv_buffer[0][local_rows, 0, :].view(DTYPE)
        expect = _mla_value(owned_locs, KV_LORA_RANK + QK_ROPE)
        _check(rank, f"mla owned stripe g{q}", got, expect)

    # ---- chunk 2: prefix gather + staged chunk, both read styles -----------
    seq_groups = chunk1_groups + [12]  # one new chunk group
    prefix_len = 3 * GRANULE
    seq_len = prefix_len + GRANULE
    req_to_token = _fake_req_to_token(seq_groups, seq_len, device)
    chunk2_locs = req_to_token[0, prefix_len:seq_len].long()
    pool.begin_shard_extend(req_to_token, torch.tensor([0]), [prefix_len], [seq_len])

    for layer_id in range(LAYER_NUM):
        layer = SimpleNamespace(layer_id=layer_id)
        # Write the current chunk (stages it into the slot + persists the
        # owned stripe), like the extend forward does before attention.
        chunk_vals = _mla_value(chunk2_locs + layer_id * 1000, KV_LORA_RANK + QK_ROPE)
        pool.set_mla_kv_buffer(
            layer,
            chunk2_locs,
            chunk_vals[:, :KV_LORA_RANK].unsqueeze(1),
            chunk_vals[:, KV_LORA_RANK:].unsqueeze(1),
        )
        # Chunked-prefix MHA style: fetch an arbitrary sub-range of the
        # prefix through get_mla_kv_buffer.
        sub = chunk1_locs[PAGE_SIZE // 2 : prefix_len - 3]
        k_nope, k_rope = pool.get_mla_kv_buffer(layer, sub, DTYPE)
        expect = _mla_value(sub + layer_id * 1000, KV_LORA_RANK + QK_ROPE)
        _check(
            rank,
            f"mla prefix read l{layer_id}",
            k_nope[:, 0, :],
            expect[:, :KV_LORA_RANK],
        )
        _check(
            rank,
            f"mla prefix rope l{layer_id}",
            k_rope[:, 0, :],
            expect[:, KV_LORA_RANK:],
        )
        # Absorbed-MLA style (what MLA-under-CP uses): read [prefix | chunk]
        # from get_key_buffer through the translated page table.
        all_locs = req_to_token[0, :seq_len].long()
        rows = pool.translate_loc_to_scratch(all_locs)
        kv_scratch = pool.get_key_buffer(layer_id)
        _check(
            rank,
            f"mla absorbed read l{layer_id}",
            kv_scratch[rows, 0, :],
            _mla_value(all_locs + layer_id * 1000, KV_LORA_RANK + QK_ROPE),
        )

    torch.distributed.barrier()
    if rank == 0:
        print("PASS: MLA page-interleave shard (attn-TP axis)")


def _run_mha(rank, world, port):
    _dist_init(rank, world, port, attn_cp_size=world)

    from sglang.srt.layers.dp_attention import get_attention_cp_group
    from sglang.srt.mem_cache.page_interleave import get_kv_shard_group
    from sglang.srt.mem_cache.page_interleave_pool import (
        PageInterleaveMHATokenToKVPool,
    )

    group = get_attention_cp_group()
    assert group.world_size == world
    # Topology-first shard-group selection: with an active CP group, both
    # GQA and MLA shard across CP (CP replicates KV for every attention
    # type; the TP axis is only the no-CP MLA fallback).
    assert get_kv_shard_group(use_mla_backend=False) is group
    assert get_kv_shard_group(use_mla_backend=True) is group
    spec = _make_spec(shard_rank=group.rank_in_group)

    pool = PageInterleaveMHATokenToKVPool(
        SIZE,
        page_size=PAGE_SIZE,
        dtype=DTYPE,
        head_num=HEAD_NUM,
        head_dim=HEAD_DIM,
        layer_num=LAYER_NUM,
        device=f"cuda:{rank}",
        enable_memory_saver=False,
        start_layer=0,
        end_layer=LAYER_NUM - 1,
        enable_alt_stream=False,
        shard_spec=spec,
        shard_group=group,
    )
    device = pool.k_buffer[0].device

    def kv_value(locs, layer_id, is_v):
        base = locs.to(torch.float32) + layer_id * 1000 + (500000 if is_v else 0)
        return (
            base.view(-1, 1, 1)
            + torch.arange(HEAD_NUM, device=device).view(1, -1, 1) * 0.01
            + torch.arange(HEAD_DIM, device=device).view(1, 1, -1) * 0.0001
        ).to(DTYPE)

    # ---- chunk 1 (prefix-less batch): stage + owner-persist ----------------
    chunk1_groups = [7, 3]
    chunk1_locs = _fake_req_to_token(chunk1_groups, 2 * GRANULE, device)[0].long()
    req_to_token = _fake_req_to_token(chunk1_groups, 2 * GRANULE, device)
    pool.begin_shard_extend(req_to_token, torch.tensor([0]), [0], [2 * GRANULE])
    for layer_id in range(LAYER_NUM):
        layer = SimpleNamespace(layer_id=layer_id)
        pool.set_kv_buffer(
            layer,
            chunk1_locs,
            kv_value(chunk1_locs, layer_id, False),
            kv_value(chunk1_locs, layer_id, True),
        )
        # The current chunk must be readable through the scratch right away.
        k_scratch = pool.get_key_buffer(layer_id)
        rows = pool.translate_loc_to_scratch(chunk1_locs)
        _check(
            rank,
            f"mha chunk stage l{layer_id}",
            k_scratch[rows],
            kv_value(chunk1_locs, layer_id, False),
        )
    torch.cuda.synchronize()
    torch.distributed.barrier()

    # ---- chunk 2: prefix gathered from peers via translated page table -----
    seq_groups = chunk1_groups + [11]
    prefix_len = 2 * GRANULE
    seq_len = prefix_len + GRANULE
    req_to_token = _fake_req_to_token(seq_groups, seq_len, device)
    chunk2_locs = req_to_token[0, prefix_len:seq_len].long()
    pool.begin_shard_extend(req_to_token, torch.tensor([0]), [prefix_len], [seq_len])

    for layer_id in range(LAYER_NUM):
        layer = SimpleNamespace(layer_id=layer_id)
        pool.set_kv_buffer(
            layer,
            chunk2_locs,
            kv_value(chunk2_locs, layer_id, False),
            kv_value(chunk2_locs, layer_id, True),
        )
        all_locs = req_to_token[0, :seq_len].long()
        rows = pool.translate_loc_to_scratch(all_locs)
        k_scratch = pool.get_key_buffer(layer_id)
        v_scratch = pool.get_value_buffer(layer_id)
        _check(
            rank,
            f"mha seq read k l{layer_id}",
            k_scratch[rows],
            kv_value(all_locs, layer_id, False),
        )
        _check(
            rank,
            f"mha seq read v l{layer_id}",
            v_scratch[rows],
            kv_value(all_locs, layer_id, True),
        )

    torch.distributed.barrier()
    if rank == 0:
        print("PASS: MHA page-interleave shard (attn-CP axis)")


def main():
    mp.spawn(_run_mla, args=(WORLD, 29811), nprocs=WORLD, join=True)
    mp.spawn(_run_mha, args=(WORLD, 29812), nprocs=WORLD, join=True)
    print("PASS: all page-interleave gather tests")


if __name__ == "__main__":
    main()
