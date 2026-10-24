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
"""Multi-GPU tests for logical-page KV cache sharding: the real NCCL
layer-ahead gather over a real 2-rank process group.

``test_page_interleave_shard.py`` pins the plan arithmetic on CPU stubs. These
tests drive real pools, which is the only check that the plan the stubs
validate actually addresses the bytes the collective delivers:

1. MLA pool sharded across the attention-TP group: replicated writes are
   owner-filtered into disjoint pool stripes; a later batch's chunked-prefix
   read (get_mla_kv_buffer) assembles the full prefix from all ranks via the
   layer-ahead NCCL allgather and must return the canonical bytes.
2. MHA pool sharded across the attention-CP group: the post-allgather full
   chunk is staged into the scratch chunk region and owner-persisted; a later
   batch reads prefix+chunk through the translated page table (the scratch),
   and the assembled rows must match the canonical bytes.
3. MTP draft pools sharing the target's sharded allocator, for MLA and MHA:
   exact KV bytes, separate target/draft storage and scratch plans, fragmented
   pages, partial tail pages, both MLA write APIs, and local/nonzero draft
   layer IDs across batches.

Registered as a base-b 2-gpu-large unit test; skips when fewer than 2 CUDA
devices are visible. Run directly with:
    CUDA_VISIBLE_DEVICES=0,1 python3 \
        test/registered/unit/mem_cache/test_page_interleave_gather_multi_gpu.py
"""

import os
import unittest
from types import SimpleNamespace

import torch
import torch.multiprocessing as mp

from sglang.srt.distributed import (
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.mem_cache.allocator.page_interleave import PageInterleavePoolAllocator
from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
from sglang.srt.mem_cache.page_interleave import PageShardSpec, get_kv_shard_group
from sglang.srt.mem_cache.page_interleave_pool import (
    PageInterleaveMHATokenToKVPool,
    PageInterleaveMLATokenToKVPool,
)
from sglang.srt.runtime_context import get_context, get_parallel, publish
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase, publish_build_topology

register_cuda_ci(est_time=120, stage="base-b", runner_config="2-gpu-large")

_GATHER_WORLD = 2
_GATHER_LAYER_NUM = 4
_GATHER_PAGE_SIZE = 16
_GATHER_GRANULE = _GATHER_WORLD * _GATHER_PAGE_SIZE
_GATHER_SIZE = _GATHER_PAGE_SIZE * 64  # physical token slots per rank
_GATHER_KV_LORA_RANK = 128
_GATHER_QK_ROPE = 32
_GATHER_HEAD_NUM = 2
_GATHER_HEAD_DIM = 32
_GATHER_DTYPE = torch.bfloat16
_MTP_SIZE = 32 * _GATHER_PAGE_SIZE  # Physical token capacity per rank.
_MTP_UNWRITTEN = -256


def _mla_value(loc, dim):
    """Deterministic canonical latent value for logical slot ``loc``."""
    loc = loc.to(torch.float32)
    return (loc.unsqueeze(-1) + torch.arange(dim, device=loc.device) * 0.001).to(
        _GATHER_DTYPE
    )


def _dist_init(rank, world, port, attn_cp_size):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    os.environ.setdefault("no_proxy", "127.0.0.1,localhost")
    torch.cuda.set_device(rank)

    init_distributed_environment(
        world_size=world,
        rank=rank,
        local_rank=rank,
        distributed_init_method=f"tcp://127.0.0.1:{port}",
        backend="nccl",
    )
    # Publish the widths the groups below are about to be built at. The derived
    # quotients (attn_tp_size, attn_dcp_size, ...) are projected from these
    # leaves at publish; initialize_model_parallel no longer supplies them, and
    # MLATokenToKVPool.set_mla_kv_buffer reads attn_dcp_size on the write path.
    publish(
        ServerArgs(model_path="dummy", tp_size=world, attn_cp_size=attn_cp_size),
        role="scheduler",
    )
    publish_build_topology(tp_size=world, attn_cp_size=attn_cp_size, world_rank=rank)
    initialize_model_parallel()


def _gather_make_spec(shard_rank, max_prefix_groups=16, chunk_groups=4):
    return PageShardSpec(
        shard_rank=shard_rank,
        shard_size=_GATHER_WORLD,
        page_size=_GATHER_PAGE_SIZE,
        max_prefix_tokens=max_prefix_groups * _GATHER_GRANULE,
        chunk_tokens=chunk_groups * _GATHER_GRANULE,
    )


def _fake_req_to_token(groups, seq_len, device):
    """req_to_token row where sequence group j is allocator group groups[j]."""
    row = torch.zeros(
        (1, len(groups) * _GATHER_GRANULE), dtype=torch.int32, device=device
    )
    for j, q in enumerate(groups):
        row[0, j * _GATHER_GRANULE : (j + 1) * _GATHER_GRANULE] = torch.arange(
            q * _GATHER_GRANULE,
            (q + 1) * _GATHER_GRANULE,
            dtype=torch.int32,
            device=device,
        )
    return row[:, :seq_len] if seq_len < row.shape[1] else row


def _check(rank, name, got, expect, atol=0.0):
    ok = torch.allclose(got.float(), expect.float(), atol=atol, rtol=0)
    max_err = (got.float() - expect.float()).abs().max().item()
    print(f"[rank {rank}] {name}: max_err={max_err:.6f} {'OK' if ok else 'FAIL'}")
    assert ok, f"[rank {rank}] {name} mismatch (max_err={max_err})"


def _run_mla(rank, world, port):
    _dist_init(rank, world, port, attn_cp_size=1)

    group = get_parallel().attn_tp_group
    assert group.world_size == world
    # Topology-first shard-group selection: no CP here, so MLA falls back to
    # the attn-TP axis, while GQA has no replicated axis (world_size 1).
    assert get_kv_shard_group(use_mla_backend=True) is group
    assert get_kv_shard_group(use_mla_backend=False).world_size == 1
    spec = _gather_make_spec(shard_rank=group.rank_in_group)

    pool = PageInterleaveMLATokenToKVPool(
        _GATHER_SIZE,
        page_size=_GATHER_PAGE_SIZE,
        dtype=_GATHER_DTYPE,
        kv_lora_rank=_GATHER_KV_LORA_RANK,
        qk_rope_head_dim=_GATHER_QK_ROPE,
        layer_num=_GATHER_LAYER_NUM,
        device=f"cuda:{rank}",
        enable_memory_saver=False,
        start_layer=0,
        end_layer=_GATHER_LAYER_NUM - 1,
        shard_spec=spec,
        shard_group=group,
    )
    device = pool.kv_buffer[0].device

    # ---- chunk 1: replicated write, owner-filtered persist -----------------
    # "Allocator" hands out fragmented groups (identical on every rank).
    chunk1_groups = [5, 2, 9]
    chunk1_locs = _fake_req_to_token(chunk1_groups, 3 * _GATHER_GRANULE, device)[
        0
    ].long()
    for layer_id in range(_GATHER_LAYER_NUM):
        layer = SimpleNamespace(layer_id=layer_id)
        vals = _mla_value(
            chunk1_locs + layer_id * 1000, _GATHER_KV_LORA_RANK + _GATHER_QK_ROPE
        )
        pool.set_mla_kv_buffer(
            layer,
            chunk1_locs,
            vals[:, :_GATHER_KV_LORA_RANK].unsqueeze(1),
            vals[:, _GATHER_KV_LORA_RANK:].unsqueeze(1),
        )
    torch.cuda.synchronize()
    torch.distributed.barrier()

    # Pool holds only the owned stripe: group Q sits at local rows [Q*ps,(Q+1)*ps)
    # on every rank, holding that rank's page of the group.
    for q in chunk1_groups:
        local_rows = torch.arange(
            q * _GATHER_PAGE_SIZE, (q + 1) * _GATHER_PAGE_SIZE, device=device
        )
        owned_locs = (
            q * _GATHER_GRANULE
            + group.rank_in_group * _GATHER_PAGE_SIZE
            + torch.arange(_GATHER_PAGE_SIZE, device=device)
        )
        got = pool.kv_buffer[0][local_rows, 0, :].view(_GATHER_DTYPE)
        expect = _mla_value(owned_locs, _GATHER_KV_LORA_RANK + _GATHER_QK_ROPE)
        _check(rank, f"mla owned stripe g{q}", got, expect)

    # ---- chunk 2: prefix gather + staged chunk, both read styles -----------
    seq_groups = chunk1_groups + [12]  # one new chunk group
    prefix_len = 3 * _GATHER_GRANULE
    seq_len = prefix_len + _GATHER_GRANULE
    req_to_token = _fake_req_to_token(seq_groups, seq_len, device)
    chunk2_locs = req_to_token[0, prefix_len:seq_len].long()
    pool.begin_shard_extend(req_to_token, torch.tensor([0]), [prefix_len], [seq_len])

    for layer_id in range(_GATHER_LAYER_NUM):
        layer = SimpleNamespace(layer_id=layer_id)
        # Write the current chunk (stages it into the slot + persists the
        # owned stripe), like the extend forward does before attention.
        chunk_vals = _mla_value(
            chunk2_locs + layer_id * 1000, _GATHER_KV_LORA_RANK + _GATHER_QK_ROPE
        )
        pool.set_mla_kv_buffer(
            layer,
            chunk2_locs,
            chunk_vals[:, :_GATHER_KV_LORA_RANK].unsqueeze(1),
            chunk_vals[:, _GATHER_KV_LORA_RANK:].unsqueeze(1),
        )
        # Chunked-prefix MHA style: fetch an arbitrary sub-range of the
        # prefix through get_mla_kv_buffer.
        sub = chunk1_locs[_GATHER_PAGE_SIZE // 2 : prefix_len - 3]
        k_nope, k_rope = pool.get_mla_kv_buffer(layer, sub, _GATHER_DTYPE)
        expect = _mla_value(
            sub + layer_id * 1000, _GATHER_KV_LORA_RANK + _GATHER_QK_ROPE
        )
        _check(
            rank,
            f"mla prefix read l{layer_id}",
            k_nope[:, 0, :],
            expect[:, :_GATHER_KV_LORA_RANK],
        )
        _check(
            rank,
            f"mla prefix rope l{layer_id}",
            k_rope[:, 0, :],
            expect[:, _GATHER_KV_LORA_RANK:],
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
            _mla_value(
                all_locs + layer_id * 1000, _GATHER_KV_LORA_RANK + _GATHER_QK_ROPE
            ),
        )

    torch.distributed.barrier()
    if rank == 0:
        print("PASS: MLA page-interleave shard (attn-TP axis)")


def _run_mha(rank, world, port):
    _dist_init(rank, world, port, attn_cp_size=world)

    group = get_parallel().attn_cp_group
    assert group.world_size == world
    # Topology-first shard-group selection: with an active CP group, both
    # GQA and MLA shard across CP (CP replicates KV for every attention
    # type; the TP axis is only the no-CP MLA fallback).
    assert get_kv_shard_group(use_mla_backend=False) is group
    assert get_kv_shard_group(use_mla_backend=True) is group
    spec = _gather_make_spec(shard_rank=group.rank_in_group)

    pool = PageInterleaveMHATokenToKVPool(
        _GATHER_SIZE,
        page_size=_GATHER_PAGE_SIZE,
        dtype=_GATHER_DTYPE,
        head_num=_GATHER_HEAD_NUM,
        head_dim=_GATHER_HEAD_DIM,
        layer_num=_GATHER_LAYER_NUM,
        device=f"cuda:{rank}",
        enable_memory_saver=False,
        start_layer=0,
        end_layer=_GATHER_LAYER_NUM - 1,
        enable_alt_stream=False,
        shard_spec=spec,
        shard_group=group,
    )
    device = pool.k_buffer[0].device

    def kv_value(locs, layer_id, is_v):
        base = locs.to(torch.float32) + layer_id * 1000 + (500000 if is_v else 0)
        return (
            base.view(-1, 1, 1)
            + torch.arange(_GATHER_HEAD_NUM, device=device).view(1, -1, 1) * 0.01
            + torch.arange(_GATHER_HEAD_DIM, device=device).view(1, 1, -1) * 0.0001
        ).to(_GATHER_DTYPE)

    # ---- chunk 1 (prefix-less batch): stage + owner-persist ----------------
    chunk1_groups = [7, 3]
    chunk1_locs = _fake_req_to_token(chunk1_groups, 2 * _GATHER_GRANULE, device)[
        0
    ].long()
    req_to_token = _fake_req_to_token(chunk1_groups, 2 * _GATHER_GRANULE, device)
    pool.begin_shard_extend(req_to_token, torch.tensor([0]), [0], [2 * _GATHER_GRANULE])
    for layer_id in range(_GATHER_LAYER_NUM):
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
    prefix_len = 2 * _GATHER_GRANULE
    seq_len = prefix_len + _GATHER_GRANULE
    req_to_token = _fake_req_to_token(seq_groups, seq_len, device)
    chunk2_locs = req_to_token[0, prefix_len:seq_len].long()
    pool.begin_shard_extend(req_to_token, torch.tensor([0]), [prefix_len], [seq_len])

    for layer_id in range(_GATHER_LAYER_NUM):
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


def _mtp_make_pool(rank, group, kind, start_layer, layer_num):
    kwargs = dict(
        size=_MTP_SIZE,
        page_size=_GATHER_PAGE_SIZE,
        dtype=_GATHER_DTYPE,
        layer_num=layer_num,
        device=f"cuda:{rank}",
        enable_memory_saver=False,
        start_layer=start_layer,
        end_layer=start_layer + layer_num - 1,
        shard_spec=PageShardSpec(
            shard_rank=group.rank_in_group,
            shard_size=_GATHER_WORLD,
            page_size=_GATHER_PAGE_SIZE,
            max_prefix_tokens=16 * _GATHER_WORLD * _GATHER_PAGE_SIZE,
            chunk_tokens=4 * _GATHER_PAGE_SIZE,
        ),
        shard_group=group,
    )
    if kind == "mla":
        pool = PageInterleaveMLATokenToKVPool(
            **kwargs,
            kv_lora_rank=_GATHER_KV_LORA_RANK,
            qk_rope_head_dim=_GATHER_QK_ROPE,
        )
    else:
        pool = PageInterleaveMHATokenToKVPool(
            **kwargs,
            head_num=_GATHER_HEAD_NUM,
            head_dim=_GATHER_HEAD_DIM,
            enable_alt_stream=False,
        )
    for local_layer in range(layer_num):
        for buffer in _mtp_buffers(pool, kind, local_layer):
            buffer.fill_(_MTP_UNWRITTEN)
    return pool


def _mtp_buffers(pool, kind, local_layer):
    if kind == "mla":
        return (pool.kv_buffer[local_layer],)
    return (pool.k_buffer[local_layer], pool.v_buffer[local_layer])


def _mtp_make_draft_pool(target, allocator, kind, start_layer):
    # Skip model loading while using the production pool dispatch and sharing
    # the target allocator. A replicated draft pool must fail this test.
    kvc = KVCacheConfigurator.__new__(KVCacheConfigurator)
    kvc.device = target.device
    kvc.is_draft_worker = True
    kvc.page_size = _GATHER_PAGE_SIZE
    kvc.use_mla_backend = kind == "mla"
    kvc.kv_cache_dtype = _GATHER_DTYPE
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
        kv_lora_rank=_GATHER_KV_LORA_RANK,
        qk_rope_head_dim=_GATHER_QK_ROPE,
        head_dim=_GATHER_HEAD_DIM,
        v_head_dim=_GATHER_HEAD_DIM,
        get_num_kv_heads=lambda *_args: _GATHER_HEAD_NUM,
    )
    kvc.layer_info = SimpleNamespace(
        start_layer=start_layer, end_layer=start_layer + 1, num_effective_layers=1
    )
    with get_context().override_server_args(
        page_size=_GATHER_PAGE_SIZE,
        tp_size=_GATHER_WORLD,
        attn_cp_size=_GATHER_WORLD if kind == "mha" else 1,
        speculative_algorithm="EAGLE",
        enable_multi_layer_eagle=False,
    ):
        pool = kvc._build_token_to_kv_pool(
            sizes=SimpleNamespace(max_total_num_tokens=target.size),
            is_dsa_model=False,
            is_dsv4_model=False,
            req_to_token_pool=None,
        )
    assert pool.shard_spec is target.shard_spec
    assert pool.shard_group is target.shard_group
    for buffer in _mtp_buffers(pool, kind, 0):
        buffer.fill_(_MTP_UNWRITTEN)
    return pool


def _mtp_values(locs, width, tag, local_layer, is_v=False):
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
    return values.to(_GATHER_DTYPE)


def _mtp_canonical(kind, locs, tag, local_layer):
    if kind == "mla":
        return (
            _mtp_values(
                locs, _GATHER_KV_LORA_RANK + _GATHER_QK_ROPE, tag, local_layer
            ).unsqueeze(1),
        )
    return tuple(
        _mtp_values(
            locs, _GATHER_HEAD_NUM * _GATHER_HEAD_DIM, tag, local_layer, is_v
        ).reshape(-1, _GATHER_HEAD_NUM, _GATHER_HEAD_DIM)
        for is_v in (False, True)
    )


def _mtp_assert_bytes(got, expected, label):
    assert got.shape == expected.shape, label
    assert got.dtype == expected.dtype == _GATHER_DTYPE, label
    assert torch.equal(
        got.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
    ), f"{label}: KV bytes differ"


def _mtp_logical_chain(device):
    # Owners alternate from rank 1, while local pages run out of order. Some
    # logical slots exceed the physical pool size, catching a replicated draft
    # indexing them raw.
    local_pages = torch.tensor([25, 3, 20, 6, 28, 4, 30, 8, 22], device=device)
    owners = (1 + torch.arange(local_pages.numel(), device=device)) % _GATHER_WORLD
    pages = local_pages * _GATHER_WORLD + owners
    return (
        pages[:, None] * _GATHER_PAGE_SIZE
        + torch.arange(_GATHER_PAGE_SIZE, device=device)
    ).reshape(-1)


def _mtp_write(pool, kind, locs, tag, local_layer, epoch):
    layer = SimpleNamespace(layer_id=pool.start_layer + local_layer)
    values = _mtp_canonical(kind, locs, tag, local_layer)
    if kind == "mha":
        pool.set_kv_buffer(layer, locs, *values)
    elif epoch % 2:
        # Both MLA write entry points must owner-filter and stage MTP rows.
        pool.set_mla_kv_buffer(
            layer,
            locs,
            values[0][..., :_GATHER_KV_LORA_RANK],
            values[0][..., _GATHER_KV_LORA_RANK:],
        )
    else:
        pool.set_kv_buffer(
            layer, locs, values[0], values[0][..., :_GATHER_KV_LORA_RANK]
        )


def _mtp_check_persisted(pool, kind, locs, tag, rank):
    owned_locs = locs[(locs // _GATHER_PAGE_SIZE) % _GATHER_WORLD == rank]
    local_rows = (
        owned_locs // (_GATHER_WORLD * _GATHER_PAGE_SIZE) * _GATHER_PAGE_SIZE
        + owned_locs % _GATHER_PAGE_SIZE
    )
    for local_layer in range(pool.layer_num):
        canonical = _mtp_canonical(kind, owned_locs, tag, local_layer)
        for buffer, values in zip(_mtp_buffers(pool, kind, local_layer), canonical):
            expected = torch.full_like(buffer, _MTP_UNWRITTEN)
            expected[local_rows] = values
            _mtp_assert_bytes(
                buffer, expected, f"{kind} pool {tag} owned layer {local_layer}"
            )


def _mtp_check_scratch(pool, kind, locs, prefix_len, tag, local_layer):
    layer_id = pool.start_layer + local_layer
    rows = pool.translate_loc_to_scratch(locs)
    expected = _mtp_canonical(kind, locs, tag, local_layer)
    _mtp_assert_bytes(
        pool.get_key_buffer(layer_id)[rows], expected[0], f"{kind} pool {tag} key"
    )
    _mtp_assert_bytes(
        pool.get_value_buffer(layer_id)[rows],
        expected[0][..., :_GATHER_KV_LORA_RANK] if kind == "mla" else expected[1],
        f"{kind} pool {tag} value",
    )
    if kind == "mla" and prefix_len:
        # Arbitrary prefix indices also exercise the chunked MLA consumer.
        subset = locs[3:prefix_len:7].clone()
        k_nope, k_rope = pool.get_mla_kv_buffer(
            SimpleNamespace(layer_id=layer_id), subset, _GATHER_DTYPE
        )
        reference = _mtp_canonical(kind, subset, tag, local_layer)[0]
        _mtp_assert_bytes(
            k_nope, reference[..., :_GATHER_KV_LORA_RANK], f"pool {tag} latent"
        )
        _mtp_assert_bytes(
            k_rope, reference[..., _GATHER_KV_LORA_RANK:], f"pool {tag} rope"
        )


def _run_mtp(rank, port, kind):
    """Check exact KV bytes and independent target/draft state across batches."""
    _dist_init(
        rank, _GATHER_WORLD, port, attn_cp_size=_GATHER_WORLD if kind == "mha" else 1
    )
    group = get_kv_shard_group(use_mla_backend=kind == "mla")
    assert group.world_size == _GATHER_WORLD
    target = _mtp_make_pool(
        rank, group, kind, start_layer=0, layer_num=_GATHER_LAYER_NUM
    )
    allocator = PageInterleavePoolAllocator(
        size=target.size,
        physical_page_size=_GATHER_PAGE_SIZE,
        shard_size=_GATHER_WORLD,
        dtype=_GATHER_DTYPE,
        device=target.device,
        kvcache=target,
        need_sort=False,
        shard_spec=target.shard_spec,
    )
    # DeepSeek's NextN uses local layer 0. Also cover an absolute/offset layer ID
    # so the one-layer gather terminates correctly on either scratch-slot parity.
    drafts = [
        _mtp_make_draft_pool(target, allocator, kind, start_layer=start)
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

    locs = _mtp_logical_chain(target.device)
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
                _mtp_write(pool, kind, chunk_locs, tag, local_layer, epoch)
                _mtp_check_scratch(
                    pool, kind, active_locs, prefix_len, tag, local_layer
                )
            _mtp_check_persisted(pool, kind, active_locs, tag, rank)
            # MTP metadata/writes/reads must leave the target's last resident
            # layer and gather plan usable, despite identical logical locations.
            assert target._epoch == target_epoch + 1
            _mtp_check_scratch(
                target, kind, active_locs, prefix_len, 1, _GATHER_LAYER_NUM - 1
            )
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
    _mtp_check_scratch(drafts[1], kind, locs, 128, 3, 0)
    _mtp_check_scratch(target, kind, locs, 128, 1, _GATHER_LAYER_NUM - 1)
    torch.cuda.synchronize()
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


@unittest.skipIf(
    torch.cuda.device_count() < 2, "page-interleave gather needs 2 CUDA devices"
)
class TestPageInterleaveGatherMultiGpu(CustomTestCase):
    """Real pools, real NCCL, 2 ranks — one mp.spawn per phase.

    Separate spawns (and separate ports) because each phase builds its own
    process group with a different attention-CP width.
    """

    def test_mla_shard_over_attention_tp(self):
        mp.spawn(_run_mla, args=(_GATHER_WORLD, 29811), nprocs=_GATHER_WORLD, join=True)

    def test_mha_shard_over_attention_cp(self):
        mp.spawn(_run_mha, args=(_GATHER_WORLD, 29812), nprocs=_GATHER_WORLD, join=True)

    def test_mtp_mla_shard_over_attention_tp(self):
        mp.spawn(_run_mtp, args=(29813, "mla"), nprocs=_GATHER_WORLD, join=True)

    def test_mtp_mha_shard_over_attention_cp(self):
        mp.spawn(_run_mtp, args=(29814, "mha"), nprocs=_GATHER_WORLD, join=True)


if __name__ == "__main__":
    unittest.main()
