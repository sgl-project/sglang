# Copyright 2023-2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""DSA indexer caches and attention over page-interleaved KV pools.

The pool tests check byte-exact indexer keys, scales, and latent KV on 2 and 4
GPUs, including packed FP8 reads, page reuse, and asynchronous layer prefetch.
Writer and metadata tests cover fused stores and CP request filtering. The
attention test reuses the pool fixture to compare real indexer reads and sparse
attention with an unsharded cache, including full and shared indexer layers,
packed FlashMLA FP8, and raw TRT-LLM FP8 caches.

General placement, allocation, scratch planning, and MLA/MHA gather tests live
in ``test_page_interleave_shard.py``.
"""

import os
import tempfile
import unittest
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import torch
import torch.multiprocessing as mp

from sglang.srt.environ import envs
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool
from sglang.srt.mem_cache.page_interleave_pool import (
    PageInterleaveDSATokenToKVPool,
    PageInterleaveKVPoolMixin,
)
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase, publish_build_topology

register_cpu_ci(est_time=3, suite="base-a-test-cpu")
register_cuda_ci(est_time=120, stage="base-c", runner_config="4-gpu-b200")


# =============================================================================
# Indexer: packed FP8 keys and FP32 scales over real NCCL (2 and 4 GPUs).
# =============================================================================

_INDEXER_PAGE_SIZE = 64
_INDEXER_SIZE = _INDEXER_PAGE_SIZE * 32
_INDEXER_HEAD_DIM = 128
_INDEXER_KV_LORA_RANK = 512
_INDEXER_QK_ROPE = 64
_INDEXER_START_LAYER = 3
_INDEXER_SKIP_TOPK = [False, True, False, False, False]


def _assert_bytes(got, expected, label):
    got = got.contiguous().view(torch.uint8)
    expected = expected.contiguous().view(torch.uint8)
    assert got.shape == expected.shape, label
    assert torch.equal(got, expected), f"{label}: cache bytes differ"


def _index_values(locs, layer_id, generation):
    dims = torch.arange(_INDEXER_HEAD_DIM, device=locs.device)
    keys = (
        (locs[:, None] * 7 + dims * 3 + layer_id * 11 + generation * 13) % 31 - 15
    ).to(torch.float8_e4m3fn)
    scales = (
        (locs % 29 + layer_id * 31 + generation * 17 + 1).float() / 128
    ).unsqueeze(1)
    return keys, scales


def _latent_values(locs, layer_id, generation):
    dims = torch.arange(_INDEXER_KV_LORA_RANK + _INDEXER_QK_ROPE, device=locs.device)
    values = ((locs[:, None] * 3 + dims + layer_id * 5 + generation * 7) % 113 - 56).to(
        torch.bfloat16
    ) / 16
    # Vary the quantization scale across tokens and 128-value NoPE groups.
    amplitude = torch.exp2(
        ((locs[:, None] + dims // 128 + layer_id + generation) % 5).float() - 2
    ).to(torch.bfloat16)
    return (values * amplitude).unsqueeze(1)


def _locs(pages, seq_len, device):
    return (
        torch.tensor(pages, dtype=torch.int64, device=device)[:, None]
        * _INDEXER_PAGE_SIZE
        + torch.arange(_INDEXER_PAGE_SIZE, device=device)
    ).flatten()[:seq_len]


def _unpack_index(buf, pages, seq_len):
    """Independent unpacking of [page FP8 keys | page FP32 scales]."""
    selected = buf[pages.long()]
    split = _INDEXER_PAGE_SIZE * _INDEXER_HEAD_DIM
    keys = selected[:, :split].reshape(-1, _INDEXER_HEAD_DIM)[:seq_len]
    scales = selected[:, split:].reshape(-1, 4)[:seq_len]
    return keys, scales


def _unpack_latent(buf, locs, packed):
    """Decode reference rows independently of the production DSA kernels."""
    selected = buf[locs]
    if not packed:
        return (
            selected[..., :_INDEXER_KV_LORA_RANK],
            selected[..., _INDEXER_KV_LORA_RANK:],
        )
    raw = selected.contiguous().view(torch.uint8)
    scale_end = _INDEXER_KV_LORA_RANK + _INDEXER_KV_LORA_RANK // 128 * 4
    values = raw[..., :_INDEXER_KV_LORA_RANK].view(torch.float8_e4m3fn).float()
    scales = (
        raw[..., _INDEXER_KV_LORA_RANK:scale_end]
        .contiguous()
        .view(torch.float32)
        .repeat_interleave(128, dim=-1)
    )
    return (
        (values * scales).to(torch.bfloat16),
        raw[..., scale_end:].contiguous().view(torch.bfloat16),
    )


class _IndexerCacheCheck:
    def __init__(self, group, packed, *, raw_fp8=False, prefix_contexts=8):
        from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool
        from sglang.srt.mem_cache.page_interleave import PageShardSpec
        from sglang.srt.mem_cache.page_interleave_pool import (
            PageInterleaveDSATokenToKVPool,
        )

        self.group = group
        self.world = group.world_size
        self.rank = group.rank_in_group
        self.device = group.device
        self.packed = packed
        assert not (packed and raw_fp8)
        self.kwargs = kwargs = dict(
            page_size=_INDEXER_PAGE_SIZE,
            kv_lora_rank=_INDEXER_KV_LORA_RANK,
            dtype=torch.float8_e4m3fn if packed or raw_fp8 else torch.bfloat16,
            qk_rope_head_dim=_INDEXER_QK_ROPE,
            layer_num=len(_INDEXER_SKIP_TOPK),
            device=str(self.device),
            index_head_dim=_INDEXER_HEAD_DIM,
            enable_memory_saver=False,
            kv_cache_dim=656 if packed else _INDEXER_KV_LORA_RANK + _INDEXER_QK_ROPE,
            start_layer=_INDEXER_START_LAYER,
            end_layer=_INDEXER_START_LAYER + len(_INDEXER_SKIP_TOPK) - 1,
            skip_topk_layers=_INDEXER_SKIP_TOPK,
        )
        self.pool = PageInterleaveDSATokenToKVPool(
            _INDEXER_SIZE,
            **kwargs,
            shard_spec=PageShardSpec(
                shard_rank=self.rank,
                shard_size=self.world,
                page_size=_INDEXER_PAGE_SIZE,
                max_prefix_tokens=prefix_contexts * self.world * _INDEXER_PAGE_SIZE,
                chunk_tokens=8 * _INDEXER_PAGE_SIZE,
            ),
            shard_group=group,
        )
        # Include each rank's reserved/padded physical page in the reference.
        self.reference = DSATokenToKVPool(
            (_INDEXER_SIZE + _INDEXER_PAGE_SIZE) * self.world, **kwargs
        )

    def rejects_kpool(self):
        from sglang.srt.mem_cache.page_interleave import PageShardSpec
        from sglang.srt.mem_cache.page_interleave_pool import (
            PageInterleaveDSATokenToKVPool,
        )

        # A 128-token page with index_kpool=2 holds 64 indexer slots, so the
        # token-unit location translation would address the wrong rows.
        page_size = 2 * _INDEXER_PAGE_SIZE
        with unittest.TestCase().assertRaisesRegex(
            AssertionError, "does not support index_kpool=2"
        ):
            PageInterleaveDSATokenToKVPool(
                _INDEXER_SIZE,
                **{**self.kwargs, "page_size": page_size, "index_kpool": 2},
                shard_spec=PageShardSpec(
                    shard_rank=self.rank,
                    shard_size=self.world,
                    page_size=page_size,
                    max_prefix_tokens=8 * self.world * page_size,
                    chunk_tokens=8 * page_size,
                ),
                shard_group=self.group,
            )

    def store(self, locs, layer_id, generation):
        values = _latent_values(locs, layer_id, generation)
        for pool in (self.pool, self.reference):
            pool.set_mla_kv_buffer(
                SimpleNamespace(layer_id=layer_id),
                locs,
                values[..., :_INDEXER_KV_LORA_RANK].contiguous(),
                values[..., _INDEXER_KV_LORA_RANK:].contiguous(),
            )
        if _INDEXER_SKIP_TOPK[layer_id - _INDEXER_START_LAYER]:
            return
        keys, scales = _index_values(locs, layer_id, generation)
        write_scales = scales[:, 0] if generation % 2 else scales
        for pool in (self.pool, self.reference):
            pool.set_index_k_scale_buffer(layer_id, locs, keys, write_scales)

        # Check the reference's packing independently, including both scales
        # shapes accepted by SetKAndS and every valid token in partial pages.
        buf = self.reference.index_key_cache.get_local_buffer(layer_id)
        pages, offsets = locs // _INDEXER_PAGE_SIZE, locs % _INDEXER_PAGE_SIZE
        unpacked_keys = buf[:, : _INDEXER_PAGE_SIZE * _INDEXER_HEAD_DIM].reshape(
            -1, _INDEXER_PAGE_SIZE, _INDEXER_HEAD_DIM
        )
        unpacked_scales = buf[:, _INDEXER_PAGE_SIZE * _INDEXER_HEAD_DIM :].reshape(
            -1, _INDEXER_PAGE_SIZE, 4
        )
        _assert_bytes(unpacked_keys[pages, offsets], keys, "reference FP8 keys")
        _assert_bytes(unpacked_scales[pages, offsets], scales, "reference FP32 scales")

    def assert_owned(self, layer_id):
        local_layer = layer_id - _INDEXER_START_LAYER
        owned = self.pool.index_key_cache.get_local_buffer(layer_id)
        assert (
            owned.data_ptr()
            == self.pool.index_k_with_scale_buffer[local_layer].data_ptr()
        )
        if _INDEXER_SKIP_TOPK[local_layer]:
            assert owned.shape[0] == 0
            return
        logical_pages = (
            torch.arange(owned.shape[0], device=self.device) * self.world + self.rank
        )
        expected = self.reference.index_k_with_scale_buffer[local_layer][logical_pages]
        # Comparing the whole allocation catches writes to foreign pages and
        # overwrites of untouched bytes in a partially filled page.
        _assert_bytes(owned, expected, f"owned index layer {layer_id}")

    def seed(self, pages, generation):
        self.pool.end_shard_extend()
        locs = _locs(pages, len(pages) * _INDEXER_PAGE_SIZE, self.device)
        for layer_id in range(
            _INDEXER_START_LAYER, _INDEXER_START_LAYER + len(_INDEXER_SKIP_TOPK)
        ):
            self.store(locs, layer_id, generation)
            self.assert_owned(layer_id)

    def assert_latent_reads(self, layer_id, rows):
        # Cover every gathered prefix/chunk row in reverse order, repeated
        # locations, partial pages, and a noncontiguous logical-location tensor.
        all_locs = torch.cat(rows)
        selected = torch.cat((all_locs.flip(0), all_locs[1:2].expand(2)))
        storage = torch.empty(
            selected.numel() * 2, dtype=selected.dtype, device=self.device
        )
        read_locs = storage[::2]
        read_locs.copy_(selected)
        reference = self.reference.kv_buffer[layer_id - _INDEXER_START_LAYER]
        layer = SimpleNamespace(layer_id=layer_id)
        for locs in (read_locs, read_locs[:0]):
            expected = _unpack_latent(reference, locs, self.packed)
            for dtype in (None, torch.bfloat16, torch.float16, torch.float32):
                actual = self.pool.get_mla_kv_buffer(layer, locs, dst_dtype=dtype)
                expected_dtype = dtype or torch.bfloat16
                for got, value in zip(actual, expected):
                    assert got.shape == value.shape
                    assert got.dtype == expected_dtype
                    assert got.is_contiguous()
                    _assert_bytes(got, value.to(expected_dtype), "decoded latent KV")

    def read_index(self, layer_id, pages, seq_len):
        """Gather one request's keys and scales by logical page IDs."""
        lengths = torch.tensor([seq_len], dtype=torch.int32, device=self.device)
        return self.pool.get_index_k_scale_buffer(
            layer_id, lengths, pages[None], seq_len, seq_len
        )

    def batch(self, requests, generation):
        """requests contains (logical pages, prefix token count, sequence length)."""
        seq_lens = [seq_len for _, _, seq_len in requests]
        prefix_lens = [prefix_len for _, prefix_len, _ in requests]
        rows = [_locs(pages, seq_len, self.device) for pages, _, seq_len in requests]
        # Nonconsecutive request IDs ensure the plan indexes req_pool_indices.
        req_ids = torch.arange(len(rows), device=self.device) * 2 + 1
        req_to_token = torch.zeros(
            (2 * len(rows) + 1, max(seq_lens)), dtype=torch.int64, device=self.device
        )
        page_table = torch.zeros(
            (len(rows), max(len(pages) for pages, _, _ in requests)),
            dtype=torch.int32,
            device=self.device,
        )
        for i, ((pages, _, seq_len), row) in enumerate(zip(requests, rows)):
            req_to_token[req_ids[i], :seq_len] = row
            page_table[i, : len(pages)] = torch.tensor(pages, device=self.device)
        chunk_locs = torch.cat([row[p:] for row, p in zip(rows, prefix_lens)])
        self.pool.begin_shard_extend(req_to_token, req_ids, prefix_lens, seq_lens)

        for layer_id in range(
            _INDEXER_START_LAYER, _INDEXER_START_LAYER + len(_INDEXER_SKIP_TOPK)
        ):
            if chunk_locs.numel():
                self.store(chunk_locs, layer_id, generation)
            # Read before the raw-key accessor so MLA reads must themselves
            # wait for the prefix allgather and include the locally staged chunk.
            self.assert_latent_reads(layer_id, rows)
            if _INDEXER_SKIP_TOPK[layer_id - _INDEXER_START_LAYER]:
                assert self.pool.get_index_k_with_scale_buffer(layer_id).shape[0] == 0
            # Every layer advances prefetch, including layers reusing top-k.
            latent = self.pool.get_key_buffer(layer_id)
            for row in rows:
                scratch_rows = self.pool.translate_loc_to_scratch(row)
                expected = self.reference.kv_buffer[layer_id - _INDEXER_START_LAYER][
                    row
                ]
                _assert_bytes(latent[scratch_rows], expected, "latent KV")

            self.assert_owned(layer_id)
            if _INDEXER_SKIP_TOPK[layer_id - _INDEXER_START_LAYER]:
                with unittest.TestCase().assertRaisesRegex(AssertionError, "skip-topk"):
                    self.read_index(layer_id, page_table[0], 1)
                continue

            # Repeated public reads must preserve the resident layer while
            # the next layer's NCCL allgather is in flight on the other slot.
            for _ in range(2):
                for i, (_, _, seq_len) in enumerate(requests):
                    pages = page_table[i].contiguous()
                    reference_buf = self.reference.get_index_k_with_scale_buffer(
                        layer_id
                    )
                    expected_k, expected_s = _unpack_index(
                        reference_buf, pages, seq_len
                    )
                    got_k, got_s = self.read_index(layer_id, pages, seq_len)
                    _assert_bytes(got_k, expected_k, "per-request FP8 keys")
                    _assert_bytes(got_s, expected_s, "per-request FP32 scales")
                    scratch_pages = (
                        self.pool.translate_loc_to_scratch(
                            pages.long() * _INDEXER_PAGE_SIZE
                        )
                        // _INDEXER_PAGE_SIZE
                    )
                    raw = self.pool.get_index_k_with_scale_buffer(layer_id)
                    raw_k, raw_s = _unpack_index(raw, scratch_pages, seq_len)
                    _assert_bytes(raw_k, expected_k, "raw scratch FP8 keys")
                    _assert_bytes(raw_s, expected_s, "raw scratch FP32 scales")

                lengths = torch.tensor(seq_lens, dtype=torch.int32, device=self.device)
                got = self.pool.get_index_k_scale_buffer(
                    layer_id, lengths, page_table, sum(seq_lens), max(seq_lens)
                )
                expected = self.reference.get_index_k_scale_buffer(
                    layer_id, lengths, page_table, sum(seq_lens), max(seq_lens)
                )
                for actual, reference in zip(got, expected):
                    _assert_bytes(actual, reference, "batched index gather")

        self.pool.end_shard_extend()
        for layer_id in range(
            _INDEXER_START_LAYER, _INDEXER_START_LAYER + len(_INDEXER_SKIP_TOPK)
        ):
            assert self.pool.get_index_k_with_scale_buffer(layer_id).data_ptr() == (
                self.pool.index_key_cache.get_local_buffer(layer_id).data_ptr()
            )
        torch.cuda.synchronize()
        torch.distributed.barrier()

    def mutated_index_locations(self):
        # Indexer callers may reuse the same storage with new logical locations;
        # a write plan cached only by pointer/length must not survive that change.
        mutable_loc = torch.empty(1, dtype=torch.int64, device=self.device)
        for active in (False, True):
            pages = torch.tensor(
                [28 * self.world + 1, 29 * self.world], device=self.device
            )
            if active:
                self.pool.begin_shard_extend(
                    pages[:, None] * _INDEXER_PAGE_SIZE,
                    torch.arange(2, device=self.device),
                    [0, 0],
                    [1, 1],
                )
            else:
                with unittest.TestCase().assertRaisesRegex(
                    AssertionError, "begin_shard_extend"
                ):
                    self.read_index(_INDEXER_START_LAYER, pages[:1], 1)

            for layer_id in range(
                _INDEXER_START_LAYER, _INDEXER_START_LAYER + len(_INDEXER_SKIP_TOPK)
            ):
                # An empty store must also be harmless on a skip-topk layer.
                empty_loc = mutable_loc[:0]
                empty_k, empty_s = _index_values(empty_loc, layer_id, 1)
                self.pool.set_index_k_scale_buffer(
                    layer_id, empty_loc, empty_k, empty_s
                )
                if _INDEXER_SKIP_TOPK[layer_id - _INDEXER_START_LAYER]:
                    continue

                for i in range(2):
                    mutable_loc.copy_(pages[i : i + 1] * _INDEXER_PAGE_SIZE)
                    # Both an in-place mutation and another tensor object
                    # aliasing the old allocation retain the same data pointer.
                    loc = mutable_loc if i == 0 else mutable_loc.view_as(mutable_loc)
                    keys, scales = _index_values(loc, layer_id, 7 + i + int(active))
                    for pool in (self.pool, self.reference):
                        pool.set_index_k_scale_buffer(layer_id, loc, keys, scales)
                    self.assert_owned(layer_id)

                if active:
                    for i in range(2):
                        expected_k, expected_s = _index_values(
                            pages[i : i + 1] * _INDEXER_PAGE_SIZE, layer_id, 8 + i
                        )
                        got_k, got_s = self.read_index(layer_id, pages[i : i + 1], 1)
                        _assert_bytes(got_k, expected_k, "mutated location FP8 keys")
                        _assert_bytes(got_s, expected_s, "mutated location FP32 scales")
            self.pool.end_shard_extend()
        torch.cuda.synchronize()
        torch.distributed.barrier()

    def async_prefix(self, pages):
        """Queue all layers' reads before synchronizing to exercise slot reuse."""
        locs = _locs(pages, len(pages) * _INDEXER_PAGE_SIZE, self.device)
        page_ids = torch.tensor(pages, dtype=torch.int32, device=self.device)
        self.pool.begin_shard_extend(
            locs[None, :],
            torch.tensor([0], device=self.device),
            [locs.numel()],
            [locs.numel()],
        )
        scratch_pages = (
            self.pool.translate_loc_to_scratch(page_ids.long() * _INDEXER_PAGE_SIZE)
            // _INDEXER_PAGE_SIZE
        )
        snapshots = []
        for layer_id in range(
            _INDEXER_START_LAYER, _INDEXER_START_LAYER + len(_INDEXER_SKIP_TOPK)
        ):
            decoded = self.pool.get_mla_kv_buffer(
                SimpleNamespace(layer_id=layer_id), locs
            )
            expected_latent = _unpack_latent(
                self.reference.kv_buffer[layer_id - _INDEXER_START_LAYER],
                locs,
                self.packed,
            )
            snapshots.extend(zip(decoded, expected_latent))
            raw = self.pool.get_index_k_with_scale_buffer(layer_id)
            if _INDEXER_SKIP_TOPK[layer_id - _INDEXER_START_LAYER]:
                assert raw.shape[0] == 0
                continue
            reference = self.reference.get_index_k_with_scale_buffer(layer_id)
            snapshots.append((raw[scratch_pages].clone(), reference[page_ids]))
            expected_k, expected_s = _unpack_index(reference, page_ids, locs.numel())
            got_k, got_s = self.read_index(layer_id, page_ids, locs.numel())
            snapshots.extend(((got_k, expected_k), (got_s, expected_s)))
        torch.cuda.synchronize()
        for actual, expected in snapshots:
            _assert_bytes(actual, expected, "asynchronous layer pipeline")
        self.pool.end_shard_extend()
        torch.distributed.barrier()


def _run_indexer(rank, world, init_method):
    os.environ.update(
        RANK=str(rank),
        WORLD_SIZE=str(world),
    )
    os.environ.setdefault("no_proxy", "127.0.0.1,localhost")
    os.environ["SGLANG_DEBUG_MEMORY_POOL"] = "1"
    torch.cuda.set_device(rank)
    from sglang.srt.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from sglang.srt.runtime_context import get_parallel

    init_distributed_environment(
        world_size=world,
        rank=rank,
        local_rank=rank,
        distributed_init_method=init_method,
        backend="nccl",
    )
    publish_build_topology(tp_size=world, attn_cp_size=world, world_rank=rank)
    initialize_model_parallel()
    with get_parallel().override(attn_cp_size=world):
        group = get_parallel().attn_cp_group
        for packed in (False, True):
            check = _IndexerCacheCheck(group, packed)
            check.rejects_kpool()
            # Fragmented physical pages with a rotated cyclic ownership run.
            prefix = [
                physical * world + (i + 1) % world
                for i, physical in enumerate([5, 2, 9, 4, 7][: world + 1])
            ]
            check.seed(prefix, generation=1)
            chunk = [12 * world + (world + 2) % world, 11 * world + (world + 3) % world]
            check.batch(
                [
                    (
                        prefix + chunk,
                        len(prefix) * _INDEXER_PAGE_SIZE,
                        (len(prefix) + 1) * _INDEXER_PAGE_SIZE + 13,
                    ),
                    (
                        [prefix[0], 16 * world + 2 % world],
                        _INDEXER_PAGE_SIZE,
                        _INDEXER_PAGE_SIZE + 7,
                    ),
                ],
                generation=2,
            )
            # Prefix-only batch: exactly one rank owns a page, all peers send
            # padding. Also checks double-buffer epoch invalidation.
            check.batch(
                [([prefix[0]], _INDEXER_PAGE_SIZE, _INDEXER_PAGE_SIZE)], generation=3
            )
            fresh = [20 * world + world - 1, 21 * world]
            check.batch([(fresh, 0, _INDEXER_PAGE_SIZE + 9)], generation=4)
            # Same logical prefix and layer IDs, new bytes after page reuse.
            check.seed([prefix[0]], generation=5)
            check.batch(
                [
                    (
                        [prefix[0], 23 * world + 2 % world],
                        _INDEXER_PAGE_SIZE,
                        _INDEXER_PAGE_SIZE + 19,
                    )
                ],
                generation=6,
            )
            check.async_prefix(prefix)
            check.mutated_index_locations()
            print(f"rank {rank}: {'packed FP8' if packed else 'BF16'} DSA cache OK")
            del check
    destroy_model_parallel()
    destroy_distributed_environment()


class TestPageInterleaveIndexer(CustomTestCase):
    def _test_world(self, world):
        if torch.cuda.device_count() < world:
            self.skipTest(f"indexer KV sharding test requires {world} GPUs")
        with tempfile.TemporaryDirectory(prefix="sglang-indexer-shard-") as tmp:
            mp.spawn(
                _run_indexer,
                args=(world, f"file://{tmp}/rendezvous"),
                nprocs=world,
                join=True,
            )

    def test_two_gpu(self):
        self._test_world(2)

    def test_four_gpu(self):
        self._test_world(4)


class TestShardedDSAIndexerWrite(unittest.TestCase):
    def setUp(self):
        # Standalone pools size their index buffer from the DCP width.
        cm = get_parallel().override(attn_dcp_size=1)
        cm.__enter__()
        self.addCleanup(cm.__exit__, None, None, None)

    def test_cuda_writers_match_unsharded_bytes(self):
        from sglang.srt.layers.attention.dsa import dsa_indexer

        if not dsa_indexer._is_cuda:
            self.skipTest("requires CUDA indexer writers")

        from sglang.kernels.ops.quantization.act_quant_triton import act_quant
        from sglang.srt.mem_cache.page_interleave import (
            PageInterleavePlacement,
            PageShardSpec,
        )

        self.assertTrue(
            dsa_indexer.can_use_dsa_fused_store(torch.bfloat16, torch.int64, 64),
            "both pools must take the CUDA fused writers",
        )
        device = "cuda"
        generator = torch.Generator(device=device).manual_seed(20261013)
        num_tokens, num_pages, page_size = 2305, 80, 64
        # Nonconsecutive logical locations span both owners and partial pages.
        locations = (
            torch.randperm(
                (num_pages - 2) * page_size, device=device, generator=generator
            )[:num_tokens]
            + 2 * page_size
        )
        key = torch.randn(
            (num_tokens, 128), device=device, dtype=torch.bfloat16, generator=generator
        )
        angles = torch.randn((4096, 32), device=device, generator=generator)
        positions = torch.randint(
            0, angles.shape[0], (num_tokens,), device=device, generator=generator
        )
        indexer = SimpleNamespace(
            block_size=128,
            scale_fmt="ue8m0",
            k_norm=SimpleNamespace(
                weight=torch.randn((128,), device=device, generator=generator),
                bias=torch.randn((128,), device=device, generator=generator),
                variance_epsilon=1e-6,
            ),
            _indexer_cos_sin_cache=torch.cat((angles.cos(), angles.sin()), dim=-1),
        )
        indexer._store_index_k_cache = MethodType(
            dsa_indexer.Indexer._store_index_k_cache, indexer
        )
        batch = SimpleNamespace(out_cache_loc=locations)
        pool_args = dict(
            page_size=page_size,
            kv_lora_rank=512,
            dtype=torch.bfloat16,
            qk_rope_head_dim=64,
            layer_num=1,
            device=device,
            index_head_dim=128,
            enable_memory_saver=False,
            kv_cache_dim=576,
        )
        reference = DSATokenToKVPool(num_pages * page_size, **pool_args)
        sharded = PageInterleaveDSATokenToKVPool.__new__(PageInterleaveDSATokenToKVPool)
        # Rank 0 of 2 with an active extend; a write-only batch needs no gather.
        DSATokenToKVPool.__init__(sharded, num_pages * page_size // 2, **pool_args)
        sharded.shard_spec = PageShardSpec(
            shard_rank=0,
            shard_size=2,
            page_size=page_size,
            max_prefix_tokens=0,
            chunk_tokens=num_pages * page_size,
        )
        sharded.shard_rank = 0
        sharded.placement = PageInterleavePlacement(sharded.shard_spec)
        sharded._shard_extend_active = True
        reference_buffer = reference.index_key_cache.get_local_buffer(0)
        scratch = torch.zeros_like(reference_buffer)
        sharded._slots = [SimpleNamespace(tensors={"index_k": scratch})]
        sharded._page_pos = torch.arange(
            reference_buffer.shape[0], device=device, dtype=torch.int32
        ).roll(7)
        local_buffer = sharded.index_key_cache.get_local_buffer(0)

        for fused_prepare in (False, True):
            with self.subTest(fused_prepare=fused_prepare):
                for buffer in (reference_buffer, local_buffer, scratch):
                    buffer.zero_()
                for pool in (reference, sharded):
                    with patch.object(
                        dsa_indexer, "get_token_to_kv_pool", return_value=pool
                    ):
                        if fused_prepare:
                            dsa_indexer.Indexer._fused_k_prepare_and_store(
                                indexer, key, positions, batch, 0, act_quant
                            )
                        else:
                            indexer._store_index_k_cache(
                                batch, 0, key, act_quant=act_quant
                            )
                # Every key is staged at its scratch row; rank 0 keeps even pages.
                torch.testing.assert_close(
                    scratch[sharded._page_pos.long()], reference_buffer, rtol=0, atol=0
                )
                torch.testing.assert_close(
                    local_buffer, reference_buffer[::2], rtol=0, atol=0
                )
        torch.cuda.synchronize()


def _attention_backend(pool, req_to_token, device, prefill_impl="flashmla_sparse"):
    from sglang.srt.layers.attention.dsa.dsa_topk_backend import DSATopKBackend
    from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend
    from sglang.srt.layers.attention.kv_shard_hooks import get_kv_shard_pool

    backend = DeepseekSparseAttnBackend.__new__(DeepseekSparseAttnBackend)
    backend.device = device
    backend.token_to_kv_pool = pool
    backend.req_to_token = req_to_token
    backend.req_to_token_pool = SimpleNamespace(req_to_token=req_to_token)
    backend._kv_shard_pool = get_kv_shard_pool(pool)
    backend.dsa_index_kpool = 1
    backend.dsa_index_topk = 2048
    backend.dsa_topk_backend = DSATopKBackend.SGL_KERNEL
    backend.physical_page_size = 64
    backend.use_mha = False
    backend.use_fused_topk = False
    backend.num_q_heads = 8
    backend.dcp_size = 1
    backend.num_dcp_q_heads = backend.num_q_heads
    backend.kv_lora_rank = 512
    backend.qk_nope_head_dim = 128
    backend.qk_rope_head_dim = 64
    backend.kv_cache_dim = getattr(pool, "kv_cache_dim", 576)
    backend.kv_cache_dtype = pool.dtype
    backend.dsa_kv_cache_store_fp8 = pool.dsa_kv_cache_store_fp8
    backend.dsa_prefill_impl = prefill_impl
    backend.dsa_decode_impl = "flashmla_kv"
    backend.flashmla_kv_num_q_heads = 64
    backend.flashmla_kv_format = "V32"
    backend.hisparse_coordinator = None
    backend.device_sm_major = (
        torch.cuda.get_device_capability()[0]
        if torch.device(device).type == "cuda"
        else 0
    )
    backend._arange_buf = torch.arange(1024, dtype=torch.int32, device=device)
    if prefill_impl == "trtllm":
        from sglang.srt.layers.attention.trtllm_mla_backend import (
            make_persistent_multi_ctas_kv_counter_buffer,
        )

        backend.workspace_buffer = torch.empty(
            envs.SGLANG_FLASHINFER_WORKSPACE_SIZE.get(),
            dtype=torch.uint8,
            device=device,
        )
        backend._multi_ctas_kv_counter_buffer = (
            make_persistent_multi_ctas_kv_counter_buffer(
                torch.device(device), backend.num_q_heads, max_batch_size=128
            )
        )
    # Fix the backend choice and avoid optional top-k-v2 planning; the metadata
    # construction, cache accesses, index transforms, and attention are real.
    backend.set_dsa_prefill_impl = lambda _: None
    backend._build_topk_v2_plan = lambda _: None
    return backend


class _MetadataShardPool(PageInterleaveKVPoolMixin):
    def __init__(self):
        self.dsa_kv_cache_store_fp8 = True
        self.dtype = torch.float8_e4m3fn
        self.size = 64
        self.page_size = 64
        self.begin_args = None

    def begin_shard_extend(self, *args):
        self.begin_args = args

    def translate_loc_to_scratch(self, loc):
        # Nonidentity, page-aligned address translation. The real collective
        # and translation kernels are covered by the multi-GPU test below.
        return loc % 512


class TestShardedDSAMetadata(unittest.TestCase):
    def test_cp_request_filtering_keeps_both_address_spaces_aligned(self):
        from sglang.srt.layers.attention import dsa_backend
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        lengths = torch.tensor([129, 65, 193])
        req_ids = torch.tensor([2, 0, 4])
        req_to_token = torch.zeros((5, 193), dtype=torch.int64)
        for req_id, base in zip(req_ids, [11, 17, 23]):
            req_to_token[req_id] = base * 64 + torch.arange(193)
        batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            batch_size=3,
            req_pool_indices=req_ids,
            seq_lens=lengths,
            seq_lens_cpu=lengths,
            seq_lens_sum=int(lengths.sum()),
            extend_seq_lens=torch.ones(3, dtype=torch.int32),
            extend_seq_lens_cpu=[1, 1, 1],
            extend_prefix_lens_cpu=[128, 64, 192],
            attn_cp_metadata=object(),
        )
        pool = _MetadataShardPool()
        backend = _attention_backend(pool, req_to_token, "cpu")
        # Rank one of CP2 receives the middle request's only query. The other
        # requests must disappear from both indexer page tables and lengths.
        strategy = SimpleNamespace(
            shard_local_tokens=lambda tensor: tensor[1::2],
            shard_per_request=lambda *_: (
                [1],
                torch.tensor([1], dtype=torch.int32),
                [1],
                torch.tensor([1]),
            ),
        )
        with (
            patch.object(
                dsa_backend, "can_dsa_prefill_cp_interleave", return_value=True
            ),
            patch.object(dsa_backend, "get_cp_strategy", return_value=strategy),
            patch.object(
                dsa_backend, "pad_dsa_cache_seqlens", side_effect=lambda _, x: x
            ),
        ):
            backend.init_forward_metadata(batch)
        metadata = backend.get_indexer_metadata(0, batch)
        logical = req_to_token[req_ids[1], :65]
        scratch = pool.translate_loc_to_scratch(logical)
        torch.testing.assert_close(
            metadata.get_logical_page_table(), torch.tensor([[17, 18]])
        )
        torch.testing.assert_close(metadata.get_page_table_64(), torch.tensor([[1, 2]]))
        torch.testing.assert_close(metadata.get_page_table_1(), scratch[None])
        torch.testing.assert_close(
            backend.forward_metadata.page_table_1_flattened, scratch
        )
        torch.testing.assert_close(
            metadata.get_indexer_seq_len_cpu(), torch.tensor([65])
        )
        self.assertIs(pool.begin_args[1], req_ids)
        self.assertEqual(pool.begin_args[2], [128, 64, 192])


def _real_indexer_topk(check, backends, batch, layer_id, query, weights):
    from sglang.srt.layers.attention.dsa import dsa_indexer
    from sglang.srt.layers.attention.dsa.dsa_topk_backend import TopkTransformMethod

    indexer = SimpleNamespace(
        index_topk=2048,
        num_init_tokens=0,
        num_local_tokens=0,
        logits_with_pp_recv=False,
        sm_count=torch.cuda.get_device_properties(check.device).multi_processor_count,
        _pad_heads_for_deep_gemm=dsa_indexer.Indexer._pad_heads_for_deep_gemm,
    )
    for name in ("_with_real_sm_count", "_mask_init_and_local_tokens"):
        setattr(indexer, name, MethodType(getattr(dsa_indexer.Indexer, name), indexer))

    class RecordingMetadata:
        def __init__(self, metadata):
            self.metadata = metadata
            self.logits = None

        def __getattr__(self, name):
            return getattr(self.metadata, name)

        def topk_transform(self, logits, *args, **kwargs):
            self.logits = logits.clone()
            return self.metadata.topk_transform(logits, *args, **kwargs)

    results, logits = [], []
    for backend in backends:
        metadata = RecordingMetadata(backend.get_indexer_metadata(layer_id, batch))
        with patch.object(
            dsa_indexer, "get_token_to_kv_pool", return_value=backend.token_to_kv_pool
        ):
            results.append(
                dsa_indexer.Indexer._get_topk_ragged(
                    indexer, False, batch, layer_id, query, weights, metadata
                )
            )
        starts, ends = metadata.get_indexer_kvcache_range()
        columns = torch.arange(metadata.logits.shape[1], device=query.device)
        valid = (columns[None] >= starts[:, None]) & (columns[None] < ends[:, None])
        logits.append(metadata.logits[valid])
    torch.testing.assert_close(*logits, rtol=0, atol=0)
    expected = results[1]
    if metadata.topk_transform_method == TopkTransformMethod.PAGED:
        expected = torch.where(
            expected >= 0,
            check.pool.translate_loc_to_scratch(expected.clamp_min(0)),
            -1,
        ).to(results[0].dtype)
    # SGL top-k emits selected values in atomic insertion order; FlashInfer's
    # deterministic mode also fixes the order.
    torch.testing.assert_close(
        results[0].sort(dim=-1).values, expected.sort(dim=-1).values, rtol=0, atol=0
    )
    if backends[0].dsa_topk_backend.is_flashinfer():
        torch.testing.assert_close(results[0], expected, rtol=0, atol=0)
    # Use matching selection order for the subsequent exact attention check.
    return [expected, results[1]]


def _check_attention_batch(check, requests, *, generation, prefill_impl, topk_backend):
    from sglang.srt.layers.attention.dsa import dsa_indexer
    from sglang.srt.layers.attention.dsa.dsa_topk_backend import DSATopKBackend
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    device = check.device
    lengths = [length for _, _, length in requests]
    prefixes = [prefix for _, prefix, _ in requests]
    extends = [length - prefix for length, prefix in zip(lengths, prefixes)]
    rows = [_locs(pages, length, device) for pages, _, length in requests]
    req_ids = torch.arange(len(rows), device=device) * 2 + 1
    req_to_token = torch.zeros(
        (2 * len(rows) + 1, max(lengths)), device=device, dtype=torch.int32
    )
    for req_id, row in zip(req_ids, rows):
        req_to_token[req_id, : row.numel()] = row
    logical_table = req_to_token[req_ids].clone()
    chunk_locs = torch.cat([row[prefix:] for row, prefix in zip(rows, prefixes)])
    batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        batch_size=len(rows),
        req_pool_indices=req_ids,
        seq_lens=torch.tensor(lengths, device=device, dtype=torch.int64),
        seq_lens_cpu=torch.tensor(lengths, dtype=torch.int64),
        seq_lens_sum=sum(lengths),
        extend_seq_lens=torch.tensor(extends, device=device, dtype=torch.int32),
        extend_seq_lens_cpu=extends,
        extend_prefix_lens_cpu=prefixes,
        extend_num_tokens=sum(extends),
        out_cache_loc=chunk_locs,
        attn_cp_metadata=None,
        global_num_tokens_cpu=None,
    )
    backends = [
        _attention_backend(pool, req_to_token, device, prefill_impl)
        for pool in (check.pool, check.reference)
    ]
    for backend in backends:
        backend.use_fused_topk = True
        backend.dsa_topk_backend = DSATopKBackend(topk_backend)
        backend.init_forward_metadata(batch)
    torch.testing.assert_close(req_to_token[req_ids], logical_table, rtol=0, atol=0)
    sharded_metadata = backends[0].get_indexer_metadata(0, batch)
    torch.testing.assert_close(
        sharded_metadata.get_logical_page_table(), logical_table[:, ::64] // 64
    )
    torch.testing.assert_close(
        sharded_metadata.get_page_table_1(),
        check.pool.translate_loc_to_scratch(logical_table),
        check_dtype=False,
    )
    assert not torch.equal(
        sharded_metadata.get_page_table_64(), sharded_metadata.get_logical_page_table()
    )

    generator = torch.Generator(device=device).manual_seed(2026 + generation)
    q = (
        torch.randn(
            (sum(extends), 8, 576),
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )
        / 8
    )
    query = torch.randn(
        (sum(extends), 32, 128),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    gates = (
        torch.rand(
            (sum(extends), 32), generator=generator, device=device, dtype=torch.bfloat16
        )
        + 0.5
    )
    angles = torch.randn((max(lengths), 32), generator=generator, device=device)
    positions = torch.cat(
        [
            torch.arange(prefix, length, device=device)
            for prefix, length in zip(prefixes, lengths)
        ]
    )
    index_q, index_weights = dsa_indexer.fused_q_indexer_rope_first_quant(
        query,
        gates,
        128**-0.5 * 32**-0.5,
        torch.cat((angles.cos(), angles.sin()), dim=-1),
        positions,
    )
    # TRT-LLM's FP8 path quantizes the split query before dispatch. FlashMLA
    # consumes the BF16 query directly, including when its KV cache is FP8.
    query_args = (
        {"q": q[..., :512].contiguous(), "q_rope": q[..., 512:].contiguous()}
        if prefill_impl == "trtllm"
        else {"q": q}
    )
    topk = None
    for local_layer, shared in enumerate(_INDEXER_SKIP_TOPK):
        layer_id = _INDEXER_START_LAYER + local_layer
        if shared:
            assert check.pool.index_key_cache.get_local_buffer(layer_id).numel() == 0
        else:
            keys, scales = _index_values(chunk_locs, layer_id, generation)
            for pool in (check.pool, check.reference):
                pool.set_index_k_scale_buffer(layer_id, chunk_locs, keys, scales)
            topk = _real_indexer_topk(
                check, backends, batch, layer_id, index_q, index_weights
            )
        # The indexer read above acquires its slot and prefetches the next
        # layer before this layer writes its latent KV chunk.
        chunk = _latent_values(chunk_locs, layer_id, generation)
        for pool in (check.pool, check.reference):
            pool.set_mla_kv_buffer(
                SimpleNamespace(layer_id=layer_id),
                chunk_locs,
                chunk[..., :512].contiguous(),
                chunk[..., 512:].contiguous(),
            )
        layer = SimpleNamespace(
            layer_id=layer_id,
            is_cross_attention=False,
            tp_q_head_num=8,
            head_dim=576,
            v_head_dim=512,
            scaling=576**-0.5,
        )
        outputs = [
            backend.forward_extend(
                **query_args,
                k=chunk[..., :512].contiguous(),
                v=chunk[..., :512].contiguous(),
                k_rope=chunk[..., 512:].contiguous(),
                layer=layer,
                forward_batch=batch,
                save_kv_cache=False,
                topk_indices=topk_indices,
            )
            for backend, topk_indices in zip(backends, topk)
        ]
        torch.testing.assert_close(
            *outputs,
            rtol=0,
            atol=0,
            msg=f"{prefill_impl}: sharded output differs at layer {layer_id}",
        )
    check.pool.end_shard_extend()
    torch.cuda.synchronize()
    torch.distributed.barrier()


def _run_attention(rank, world, init_method):
    from sglang.kernels.ops.quantization.act_quant_triton import act_quant
    from sglang.srt.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )

    os.environ.update(
        RANK=str(rank), WORLD_SIZE=str(world), SGLANG_DEBUG_MEMORY_POOL="1"
    )
    torch.cuda.set_device(rank)
    init_distributed_environment(
        world_size=world,
        rank=rank,
        local_rank=rank,
        distributed_init_method=init_method,
        backend="nccl",
    )
    publish_build_topology(tp_size=world, attn_cp_size=1, world_rank=rank)
    initialize_model_parallel()

    def index_values(locs, layer_id, generation):
        dims = torch.arange(128, device=locs.device)
        state = locs[:, None] * 1103515245 + dims * 12345 + layer_id * 789 + generation
        state = ((state ^ (state >> 13)) * 1274126177) & 0xFFFFFFFF
        return act_quant(((state % 65535).float() / 16384 - 2).bfloat16(), 128)

    envs.SGLANG_DSA_FUSE_TOPK.set(True)
    envs.SGLANG_OPT_USE_TOPK_V2.set(False)
    envs.SGLANG_DSA_TOPK_FLASHINFER_DETERMINISTIC.set(True)
    cases = [
        ("flashmla_sparse", False),
        ("flashmla_sparse", True),
        ("flashmla_kv", True),
    ]
    if torch.cuda.get_device_capability()[0] == 10:
        cases.append(("trtllm", False))
    # Consecutive shared layers exercise two-slot prefetch progression. Use
    # varied keys to avoid the byte fixture's tied scores at long contexts.
    with patch.dict(
        globals(),
        _INDEXER_SKIP_TOPK=[False, True, True, False, True],
        _index_values=index_values,
    ):
        for prefill_impl, packed in cases:
            for topk_backend in ("sgl-kernel", "flashinfer"):
                check = _IndexerCacheCheck(
                    get_parallel().attn_tp_group,
                    packed,
                    raw_fp8=prefill_impl == "trtllm",
                    prefix_contexts=16,
                )
                prefix = list(range(world, world + 32))
                check.seed(prefix, generation=1)
                # A shared prefix across two requests, then a prefix-free request.
                for generation, requests in (
                    (
                        2,
                        [
                            (prefix + list(range(40, 45)), 2048, 2305),
                            (prefix + [48], 2048, 2065),
                        ],
                    ),
                    (3, [([20 * world + world - 1, 21 * world], 0, 73)]),
                ):
                    _check_attention_batch(
                        check,
                        requests,
                        generation=generation,
                        prefill_impl=prefill_impl,
                        topk_backend=topk_backend,
                    )
                if rank == 0:
                    layout = "packed FP8" if packed else str(check.pool.dtype)
                    print(
                        f"{prefill_impl} ({layout}, {topk_backend}): "
                        "exact sharded attention match",
                        flush=True,
                    )
                del check
    destroy_model_parallel()
    destroy_distributed_environment()


class TestShardedDSAAttention(unittest.TestCase):
    def test_four_gpu_matches_unsharded_full_and_shared_layers(self):
        if torch.cuda.device_count() < 4:
            self.skipTest("requires four CUDA GPUs")
        if torch.cuda.get_device_capability()[0] not in (9, 10):
            self.skipTest("FlashMLA sparse requires Hopper or Blackwell")
        with tempfile.TemporaryDirectory(prefix="sglang-dsa-shard-") as tmp:
            mp.spawn(
                _run_attention,
                args=(4, f"file://{tmp}/init"),
                nprocs=4,
                join=True,
            )


if __name__ == "__main__":
    unittest.main()
