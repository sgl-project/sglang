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
"""Byte-exact DSA page-interleave cache tests over real NCCL on 2 and 4 GPUs.

Covers packed FP8 indexer keys and FP32 scales, BF16 and packed FP8 latent KV,
and dequantized MLA reads from the gathered scratch. Tests skip when fewer
than the required CUDA devices are visible. Run directly with:
    CUDA_VISIBLE_DEVICES=0,1,2,3 python3 \
        test/registered/unit/mem_cache/test_page_interleave_indexer_multi_gpu.py
"""

import os
import tempfile
import unittest
from types import SimpleNamespace

import torch
import torch.multiprocessing as mp

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase, publish_build_topology

register_cuda_ci(est_time=120, stage="base-c", runner_config="4-gpu-b200")

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
    def __init__(self, group, packed):
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
        self.kwargs = kwargs = dict(
            page_size=_INDEXER_PAGE_SIZE,
            kv_lora_rank=_INDEXER_KV_LORA_RANK,
            dtype=torch.float8_e4m3fn if packed else torch.bfloat16,
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
                max_prefix_tokens=8 * self.world * _INDEXER_PAGE_SIZE,
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


if __name__ == "__main__":
    unittest.main()
