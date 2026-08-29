import unittest
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention.dsa import index_buf_accessor
from sglang.srt.mem_cache.dsa_cache_layer_split import (
    LayerSplitDSATokenToKVPool,
    LayerSplitIndexKeyCache,
)
from sglang.srt.mem_cache.index_key_cache import IndexKeyCache
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool
from sglang.srt.speculative.eagle_worker_common import (
    duplicate_prefix_tail_to_draft_branches,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=20, suite="stage-b-test-1-gpu-small-amd-mi35x")

PAGE_SIZE = 64
INDEX_HEAD_DIM = 128
PAYLOAD_BYTES = INDEX_HEAD_DIM + 4


def _logical_offsets(loc: int) -> torch.Tensor:
    page, token = divmod(loc, PAGE_SIZE)
    page_base = page * PAGE_SIZE * PAYLOAD_BYTES
    cols = torch.arange(INDEX_HEAD_DIM, dtype=torch.int64)
    if index_buf_accessor._use_aiter_preshuffle:
        tile = 16
        k_offsets = (
            page_base
            + (token // tile) * tile * INDEX_HEAD_DIM
            + (cols // tile) * tile * tile
            + (token % tile) * tile
            + cols % tile
        )
    else:
        k_offsets = page_base + token * INDEX_HEAD_DIM + cols
    scale_offsets = (
        page_base
        + PAGE_SIZE * INDEX_HEAD_DIM
        + token * 4
        + torch.arange(4, dtype=torch.int64)
    )
    return torch.cat((k_offsets, scale_offsets))


def _expected_index_move(
    before: torch.Tensor, tgt_loc: torch.Tensor, src_loc: torch.Tensor
) -> torch.Tensor:
    expected = before.clone()
    before_flat = before.flatten()
    expected_flat = expected.flatten()
    for tgt, src in zip(tgt_loc.cpu().tolist(), src_loc.cpu().tolist()):
        expected_flat[_logical_offsets(tgt)] = before_flat[_logical_offsets(src)]
    return expected


def _make_pool(
    pool_cls=DSATokenToKVPool,
    *,
    size=PAGE_SIZE * 7,
    layers=2,
    kv_dim=4,
    kv_dtype=torch.float32,
):
    pool = object.__new__(pool_cls)
    pool.size = size
    pool.page_size = PAGE_SIZE
    pool.index_head_dim = INDEX_HEAD_DIM
    pool.layer_num = layers
    pool.device = torch.device("cuda")
    num_pages = (size + PAGE_SIZE + 1) // PAGE_SIZE
    pool.kv_buffer = [
        (
            torch.arange(
                (size + PAGE_SIZE) * kv_dim,
                dtype=torch.int64,
                device="cuda",
            )
            + layer * 17
        )
        .remainder(251)
        .to(kv_dtype)
        .view(size + PAGE_SIZE, 1, kv_dim)
        for layer in range(layers)
    ]
    cache_cls = (
        LayerSplitIndexKeyCache
        if pool_cls is LayerSplitDSATokenToKVPool
        else IndexKeyCache
    )
    index_key_cache = object.__new__(cache_cls)
    index_key_cache.pool = pool
    index_key_cache.buffer = [
        (
            torch.arange(
                num_pages * PAGE_SIZE * PAYLOAD_BYTES,
                dtype=torch.int64,
                device="cuda",
            ).view(num_pages, PAGE_SIZE * PAYLOAD_BYTES)
            + layer * 17
        )
        .remainder(251)
        .to(torch.uint8)
        for layer in range(layers)
    ]
    pool.index_key_cache = index_key_cache
    pool._init_dsa_move_metadata()
    return pool


@unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA/ROCm")
class TestDSAIndexCacheMove(CustomTestCase):
    def _assert_pool_move(self, pool, tgt_loc, src_loc):
        kv_before = [buf.clone() for buf in pool.kv_buffer]
        index_before = [buf.clone() for buf in pool.index_k_with_scale_buffer]

        pool.move_kv_cache(tgt_loc, src_loc)
        torch.cuda.synchronize()

        for got, before in zip(pool.kv_buffer, kv_before):
            if got.shape[0] == 0:
                torch.testing.assert_close(got, before, rtol=0, atol=0)
                continue
            expected = before.clone()
            expected[tgt_loc] = before[src_loc]
            torch.testing.assert_close(got, expected, rtol=0, atol=0)
        for got, before in zip(pool.index_k_with_scale_buffer, index_before):
            if got.shape[0] == 0:
                torch.testing.assert_close(got, before, rtol=0, atol=0)
                continue
            expected = _expected_index_move(before.cpu(), tgt_loc, src_loc).to(
                got.device
            )
            torch.testing.assert_close(got, expected, rtol=0, atol=0)

    def test_cross_page_overlap_and_identity(self):
        pool = _make_pool()
        src_loc = torch.tensor([70, 141, 260, 333], device="cuda")
        tgt_loc = torch.tensor([195, 260, 141, 333], device="cuda")
        self.assertGreater(
            int(src_loc.max()), pool.index_k_with_scale_buffer[0].shape[0]
        )
        self._assert_pool_move(pool, tgt_loc, src_loc)

    def test_empty_move(self):
        pool = _make_pool()
        kv_before = [buf.clone() for buf in pool.kv_buffer]
        index_before = [buf.clone() for buf in pool.index_k_with_scale_buffer]
        empty = torch.empty(0, dtype=torch.int64, device="cuda")
        pool.move_kv_cache(empty, empty)
        for got, before in zip(pool.kv_buffer, kv_before):
            torch.testing.assert_close(got, before, rtol=0, atol=0)
        for got, before in zip(pool.index_k_with_scale_buffer, index_before):
            torch.testing.assert_close(got, before, rtol=0, atol=0)

    def test_page64_tree_prefix_tail_duplication(self):
        pool = _make_pool(size=PAGE_SIZE * 5, layers=1)
        rows = torch.arange(
            pool.size + PAGE_SIZE, dtype=torch.int64, device="cuda"
        ).unsqueeze(0)
        kv_before = pool.kv_buffer[0].clone()
        index_before = pool.index_k_with_scale_buffer[0].clone()

        duplicate_prefix_tail_to_draft_branches(
            pool,
            rows,
            prefix_base=torch.tensor([64], dtype=torch.int64, device="cuda"),
            last_page=torch.tensor([6], dtype=torch.int64, device="cuda"),
            num_new_pages=torch.tensor([1], dtype=torch.int64, device="cuda"),
            topk=2,
            page_size=PAGE_SIZE,
        )
        torch.cuda.synchronize()

        src_loc = torch.arange(64, 70, dtype=torch.int64, device="cuda")
        tgt_loc = torch.arange(128, 134, dtype=torch.int64, device="cuda")
        expected_kv = kv_before.clone()
        expected_kv[tgt_loc] = kv_before[src_loc]
        expected_index = _expected_index_move(index_before.cpu(), tgt_loc, src_loc).to(
            "cuda"
        )
        torch.testing.assert_close(pool.kv_buffer[0], expected_kv, rtol=0, atol=0)
        torch.testing.assert_close(
            pool.index_k_with_scale_buffer[0], expected_index, rtol=0, atol=0
        )

    def test_layer_split_skips_non_owned_buffers(self):
        pool = _make_pool(LayerSplitDSATokenToKVPool, layers=2)
        pool.kv_buffer[1] = pool.kv_buffer[1][:0]
        pool.index_k_with_scale_buffer[1] = pool.index_k_with_scale_buffer[1][:0]
        pool._init_dsa_move_metadata()
        src_loc = torch.tensor([77, 202], dtype=torch.int64, device="cuda")
        tgt_loc = torch.tensor([143, 271], dtype=torch.int64, device="cuda")
        self._assert_pool_move(pool, tgt_loc, src_loc)

    def test_78_layer_overlap(self):
        pool = _make_pool(layers=78)
        src_loc = torch.tensor([70, 141, 260, 333], device="cuda")
        tgt_loc = torch.tensor([195, 260, 141, 333], device="cuda")
        self._assert_pool_move(pool, tgt_loc, src_loc)
        self._assert_pool_move(pool, src_loc, tgt_loc)

    def test_duplicate_identity_padding(self):
        pool = _make_pool(layers=2)
        src_loc = torch.tensor([70, 141, 0, 0], device="cuda")
        tgt_loc = torch.tensor([195, 260, 0, 0], device="cuda")
        self._assert_pool_move(pool, tgt_loc, src_loc)

    def test_page1_row_major_int32_noncontiguous_locations(self):
        with (
            patch.object(index_buf_accessor, "_use_aiter_preshuffle", False),
            patch(f"{__name__}.PAGE_SIZE", 1),
        ):
            pool = _make_pool(size=32, layers=2)
            src_base = torch.tensor(
                [3, 99, 7, 99, 11, 99], dtype=torch.int32, device="cuda"
            )
            tgt_base = torch.tensor(
                [18, 99, 22, 99, 26, 99], dtype=torch.int32, device="cuda"
            )
            src_loc = src_base[::2]
            tgt_loc = tgt_base[::2]
            self.assertFalse(src_loc.is_contiguous())
            self.assertFalse(tgt_loc.is_contiguous())
            self._assert_pool_move(pool, tgt_loc, src_loc)

    def test_mismatched_location_lengths_fail(self):
        pool = _make_pool()
        src_loc = torch.tensor([70, 141], device="cuda")
        tgt_loc = torch.tensor([195], device="cuda")
        with self.assertRaises(AssertionError):
            pool.move_kv_cache(tgt_loc, src_loc)

    def test_uneven_layer_split_ownership(self):
        pool = _make_pool(LayerSplitDSATokenToKVPool, layers=5)
        for layer in (1, 3):
            pool.kv_buffer[layer] = pool.kv_buffer[layer][:0]
            pool.index_k_with_scale_buffer[layer] = pool.index_k_with_scale_buffer[
                layer
            ][:0]
        pool._init_dsa_move_metadata()
        src_loc = torch.tensor([77, 202], dtype=torch.int64, device="cuda")
        tgt_loc = torch.tensor([143, 271], dtype=torch.int64, device="cuda")
        self._assert_pool_move(pool, tgt_loc, src_loc)

    def test_real_width_large_slot_move(self):
        pool = _make_pool(
            size=PAGE_SIZE * 20,
            layers=8,
            kv_dim=656,
            kv_dtype=torch.uint8,
        )
        src_loc = torch.arange(64, 576, dtype=torch.int64, device="cuda")
        tgt_loc = torch.arange(704, 1216, dtype=torch.int64, device="cuda")
        self._assert_pool_move(pool, tgt_loc, src_loc)

    def test_concurrent_streams(self):
        pool = _make_pool(layers=78)
        src_1 = torch.tensor([70, 141], device="cuda")
        tgt_1 = torch.tensor([195, 260], device="cuda")
        src_2 = torch.tensor([80, 151], device="cuda")
        tgt_2 = torch.tensor([205, 270], device="cuda")
        kv_before = [buf.clone() for buf in pool.kv_buffer]
        index_before = [buf.clone() for buf in pool.index_k_with_scale_buffer]

        stream_1 = torch.cuda.Stream()
        stream_2 = torch.cuda.Stream()
        with torch.cuda.stream(stream_1):
            pool.move_kv_cache(tgt_1, src_1)
        with torch.cuda.stream(stream_2):
            pool.move_kv_cache(tgt_2, src_2)
        torch.cuda.synchronize()

        for got, before in zip(pool.kv_buffer, kv_before):
            expected = before.clone()
            expected[tgt_1] = before[src_1]
            expected[tgt_2] = before[src_2]
            torch.testing.assert_close(got, expected, rtol=0, atol=0)
        for got, before in zip(pool.index_k_with_scale_buffer, index_before):
            expected = _expected_index_move(before.cpu(), tgt_1, src_1)
            expected = _expected_index_move(expected, tgt_2, src_2).to("cuda")
            torch.testing.assert_close(got, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
