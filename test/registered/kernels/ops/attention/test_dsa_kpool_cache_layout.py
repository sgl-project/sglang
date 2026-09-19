"""Cache-layout regressions for the DSA kpool FP8 index cache.

The index K-cache stores 64 token rows of 128 bytes per page. When the AITER
preshuffle paged-MQA path is active, writers and gatherers must swizzle each
page into 16x16 tiles; otherwise the cache stays linear. These tests exercise
the public gather and production writer against an independent address oracle
in both layouts.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.dsa import kpool_fp8_index
from sglang.srt.layers.attention.dsa.kpool_fp8_index import (
    INDEX_HEAD_DIM,
    gather_index_k_scale_prefix_into,
    kpool_softmax_rotate_write_cache,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")
# backend-specific: AITER paged MQA consumes the HIP preshuffle cache layout.
register_amd_ci(est_time=25, stage="jit-kernel-unit", runner_config="amd")

PAGE_SIZE = 64
SLOTS_PER_PAGE = 64
PRESHUFFLE_TILE = 16
S_OFFSET_NBYTES_IN_PAGE = SLOTS_PER_PAGE * INDEX_HEAD_DIM
PAGE_BYTES = S_OFFSET_NBYTES_IN_PAGE + SLOTS_PER_PAGE * 4


def _oracle_index_offset(page, slot, col, tile):
    if tile:
        token_tile_id, token_in_tile = divmod(slot, tile)
        col_tile_id, col_in_tile = divmod(col, tile)
        return (
            page * PAGE_BYTES
            + token_tile_id * (tile * INDEX_HEAD_DIM)
            + col_tile_id * (tile * tile)
            + token_in_tile * tile
            + col_in_tile
        )
    return page * PAGE_BYTES + slot * INDEX_HEAD_DIM + col


def _oracle_scale_offset(page, slot):
    return page * (PAGE_BYTES // 4) + S_OFFSET_NBYTES_IN_PAGE // 4 + slot


def _fill_cache(buf, pages, tile):
    buf.zero_()
    u8 = buf.view(torch.uint8).reshape(-1)
    f32 = buf.view(torch.float32).reshape(-1)
    offsets = []
    values = []
    for page in pages:
        for slot in range(SLOTS_PER_PAGE):
            f32[_oracle_scale_offset(page, slot)] = page * 10.0 + slot
            for col in range(INDEX_HEAD_DIM):
                offsets.append(_oracle_index_offset(page, slot, col, tile))
                values.append((page * SLOTS_PER_PAGE * 7 + slot * 7 + col) % 251)
    u8[torch.tensor(offsets, dtype=torch.int64, device=buf.device)] = torch.tensor(
        values, dtype=torch.uint8, device=buf.device
    )


def _pool() -> SimpleNamespace:
    return SimpleNamespace(
        page_size=PAGE_SIZE,
        index_head_dim=INDEX_HEAD_DIM,
        slots_per_page=SLOTS_PER_PAGE,
        index_kpool=4,
        quant_block_size=128,
    )


@unittest.skipUnless(torch.cuda.is_available(), "Test requires CUDA")
class TestDsaKpoolCacheLayout(CustomTestCase):
    def _run_gather(self, tile):
        pages = [0, 1, 7]
        buf = torch.zeros(
            (max(pages) + 1, PAGE_BYTES), dtype=torch.uint8, device="cuda"
        )
        _fill_cache(buf, pages, tile)
        torch.cuda.synchronize()

        seq_len = len(pages) * PAGE_SIZE
        page_indices = torch.tensor(pages, dtype=torch.int32, device="cuda")
        k_out = torch.zeros((seq_len, INDEX_HEAD_DIM), dtype=torch.uint8, device="cuda")
        scale_out = torch.zeros((seq_len,), dtype=torch.float32, device="cuda")
        with patch.object(
            kpool_fp8_index,
            "aiter_can_use_preshuffle_paged_mqa",
            return_value=(tile != 0),
            create=True,
        ):
            gather_index_k_scale_prefix_into(
                _pool(),
                buf,
                page_indices,
                seq_len,
                k_out,
                scale_out,
            )
        torch.cuda.synchronize()

        flat = buf.view(torch.uint8).reshape(-1)
        offsets = torch.tensor(
            [
                _oracle_index_offset(page, slot, col, tile)
                for page in pages
                for slot in range(PAGE_SIZE)
                for col in range(INDEX_HEAD_DIM)
            ],
            dtype=torch.int64,
            device="cuda",
        )
        torch.testing.assert_close(
            k_out,
            flat[offsets].reshape(seq_len, INDEX_HEAD_DIM),
            atol=0,
            rtol=0,
        )
        expected_scale = torch.tensor(
            [page * 10.0 + slot for page in pages for slot in range(PAGE_SIZE)],
            dtype=torch.float32,
            device="cuda",
        )
        torch.testing.assert_close(scale_out, expected_scale, atol=0, rtol=0)

    def test_gather_linear_and_preshuffle(self):
        for tile in (0, PRESHUFFLE_TILE):
            with self.subTest(tile=tile):
                self._run_gather(tile)

    def test_writer_cache_layout(self):
        torch.manual_seed(42)
        pages = [0, 1, 7]
        for tile in (0, PRESHUFFLE_TILE):
            with self.subTest(tile=tile):
                buf = torch.zeros(
                    (max(pages) + 1, PAGE_BYTES), dtype=torch.uint8, device="cuda"
                )
                pool = _pool()
                num_rows = 3
                slot_k = torch.randn(
                    (num_rows, 4, INDEX_HEAD_DIM),
                    dtype=torch.bfloat16,
                    device="cuda",
                )
                slot_score = torch.randn_like(slot_k).abs() + 1.0
                ape = torch.randn(
                    (4, INDEX_HEAD_DIM), dtype=torch.float32, device="cuda"
                )
                loc = torch.tensor([30, 95, 500], dtype=torch.int64, device="cuda")
                with patch.object(
                    kpool_fp8_index,
                    "aiter_can_use_preshuffle_paged_mqa",
                    return_value=(tile != 0),
                    create=True,
                ):
                    compressed_k, compressed_scale = kpool_softmax_rotate_write_cache(
                        pool,
                        buf,
                        slot_k,
                        slot_score,
                        ape,
                        loc,
                        return_compressed=True,
                        write_cache=True,
                    )
                torch.cuda.synchronize()

                # Build the whole expected cache independently, including the
                # untouched slots. Returned quantized rows isolate layout from
                # the compression/quantization numerical contract.
                expected = torch.zeros_like(buf)
                locations = [divmod(int(l), SLOTS_PER_PAGE) for l in loc.tolist()]
                k_offsets = torch.tensor(
                    [
                        _oracle_index_offset(page, slot, col, tile)
                        for page, slot in locations
                        for col in range(INDEX_HEAD_DIM)
                    ],
                    dtype=torch.int64,
                    device="cuda",
                )
                s_offsets = torch.tensor(
                    [_oracle_scale_offset(page, slot) for page, slot in locations],
                    dtype=torch.int64,
                    device="cuda",
                )
                expected.flatten()[k_offsets] = compressed_k.view(torch.uint8).flatten()
                expected.view(torch.float32).flatten()[s_offsets] = (
                    compressed_scale.flatten()
                )
                torch.testing.assert_close(buf, expected, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
