"""Layout coverage for the k-pool index-K cache under AITER preshuffle.

`kpool_fp8_index` writes this buffer from four kernels and reads it back from a
fifth, and AITER's preshuffle paged-MQA kernel reads it as well. A disagreement
between any two of them does not raise -- it silently returns the wrong top-k --
so the offsets are pinned here directly.

`_kpool_cache_k_offsets` must also agree with `_set_k_and_s_triton_kernel` in
`kernels/ops/attention/dsa/index_buf_accessor.py`, which lays out the non-pooled
indexer cache for the same AITER kernel.
"""

import unittest
from types import SimpleNamespace

import torch
import triton
import triton.language as tl

from sglang.srt.layers.attention.dsa.kpool_fp8_index import (
    INDEX_HEAD_DIM,
    _kpool_cache_k_offsets,
    _preshuffle_tile,
    gather_index_k_scale_prefix_into,
)
from sglang.srt.layers.attention.dsa.utils import INDEXER_K_CACHE_PRESHUFFLE_TILE
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd-mi35x")

PAGE_SIZE = 64
BUF_NUMEL_PER_PAGE = PAGE_SIZE * INDEX_HEAD_DIM + PAGE_SIZE * 4


@triton.jit
def _materialize_offsets_kernel(
    out_ptr,
    BUF_NUMEL_PER_PAGE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PRESHUFFLE_TILE: tl.constexpr,
):
    """One program per slot; writes that slot's HEAD_DIM cache offsets."""
    slot = tl.program_id(0)
    cols = tl.arange(0, HEAD_DIM)
    offsets = _kpool_cache_k_offsets(
        0, slot, cols, BUF_NUMEL_PER_PAGE, HEAD_DIM, PRESHUFFLE_TILE
    )
    tl.store(out_ptr + slot * HEAD_DIM + cols, offsets)


def _reference_offsets(tile: int) -> torch.Tensor:
    """The layout `index_buf_accessor._set_k_and_s_triton_kernel` writes."""
    slot = torch.arange(PAGE_SIZE).unsqueeze(1)
    cols = torch.arange(INDEX_HEAD_DIM).unsqueeze(0)
    if not tile:
        return (slot * INDEX_HEAD_DIM + cols).to(torch.int32)
    return (
        (slot // tile) * (tile * INDEX_HEAD_DIM)
        + (cols // tile) * (tile * tile)
        + (slot % tile) * tile
        + (cols % tile)
    ).to(torch.int32)


@unittest.skipUnless(torch.cuda.is_available(), "Test requires a GPU")
class TestKpoolPreshuffledCacheLayout(CustomTestCase):
    def _materialize(self, tile: int) -> torch.Tensor:
        out = torch.empty(PAGE_SIZE, INDEX_HEAD_DIM, dtype=torch.int32, device="cuda")
        _materialize_offsets_kernel[(PAGE_SIZE,)](
            out,
            BUF_NUMEL_PER_PAGE=BUF_NUMEL_PER_PAGE,
            HEAD_DIM=INDEX_HEAD_DIM,
            PRESHUFFLE_TILE=tile,
        )
        return out.cpu()

    def test_offsets_match_the_non_pooled_layout(self):
        for tile in (0, INDEXER_K_CACHE_PRESHUFFLE_TILE):
            with self.subTest(tile=tile):
                torch.testing.assert_close(
                    self._materialize(tile), _reference_offsets(tile), atol=0, rtol=0
                )

    def test_preshuffle_only_permutes_the_k_region(self):
        # Every (slot, col) must still land on its own byte, inside the K half of
        # the page -- the scale half sits immediately after it and must not be
        # clobbered. A formula slip usually shows up here as a collision.
        tile = INDEXER_K_CACHE_PRESHUFFLE_TILE
        shuffled = self._materialize(tile).flatten()
        self.assertEqual(shuffled.min().item(), 0)
        self.assertEqual(shuffled.max().item(), PAGE_SIZE * INDEX_HEAD_DIM - 1)
        self.assertEqual(shuffled.unique().numel(), PAGE_SIZE * INDEX_HEAD_DIM)

    def test_gather_reads_back_what_the_layout_wrote(self):
        # The reader is one of the five sites that share `_kpool_cache_k_offsets`,
        # so a round trip against an independently computed layout catches the
        # reader and writer drifting apart. Follow the runtime tile rather than
        # pinning 16: without AITER's preshuffle kernel the reader stays
        # row-major, and this has to hold there too.
        tile = _preshuffle_tile()
        torch.manual_seed(0)
        num_pages, seq_len = 2, PAGE_SIZE + 5
        buf = torch.zeros(num_pages, BUF_NUMEL_PER_PAGE, dtype=torch.uint8).cuda()
        page_indices = torch.tensor([0, 1], dtype=torch.int32, device="cuda")

        k = torch.randint(
            1, 255, (seq_len, INDEX_HEAD_DIM), dtype=torch.uint8, device="cuda"
        )
        scale = torch.randn(seq_len, dtype=torch.float32, device="cuda")

        layout = _reference_offsets(tile).cuda()
        for token in range(seq_len):
            page, slot = divmod(token, PAGE_SIZE)
            buf[page].scatter_(0, layout[slot].to(torch.int64), k[token])
        buf_f32 = buf.view(torch.float32)
        scale_base = PAGE_SIZE * INDEX_HEAD_DIM // 4
        for token in range(seq_len):
            page, slot = divmod(token, PAGE_SIZE)
            buf_f32[page, scale_base + slot] = scale[token]

        k_out = torch.zeros_like(k)
        scale_out = torch.zeros_like(scale)
        gather_index_k_scale_prefix_into(
            SimpleNamespace(page_size=PAGE_SIZE),
            buf,
            page_indices,
            seq_len,
            k_out,
            scale_out,
        )
        torch.testing.assert_close(k_out, k, atol=0, rtol=0)
        torch.testing.assert_close(scale_out, scale, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
