"""Hy4-Flash IndexPool kernels take batch-shaped arguments at run time: a new prompt
length, page-table width or row-chunk offset must not compile another variant of a
kernel, since each compile stalls the engine for seconds while serving."""

import unittest

import torch
import triton

from sglang.kernels.ops.attention.dsa import hy4_flash_indexpool as indexpool
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=40, stage="base-b-kernel-unit", runner_config="1-gpu-large")

PAGE = 64
HEADS = 16
TOPK = 2048
WINDOW = 512
CACHE_PAGES = 64
MROPE_SECTION = (16, 8, 8)
# (rows, page-table width, picked groups per row, MRoPE token count, chunk start).
# The first two compile each launcher; the rest vary every batch-shaped argument
# across a multiple of 16, 1 and neither. Rows under 64 take the per-row logits kernel.
WARMUP = [(128, 16, 511, 128, 0), (16, 16, 511, 16, 0)]
VARIANTS = [
    (127, 17, 128, 127, 1),
    (131, 1, 256, 133, 0),
    (64, 33, 511, 65, 1),
    (7, 17, 128, 7, 1),
    (5, 1, 384, 6, 0),
    (1, 3, 511, 1, 1),
]


def compiled_variants(kernel):
    caches = getattr(kernel, "device_caches", None)
    if caches is None:
        return None
    return sum(len(v[0]) if isinstance(v, tuple) else len(v) for v in caches.values())


def index_cache(device) -> torch.Tensor:
    """Random pooled FP8 keys (no NaN encodings) with positive scales."""
    buf = torch.randint(
        0, 255, (CACHE_PAGES, PAGE * 132), dtype=torch.uint8, device=device
    )
    keys = buf[:, : PAGE * 128]
    keys[(keys & 0x7F) == 0x7F] = 0
    buf[:, PAGE * 128 :].view(torch.float32)[:] = (
        torch.rand(CACHE_PAGES, PAGE, device=device) * 0.1 + 0.01
    )
    return buf


class IndexPoolBatch:
    """One request whose last `rows` tokens are scored, plus a decode step of
    `rows` requests sharing the same page-table width."""

    def __init__(self, rows, width, picked, mrope_tokens, start, device):
        tokens = width * PAGE
        n = rows + start
        self.rows, self.start = rows, start
        self.pages = torch.randint(
            0, CACHE_PAGES, (1, width), dtype=torch.int32, device=device
        )
        # Logits get row-chunk views; the other launchers get whole tensors.
        self.positions = (torch.arange(n, device=device) + tokens - n).clamp_min(0)
        self.row_to_batch = torch.zeros(n, dtype=torch.int64, device=device)
        self.scored_positions = self.positions[start:].clone()
        self.scored_row_to_batch = self.row_to_batch[start:].clone()
        self.q = (torch.randn(n, HEADS, 128, device=device) * 20).to(
            torch.float8_e4m3fn
        )
        self.w = torch.rand(n, HEADS, device=device)
        self.num_groups = triton.cdiv(max(tokens // 4, 1), 128) * 128
        self.picked = torch.randint(0, self.num_groups, (rows, picked), device=device)
        self.values = torch.rand(rows, picked, device=device)
        xyz = torch.arange(mrope_tokens, device=device).expand(3, -1).contiguous()
        self.mrope_positions = xyz[:, :rows].t()
        self.query = torch.randn(rows, HEADS * 128, device=device, dtype=torch.bfloat16)
        self.decode_pages = torch.randint(
            0, CACHE_PAGES, (rows, width), dtype=torch.int32, device=device
        )
        self.seq_lens = torch.randint(
            1, tokens + 1, (rows,), dtype=torch.int32, device=device
        )
        self.out_loc = torch.randperm(CACHE_PAGES * PAGE, device=device)[:rows]


class TestHy4FlashIndexPoolVariants(CustomTestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        torch.manual_seed(0)
        self.device = "cuda"
        self.buf = index_cache(self.device)
        self.inv_freq = torch.rand(32, device=self.device)
        self.raw_keys = torch.zeros(
            CACHE_PAGES * PAGE, 128, dtype=torch.bfloat16, device=self.device
        )
        self.wv_t = torch.randn(128, 128, device=self.device) * 0.05
        self.wg_t = torch.randn(128, 128, device=self.device) * 0.05
        self.ape = torch.randn(4, 128, device=self.device)
        self.key_norm = torch.ones(128, device=self.device)

    def run_batch(self, b: IndexPoolBatch):
        chunk = slice(b.start, b.start + b.rows)
        indexpool.indexpool_logits(
            b.q[chunk],
            b.w[chunk],
            b.pages,
            b.row_to_batch[chunk],
            b.positions[chunk],
            self.buf,
            PAGE,
            b.num_groups,
        )
        positions, row_to_batch = b.scored_positions, b.scored_row_to_batch
        indexpool.indexpool_expand(
            b.picked, b.values, positions, b.pages, row_to_batch, None, TOPK
        )
        indexpool.window_indices(positions, b.pages, row_to_batch, None, WINDOW, TOPK)
        indexpool.indexpool_query(
            b.query, b.mrope_positions, b.w[chunk], self.inv_freq, HEADS, MROPE_SECTION
        )
        indexpool.indexpool_decode_update(
            b.query[:, :128],
            self.key_norm.to(torch.bfloat16),
            1e-6,
            b.out_loc,
            b.seq_lens,
            b.decode_pages,
            self.raw_keys,
            self.wv_t,
            self.wg_t,
            self.ape,
            self.key_norm,
            1e-6,
            self.inv_freq,
            self.buf,
            PAGE,
        )

    def test_batch_shapes_share_one_variant(self):
        """After one batch per launcher, batches whose row count, page-table width,
        picked-group count, MRoPE token count or chunk start differ (each a multiple
        of 16, 1 or neither) compile nothing new."""
        kernels = [
            indexpool._query_kernel,
            indexpool._decode_update_kernel,
            indexpool._logits_kernel,
            indexpool._logits_rows_kernel,
            indexpool._expand_kernel,
            indexpool._window_kernel,
        ]
        if compiled_variants(kernels[0]) is None:
            self.skipTest("Triton does not expose device_caches")
        for shape in WARMUP:
            self.run_batch(IndexPoolBatch(*shape, self.device))
        torch.cuda.synchronize()
        warm = [compiled_variants(k) for k in kernels]
        for shape in VARIANTS:
            self.run_batch(IndexPoolBatch(*shape, self.device))
        torch.cuda.synchronize()
        grown = {
            k.fn.__name__: compiled_variants(k) - before
            for k, before in zip(kernels, warm)
            if compiled_variants(k) != before
        }
        self.assertEqual(grown, {})


if __name__ == "__main__":
    unittest.main()
