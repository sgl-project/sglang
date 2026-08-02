import unittest

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsv4.indexer import (
    FP8_DTYPE,
    _indexer_logits_chunk_pages,
    fp8_paged_mqa_logits_torch_sm120,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

HEAD_DIM = 128
BLOCK_SIZE = 64
# Byte layout of one KV page row: block_size * head_dim value bytes followed by
# block_size float32 scales, all addressed as FP8 elements.
SCALE_OFFSET = BLOCK_SIZE * HEAD_DIM
PAGE_BYTES = BLOCK_SIZE * (HEAD_DIM + 4)


def finite_fp8_bytes(shape, generator):
    """Random float8_e4m3fn bit patterns, both signs, with the NaN codes removed."""
    raw = torch.randint(0, 256, shape, generator=generator, dtype=torch.uint8)
    # 0x7f and 0xff are the only NaN encodings in float8_e4m3fn.
    return torch.where(raw % 0x80 == 0x7F, torch.tensor(0x38, dtype=torch.uint8), raw)


def make_inputs(batch_size, num_heads, seq_lens, max_seq_len, seed=0):
    """Build a valid CPU FP8 paged-MQA input set with finite values only."""
    generator = torch.Generator().manual_seed(seed)
    num_pages = (max_seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE * batch_size + 1

    raw = torch.empty(num_pages, PAGE_BYTES, dtype=torch.uint8)
    raw[:, :SCALE_OFFSET] = finite_fp8_bytes(
        (num_pages, SCALE_OFFSET), generator=generator
    )
    scales = 0.5 + torch.rand(num_pages, BLOCK_SIZE, generator=generator)
    raw[:, SCALE_OFFSET:] = scales.view(dtype=torch.uint8)
    kvcache_fp8 = raw.view(dtype=FP8_DTYPE).view(num_pages, BLOCK_SIZE, 1, HEAD_DIM + 4)

    q_fp8 = finite_fp8_bytes(
        (batch_size, 1, num_heads, HEAD_DIM), generator=generator
    ).view(dtype=FP8_DTYPE)

    weight = torch.randn(batch_size, num_heads, generator=generator)
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32)
    max_pages = (max_seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    page_table = torch.randint(
        0, num_pages, (batch_size, max_pages), generator=generator, dtype=torch.int32
    )
    return q_fp8, kvcache_fp8, weight, seq_lens_t, page_table


def run(inputs, max_seq_len, chunk_positions):
    q_fp8, kvcache_fp8, weight, seq_lens, page_table = inputs
    with envs.SGLANG_DSV4_INDEXER_LOGITS_SEQ_CHUNK.override(chunk_positions):
        return fp8_paged_mqa_logits_torch_sm120(
            q_fp8,
            kvcache_fp8,
            weight,
            seq_lens,
            page_table,
            None,
            max_seq_len,
            clean_logits=False,
        )


class TestIndexerLogitsChunkPages(unittest.TestCase):
    def test_disabled_returns_full_span(self):
        with envs.SGLANG_DSV4_INDEXER_LOGITS_SEQ_CHUNK.override(0):
            self.assertEqual(_indexer_logits_chunk_pages(BLOCK_SIZE, 37), 37)

    def test_rounds_down_to_whole_pages(self):
        with envs.SGLANG_DSV4_INDEXER_LOGITS_SEQ_CHUNK.override(191):
            self.assertEqual(_indexer_logits_chunk_pages(BLOCK_SIZE, 37), 2)
        with envs.SGLANG_DSV4_INDEXER_LOGITS_SEQ_CHUNK.override(192):
            self.assertEqual(_indexer_logits_chunk_pages(BLOCK_SIZE, 37), 3)

    def test_sub_page_chunk_yields_one_page(self):
        with envs.SGLANG_DSV4_INDEXER_LOGITS_SEQ_CHUNK.override(1):
            self.assertEqual(_indexer_logits_chunk_pages(BLOCK_SIZE, 37), 1)

    def test_never_exceeds_available_pages(self):
        with envs.SGLANG_DSV4_INDEXER_LOGITS_SEQ_CHUNK.override(8192):
            self.assertEqual(_indexer_logits_chunk_pages(BLOCK_SIZE, 3), 3)


class TestIndexerLogitsSeqChunk(unittest.TestCase):
    # (batch_size, num_heads, seq_lens, max_seq_len)
    SHAPES = [
        (2, 4, [200, 130], 200),
        (1, 8, [64], 64),
        (3, 2, [321, 5, 199], 321),
        (2, 16, [500, 500], 512),
    ]
    CHUNK_POSITIONS = [1, 64, 100, 128, 192, 4096]

    def test_chunked_matches_unchunked_bitwise(self):
        for batch_size, num_heads, seq_lens, max_seq_len in self.SHAPES:
            inputs = make_inputs(batch_size, num_heads, seq_lens, max_seq_len)
            reference = run(inputs, max_seq_len, 0)
            self.assertEqual(reference.shape, (batch_size, max_seq_len))
            for chunk_positions in self.CHUNK_POSITIONS:
                with self.subTest(
                    shape=(batch_size, num_heads, max_seq_len), chunk=chunk_positions
                ):
                    got = run(inputs, max_seq_len, chunk_positions)
                    self.assertEqual(got.dtype, reference.dtype)
                    torch.testing.assert_close(got, reference, atol=0, rtol=0)

    def test_length_mask_preserved(self):
        batch_size, num_heads, seq_lens, max_seq_len = 3, 2, [321, 5, 199], 321
        inputs = make_inputs(batch_size, num_heads, seq_lens, max_seq_len)
        for chunk_positions in [0, 64, 100, 4096]:
            with self.subTest(chunk=chunk_positions):
                logits = run(inputs, max_seq_len, chunk_positions)
                for row, seq_len in enumerate(seq_lens):
                    self.assertTrue(
                        torch.isinf(logits[row, seq_len:]).all()
                        and (logits[row, seq_len:] < 0).all(),
                        f"row {row} not -inf past seq_len {seq_len}",
                    )
                    self.assertTrue(
                        torch.isfinite(logits[row, :seq_len]).all(),
                        f"row {row} has non-finite values inside seq_len {seq_len}",
                    )

    def test_tail_chunk_is_not_truncated(self):
        # max_seq_len is neither a multiple of the chunk width nor of the page
        # size, so the final chunk is the only partially copied one.
        batch_size, num_heads, max_seq_len = 2, 4, 421
        seq_lens = [421, 421]
        inputs = make_inputs(batch_size, num_heads, seq_lens, max_seq_len)
        reference = run(inputs, max_seq_len, 0)
        got = run(inputs, max_seq_len, 128)
        self.assertTrue(torch.equal(got, reference))
        self.assertTrue(torch.isfinite(reference).all())


if __name__ == "__main__":
    unittest.main()
