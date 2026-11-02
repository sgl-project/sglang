"""MSA-v1 fused decode-plan update against the ops it replaces, without the provider."""

import unittest

import torch

from sglang.kernels.ops.attention.msa_v1.decode_plan import update_msa_v1_decode_plan
from sglang.srt.layers.attention.minimax_sparse_ops.msa_plan import (
    build_msa_v1_page_table,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")

PAGE = 128
# Queries per MSA-v1 decode indexer launch.
CHUNK = 16
TOPK = 16
POOL_ROWS = 5
LIVE_PAGES = 40
NUM_KV_HEADS = 2
SENTINEL = -7


def _pages(tokens: torch.Tensor) -> torch.Tensor:
    return torch.div(tokens, PAGE, rounding_mode="floor")


# Live page starts depend on the row, which catches row mix-ups; dead page p starts
# at -129 - 128p - row, where floor and truncation give different pages.
def _req_to_token(max_pages: int) -> torch.Tensor:
    req_to_token = torch.full(
        (POOL_ROWS, max_pages * PAGE + 4), -1, dtype=torch.int32, device="cuda"
    )
    pages = torch.arange(max_pages, dtype=torch.int32, device="cuda")[None, :]
    rows = torch.arange(POOL_ROWS, dtype=torch.int32, device="cuda")[:, None]
    req_to_token[:, : max_pages * PAGE : PAGE] = torch.where(
        pages < LIVE_PAGES,
        (rows * max_pages + pages) * PAGE + rows,
        -129 - PAGE * pages - rows,
    )
    return req_to_token


def _empty(*shape: int) -> torch.Tensor:
    return torch.full(shape, SENTINEL, dtype=torch.int32, device="cuda")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for this test.")
class TestMSAV1DecodePlanUpdate(CustomTestCase):
    def _check(
        self,
        *,
        query_length: int,
        num_reqs: int,
        max_pages: int,
        is_target_verify: bool,
        dtype: torch.dtype,
        with_flashinfer: bool = True,
    ) -> None:
        num_chunks = -(-query_length // CHUNK)
        num_query_rows = num_reqs * query_length
        offset = query_length if is_target_verify else 0
        gen = torch.Generator(device="cuda").manual_seed(num_reqs * query_length)
        # The last request pads a graph bucket: pool row 0, seq_len 1.
        req_pool_indices = torch.cat(
            (
                torch.randint(
                    1, POOL_ROWS, (num_reqs - 1,), device="cuda", generator=gen
                ),
                torch.zeros(1, dtype=torch.int64, device="cuda"),
            )
        ).to(dtype)
        # Lengths around page and top-k boundaries, then random ones.
        edges = torch.tensor(
            [1, PAGE, PAGE + 1, TOPK * PAGE, TOPK * PAGE + 1], device="cuda"
        )
        random = torch.randint(
            1, LIVE_PAGES * PAGE - offset, (num_reqs,), device="cuda", generator=gen
        )
        seq_lens = torch.cat((edges, random))[: num_reqs - 1]
        seq_lens = torch.cat(
            (seq_lens, torch.ones(1, dtype=torch.int64, device="cuda"))
        ).to(dtype)
        req_to_token = _req_to_token(max_pages)

        page_table = _empty(num_reqs, max_pages)
        final_seq_lens = _empty(num_reqs)
        chunk_seq_lens = _empty(num_chunks, num_reqs)
        query_seq_lens = _empty(num_query_rows)
        # Q8KV4 attention has no FlashInfer lengths; refresh() then passes None.
        flashinfer_seq_lens = (
            _empty(NUM_KV_HEADS, num_query_rows) if with_flashinfer else None
        )
        update_msa_v1_decode_plan(
            req_to_token=req_to_token,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            query_length=query_length,
            seq_len_offset=offset,
            page_table=page_table,
            final_seq_lens=final_seq_lens,
            chunk_seq_lens=chunk_seq_lens,
            query_seq_lens=query_seq_lens,
            flashinfer_seq_lens=flashinfer_seq_lens,
        )

        def assert_equal(actual, expected, name):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0, msg=name)

        assert_equal(
            page_table,
            build_msa_v1_page_table(
                req_to_token, req_pool_indices, max_seqlen_k=max_pages * PAGE
            ),
            "page_table",
        )
        final = seq_lens.to(torch.int32) + offset
        assert_equal(final_seq_lens, final, "final_seq_lens")
        # MSAV1DecodeIndexerPlan.plan(): chunk c runs queries [16c, min(16c + 16, Q)).
        assert_equal(
            chunk_seq_lens,
            torch.stack(
                [
                    final - (query_length - min(CHUNK * (chunk + 1), query_length))
                    for chunk in range(num_chunks)
                ]
            ),
            "chunk_seq_lens",
        )
        rows = torch.arange(num_query_rows, dtype=torch.int32, device="cuda")
        expected_query = (
            final[rows // query_length] - (query_length - 1 - rows % query_length)
        ).clamp(min=0)
        assert_equal(query_seq_lens, expected_query, "query_seq_lens")
        if flashinfer_seq_lens is None:
            return
        # FlashInferQ8K8Decode.update_lengths.
        skipped_pages = (_pages(expected_query - 1) - (TOPK - 1)).clamp(min=0)
        sparse_lens = (expected_query - skipped_pages * PAGE).clamp(min=1)
        assert_equal(
            flashinfer_seq_lens,
            sparse_lens.expand(NUM_KV_HEADS, -1),
            "flashinfer_seq_lens",
        )

    def test_decode_at_full_context_spans_eight_page_blocks(self):
        self._check(
            query_length=1,
            num_reqs=4,
            max_pages=8192,
            is_target_verify=False,
            dtype=torch.int64,
        )

    def test_three_chunk_verify_with_a_one_page_tail_block(self):
        self._check(
            query_length=33,
            num_reqs=4,
            max_pages=1025,
            is_target_verify=True,
            dtype=torch.int32,
        )

    def test_two_chunk_verify_without_flashinfer_lengths(self):
        self._check(
            query_length=20,
            num_reqs=4,
            max_pages=LIVE_PAGES + 2,
            is_target_verify=True,
            dtype=torch.int64,
            with_flashinfer=False,
        )

    def test_verify_crossing_the_request_and_query_row_tiles(self):
        # 257 requests and 2056 query rows cross the 256-request and 1024-row tiles.
        self._check(
            query_length=8,
            num_reqs=257,
            max_pages=LIVE_PAGES + 2,
            is_target_verify=True,
            dtype=torch.int64,
        )


if __name__ == "__main__":
    unittest.main()
