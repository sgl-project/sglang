"""Regressions for the DSA KPool page-local cache layout."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.dsa.kpool_fp8_index import (
    INDEX_HEAD_DIM,
    build_pooled_page_table_64,
    compute_pooled_write_locs,
    gather_index_k_scale_prefix_into,
    kpool_assemble_softmax_rotate_write_cache,
    kpool_max_closed_pools,
    kpool_write_tail_and_maybe_compress,
    update_kpool_write_plan_cuda_graph,
)
from sglang.srt.layers.attention.dsa.kpool_plan import (
    _alloc_kpool_write_plan_buffers,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")


class TestDsaKpoolPageLayout(CustomTestCase):
    def test_each_source_page_owns_its_compressed_rows(self):
        page_table = torch.tensor([11, 7, 19, 3], dtype=torch.int32)
        pool_ids = torch.arange(64, dtype=torch.int64)

        write_locs = compute_pooled_write_locs(page_table, pool_ids, pool_size=4)
        expected = page_table.to(torch.int64).repeat_interleave(16) * 64
        expected += torch.arange(16, dtype=torch.int64).repeat(4)

        torch.testing.assert_close(write_locs, expected)
        torch.testing.assert_close(
            build_pooled_page_table_64(page_table, pool_size=4), page_table
        )


@unittest.skipUnless(torch.cuda.is_available(), "Test requires CUDA")
class TestDsaKpoolMultiPool(CustomTestCase):
    POOL_SIZE = 4
    PAGE_SIZE = 64
    SLOTS_PER_PAGE = 64
    POOLED_SLOTS_PER_PAGE = PAGE_SIZE // POOL_SIZE
    NUM_DRAFT_TOKENS = 6

    def _pool(self) -> SimpleNamespace:
        return SimpleNamespace(
            page_size=self.PAGE_SIZE,
            index_head_dim=INDEX_HEAD_DIM,
            slots_per_page=self.SLOTS_PER_PAGE,
            pooled_slots_per_page=self.POOLED_SLOTS_PER_PAGE,
            index_kpool=self.POOL_SIZE,
            tail_extra_slots=self.NUM_DRAFT_TOKENS,
            quant_block_size=128,
        )

    def _empty_cache(self) -> torch.Tensor:
        page_nbytes = self.SLOTS_PER_PAGE * INDEX_HEAD_DIM + self.SLOTS_PER_PAGE * 4
        return torch.zeros((1, page_nbytes), dtype=torch.uint8, device="cuda")

    def test_gather_reads_first_quarter_from_each_page(self):
        pool = self._pool()
        page_nbytes = self.SLOTS_PER_PAGE * (INDEX_HEAD_DIM + 4)
        buf = torch.zeros((4, page_nbytes), dtype=torch.uint8, device="cuda")
        expected_k = torch.empty(
            (64, INDEX_HEAD_DIM), dtype=torch.uint8, device="cuda"
        )
        expected_scale = torch.arange(64, dtype=torch.float32, device="cuda") + 1

        for page in range(4):
            values = (
                torch.arange(
                    self.POOLED_SLOTS_PER_PAGE,
                    dtype=torch.uint8,
                    device="cuda",
                )
                + page * self.POOLED_SLOTS_PER_PAGE
            )
            page_k = buf[page, : self.SLOTS_PER_PAGE * INDEX_HEAD_DIM].view(
                self.SLOTS_PER_PAGE, INDEX_HEAD_DIM
            )
            page_k[: self.POOLED_SLOTS_PER_PAGE] = values[:, None]
            page_scale = buf[page, self.SLOTS_PER_PAGE * INDEX_HEAD_DIM :].view(
                torch.float32
            )
            start = page * self.POOLED_SLOTS_PER_PAGE
            end = (page + 1) * self.POOLED_SLOTS_PER_PAGE
            scale_slice = expected_scale[start:end]
            page_scale[: self.POOLED_SLOTS_PER_PAGE] = scale_slice
            expected_k[start:end] = values[:, None]

        actual_k = torch.empty_like(expected_k)
        actual_scale = torch.empty_like(expected_scale)
        gather_index_k_scale_prefix_into(
            pool=pool,
            buf=buf,
            page_indices=torch.arange(4, dtype=torch.int32, device="cuda"),
            seq_len=64,
            k_out=actual_k,
            scale_out=actual_scale,
        )

        torch.testing.assert_close(actual_k, expected_k)
        torch.testing.assert_close(actual_scale, expected_scale)

    def test_paged_mqa_reads_logical_rows_from_physical_pages(self):
        if torch.version.cuda is None or torch.cuda.get_device_capability()[0] < 9:
            self.skipTest("FP8 paged MQA requires Hopper or newer")

        from sglang.kernels.ops.attention.dsa.tilelang_kernel import (
            tilelang_fp8_paged_mqa_logits,
        )

        torch.manual_seed(42)
        num_pages = 4
        num_heads = 64
        logical_seq_len = 61
        page_nbytes = self.SLOTS_PER_PAGE * (INDEX_HEAD_DIM + 4)
        buf = torch.zeros(
            (num_pages, page_nbytes), dtype=torch.uint8, device="cuda"
        )
        page_k = torch.randn(
            num_pages,
            self.POOLED_SLOTS_PER_PAGE,
            INDEX_HEAD_DIM,
            device="cuda",
        ).to(torch.float8_e4m3fn)
        page_scale = torch.rand(
            num_pages, self.POOLED_SLOTS_PER_PAGE, device="cuda"
        )

        for page in range(num_pages):
            buf[page, : self.POOLED_SLOTS_PER_PAGE * INDEX_HEAD_DIM].copy_(
                page_k[page].view(torch.uint8).flatten()
            )
            scale_start = self.SLOTS_PER_PAGE * INDEX_HEAD_DIM
            scale_region = buf[page, scale_start:].view(torch.float32)
            scale_region[: self.POOLED_SLOTS_PER_PAGE].copy_(page_scale[page])

        q = torch.randn(
            1, 1, num_heads, INDEX_HEAD_DIM, device="cuda"
        ).to(torch.float8_e4m3fn)
        weights = torch.randn(1, num_heads, dtype=torch.float32, device="cuda")
        page_table = torch.tensor([[3, 1, 0, 2]], dtype=torch.int32, device="cuda")
        seq_lens = torch.tensor([logical_seq_len], dtype=torch.int32, device="cuda")

        logits = tilelang_fp8_paged_mqa_logits(
            q,
            buf.view(num_pages, self.SLOTS_PER_PAGE, 1, INDEX_HEAD_DIM + 4),
            weights,
            seq_lens,
            page_table,
            deep_gemm_metadata=None,
            max_seq_len=num_pages * self.POOLED_SLOTS_PER_PAGE,
            clean_logits=False,
            logical_block_size=self.POOLED_SLOTS_PER_PAGE,
        )

        order = page_table[0].to(torch.long)
        ordered_k = page_k.index_select(0, order).flatten(0, 1)[:logical_seq_len]
        ordered_scale = page_scale.index_select(0, order).flatten()[:logical_seq_len]
        reference = torch.relu(ordered_k.float() @ q[0, 0].float().T)
        reference = (reference * weights[0]).sum(dim=1) * ordered_scale

        torch.testing.assert_close(
            logits[0, :logical_seq_len], reference, atol=2e-2, rtol=2e-2
        )

    def test_write_plan_records_every_candidate_pool(self):
        batch_size = 2
        num_draft_tokens = self.NUM_DRAFT_TOKENS
        max_closed_pools = kpool_max_closed_pools(num_draft_tokens, self.POOL_SIZE)
        self.assertEqual(max_closed_pools, 2)

        plan = _alloc_kpool_write_plan_buffers(
            max_bs=batch_size,
            num_draft_tokens=num_draft_tokens,
            pool_size=self.POOL_SIZE,
            device=torch.device("cuda"),
            is_verify=True,
        )
        self.assertEqual(plan.write_loc.shape, (batch_size, max_closed_pools))

        write_start = torch.tensor([3, 255], dtype=torch.int32, device="cuda")
        req_pool_indices = torch.tensor([7, 11], dtype=torch.int64, device="cuda")
        real_page_table = torch.zeros(
            (batch_size * num_draft_tokens, 8),
            dtype=torch.int32,
            device="cuda",
        )
        real_page_table[:num_draft_tokens, 0] = 2
        real_page_table[num_draft_tokens:, 3] = 5
        real_page_table[num_draft_tokens:, 4] = 6

        update_kpool_write_plan_cuda_graph(
            write_start=write_start,
            req_pool_indices=req_pool_indices,
            real_page_table=real_page_table,
            req_out=plan.req,
            write_start_out=plan.write_start,
            tail_logical_start_out=plan.tail_logical_start,
            write_loc_out=plan.write_loc,
            pool_seqlens_per_q_out=plan.pool_seqlens_per_q,
            seqlens_per_q_out=plan.seqlens_per_q,
            pool_size=self.POOL_SIZE,
            num_draft_tokens=num_draft_tokens,
            slots_per_page=self.POOLED_SLOTS_PER_PAGE,
            page_stride=self.SLOTS_PER_PAGE,
        )

        torch.testing.assert_close(plan.req, req_pool_indices)
        torch.testing.assert_close(plan.write_start, write_start)
        torch.testing.assert_close(
            plan.tail_logical_start,
            torch.tensor([0, 252], dtype=torch.int32, device="cuda"),
        )
        torch.testing.assert_close(
            plan.write_loc,
            torch.tensor(
                [
                    [2 * self.SLOTS_PER_PAGE, 2 * self.SLOTS_PER_PAGE + 1],
                    [
                        5 * self.SLOTS_PER_PAGE + 15,
                        6 * self.SLOTS_PER_PAGE,
                    ],
                ],
                dtype=torch.int64,
                device="cuda",
            ),
        )

    def _run_compress_case(self, effective_n: int, expected_closed_pools: int):
        torch.manual_seed(42)
        pool = self._pool()
        num_draft_tokens = self.NUM_DRAFT_TOKENS
        tail_size = self.POOL_SIZE + num_draft_tokens
        write_start_value = 3

        key = torch.randn(
            num_draft_tokens, INDEX_HEAD_DIM, dtype=torch.bfloat16, device="cuda"
        )
        score = torch.randn_like(key)
        ape = torch.randn(
            self.POOL_SIZE, INDEX_HEAD_DIM, dtype=torch.float32, device="cuda"
        )
        tail_k_initial = torch.randn(
            1, tail_size, INDEX_HEAD_DIM, dtype=torch.bfloat16, device="cuda"
        )
        tail_score_initial = torch.randn_like(tail_k_initial)

        tail_k_expected = tail_k_initial.clone()
        tail_score_expected = tail_score_initial.clone()
        for i in range(num_draft_tokens):
            physical_slot = (write_start_value + i) % tail_size
            tail_k_expected[0, physical_slot] = key[i]
            tail_score_expected[0, physical_slot] = score[i]

        expected_cache = self._empty_cache()
        dummy_chunk = torch.zeros(
            1, INDEX_HEAD_DIM, dtype=torch.bfloat16, device="cuda"
        )
        kpool_assemble_softmax_rotate_write_cache(
            pool=pool,
            buf=expected_cache,
            chunk_k=dummy_chunk,
            chunk_score=dummy_chunk,
            tail_k=tail_k_expected,
            tail_score=tail_score_expected,
            req_pool_idx=torch.zeros(
                expected_closed_pools, dtype=torch.int64, device="cuda"
            ),
            n_from_tail=torch.full(
                (expected_closed_pools,),
                self.POOL_SIZE,
                dtype=torch.int32,
                device="cuda",
            ),
            chunk_src_start=torch.zeros(
                expected_closed_pools, dtype=torch.int64, device="cuda"
            ),
            tail_logical_base=torch.arange(
                0,
                expected_closed_pools * self.POOL_SIZE,
                self.POOL_SIZE,
                dtype=torch.int32,
                device="cuda",
            ),
            ape=ape,
            loc=torch.arange(expected_closed_pools, dtype=torch.int64, device="cuda"),
            round_scale=False,
        )

        actual_cache = self._empty_cache()
        tail_k_actual = tail_k_initial.clone()
        tail_score_actual = tail_score_initial.clone()
        kpool_write_tail_and_maybe_compress(
            pool=pool,
            buf=actual_cache,
            key=key,
            score=score,
            tail_k=tail_k_actual,
            tail_score=tail_score_actual,
            ape=ape,
            req_pool_indices=torch.zeros(1, dtype=torch.int64, device="cuda"),
            write_start=torch.tensor(
                [write_start_value], dtype=torch.int32, device="cuda"
            ),
            tail_logical_start=torch.zeros(1, dtype=torch.int32, device="cuda"),
            write_loc=torch.tensor([[0, 1]], dtype=torch.int64, device="cuda"),
            out_cache_loc=torch.arange(
                1, num_draft_tokens + 1, dtype=torch.int64, device="cuda"
            ),
            num_draft_tokens=num_draft_tokens,
            round_scale=False,
            effective_n_per_batch=torch.tensor(
                [effective_n], dtype=torch.int32, device="cuda"
            ),
        )

        torch.testing.assert_close(tail_k_actual, tail_k_expected, atol=0, rtol=0)
        torch.testing.assert_close(
            tail_score_actual, tail_score_expected, atol=0, rtol=0
        )
        torch.testing.assert_close(actual_cache, expected_cache, atol=0, rtol=0)

    def test_compresses_two_pools_when_draft_window_closes_two(self):
        self._run_compress_case(
            effective_n=self.NUM_DRAFT_TOKENS,
            expected_closed_pools=2,
        )

    def test_effective_n_only_compresses_accepted_pools(self):
        self._run_compress_case(effective_n=2, expected_closed_pools=1)


if __name__ == "__main__":
    unittest.main()
