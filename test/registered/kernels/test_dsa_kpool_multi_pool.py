"""CUDA regressions for DSA kpool speculative writes spanning multiple pools."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.dsa.kpool_fp8_index import (
    INDEX_HEAD_DIM,
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


@unittest.skipUnless(torch.cuda.is_available(), "Test requires CUDA")
class TestDsaKpoolMultiPool(CustomTestCase):
    POOL_SIZE = 4
    PAGE_SIZE = 64
    SLOTS_PER_PAGE = 64
    NUM_DRAFT_TOKENS = 6

    def _pool(self) -> SimpleNamespace:
        return SimpleNamespace(
            page_size=self.PAGE_SIZE,
            index_head_dim=INDEX_HEAD_DIM,
            slots_per_page=self.SLOTS_PER_PAGE,
            index_kpool=self.POOL_SIZE,
            tail_extra_slots=self.NUM_DRAFT_TOKENS,
            quant_block_size=128,
        )

    def _empty_cache(self) -> torch.Tensor:
        page_nbytes = self.SLOTS_PER_PAGE * INDEX_HEAD_DIM + self.SLOTS_PER_PAGE * 4
        return torch.zeros((1, page_nbytes), dtype=torch.uint8, device="cuda")

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
        real_page_table[:num_draft_tokens, 4] = 3
        real_page_table[num_draft_tokens:, 0] = 5
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
            slots_per_page=self.SLOTS_PER_PAGE,
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
                        5 * self.SLOTS_PER_PAGE + 63,
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
