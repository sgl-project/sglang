"""GPU regressions for DSA kpool speculative writes and cache layout."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention.dsa import aiter_paged_mqa_logits
from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype
from sglang.srt.layers.attention.dsa import kpool_fp8_index
from sglang.srt.layers.attention.dsa.kpool_fp8_index import (
    INDEX_HEAD_DIM,
    _topk_from_pooled_history_logits_unfused,
    gather_index_k_scale_prefix_into,
    kpool_assemble_softmax_rotate_write_cache,
    kpool_max_closed_pools,
    kpool_softmax_rotate_write_cache,
    kpool_write_tail_and_maybe_compress,
    topk_from_pooled_history_logits,
    update_kpool_write_plan_cuda_graph,
)
from sglang.srt.layers.attention.dsa.kpool_plan import (
    _alloc_kpool_write_plan_buffers,
)
from sglang.srt.layers.attention.dsa.utils import (
    aiter_can_use_preshuffle_paged_mqa,
)
from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=20, stage="jit-kernel-unit", runner_config="amd")


@unittest.skipUnless(torch.cuda.is_available(), "Test requires a GPU")
class TestDsaKpoolMultiPool(CustomTestCase):
    POOL_SIZE = 4
    PAGE_SIZE = 64
    SLOTS_PER_PAGE = 64
    NUM_DRAFT_TOKENS = 6
    NUM_INDEX_HEADS = 32

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

    def test_fused_topk_matches_unfused_physical_token_sets(self):
        torch.manual_seed(45)
        batch_size = 2
        score_cols = 4096
        pool_size = self.POOL_SIZE
        topk = 2048
        logits = torch.randn(batch_size, score_cols, dtype=torch.float32, device="cuda")
        group_lengths = torch.tensor([256, 511], dtype=torch.int32, device="cuda")
        seq_lens = group_lengths * pool_size + torch.tensor(
            [0, 3], dtype=torch.int32, device="cuda"
        )
        page_table = torch.arange(
            batch_size * score_cols * pool_size,
            dtype=torch.int32,
            device="cuda",
        ).view(batch_size, score_cols * pool_size)

        actual = topk_from_pooled_history_logits(
            logits=logits,
            group_lengths=group_lengths,
            pool_size=pool_size,
            topk=topk,
            page_table=page_table,
            seq_lens=seq_lens,
        )
        expected = _topk_from_pooled_history_logits_unfused(
            logits=logits,
            group_lengths=group_lengths,
            pool_size=pool_size,
            topk=topk,
            page_table=page_table,
            seq_lens=seq_lens,
        )

        torch.testing.assert_close(
            torch.sort(actual, dim=1).values,
            torch.sort(expected, dim=1).values,
            atol=0,
            rtol=0,
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

    def test_preshuffled_cache_round_trip_and_raw_layout(self):
        torch.manual_seed(43)
        pool = self._pool()
        num_slots = self.SLOTS_PER_PAGE
        tile = 16
        num_col_tiles = INDEX_HEAD_DIM // tile

        slot_k = torch.randn(
            num_slots,
            self.POOL_SIZE,
            INDEX_HEAD_DIM,
            dtype=torch.bfloat16,
            device="cuda",
        )
        slot_score = torch.zeros_like(slot_k)
        ape = torch.zeros(
            self.POOL_SIZE,
            INDEX_HEAD_DIM,
            dtype=torch.float32,
            device="cuda",
        )
        loc = torch.arange(num_slots, dtype=torch.int64, device="cuda")
        cache = self._empty_cache()

        with patch.object(kpool_fp8_index, "_preshuffle_tile", return_value=tile):
            compressed_k, compressed_scale = kpool_softmax_rotate_write_cache(
                pool=pool,
                buf=cache,
                slot_k=slot_k,
                slot_score=slot_score,
                ape=ape,
                loc=loc,
                return_compressed=True,
            )

            gathered_k = torch.empty(
                (num_slots, INDEX_HEAD_DIM), dtype=torch.uint8, device="cuda"
            )
            gathered_scale = torch.empty(
                (num_slots,), dtype=torch.float32, device="cuda"
            )
            gather_index_k_scale_prefix_into(
                pool=pool,
                buf=cache,
                page_indices=torch.zeros(1, dtype=torch.int32, device="cuda"),
                seq_len=num_slots,
                k_out=gathered_k,
                scale_out=gathered_scale,
            )

        compressed_k_u8 = compressed_k.view(torch.uint8)
        torch.testing.assert_close(gathered_k, compressed_k_u8, atol=0, rtol=0)
        torch.testing.assert_close(gathered_scale, compressed_scale, atol=0, rtol=0)

        raw_k = cache[0, : num_slots * INDEX_HEAD_DIM].view(
            num_slots // tile, num_col_tiles, tile, tile
        )
        expected_raw_k = (
            compressed_k_u8.view(num_slots // tile, tile, num_col_tiles, tile)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        torch.testing.assert_close(raw_k, expected_raw_k, atol=0, rtol=0)

    @unittest.skipUnless(
        is_hip() and aiter_can_use_preshuffle_paged_mqa(),
        "requires the ROCm AITER preshuffle paged-MQA path",
    )
    def test_preshuffled_paged_mqa_matches_gathered_mqa(self):
        from aiter.ops.triton.fp8_mqa_logits import fp8_mqa_logits

        torch.manual_seed(44)
        pool = self._pool()
        page_nbytes = self.SLOTS_PER_PAGE * (INDEX_HEAD_DIM + 4)
        tile = 16
        num_token_tiles = self.SLOTS_PER_PAGE // tile
        num_col_tiles = INDEX_HEAD_DIM // tile

        for batch_size in (1, 8):
            with self.subTest(batch_size=batch_size):
                cache = torch.zeros(
                    (batch_size, page_nbytes), dtype=torch.uint8, device="cuda"
                )
                k = (
                    torch.randn(
                        batch_size,
                        self.SLOTS_PER_PAGE,
                        INDEX_HEAD_DIM,
                        device="cuda",
                    )
                    * 2
                ).to(fp8_dtype)
                k_scale = (
                    torch.rand(batch_size, self.SLOTS_PER_PAGE, device="cuda") + 0.25
                )

                preshuffled = (
                    k.view(
                        batch_size,
                        num_token_tiles,
                        tile,
                        num_col_tiles,
                        tile,
                    )
                    .permute(0, 1, 3, 2, 4)
                    .contiguous()
                    .view(batch_size, -1)
                    .view(torch.uint8)
                )
                k_region_bytes = self.SLOTS_PER_PAGE * INDEX_HEAD_DIM
                cache[:, :k_region_bytes].copy_(preshuffled)
                cache[:, k_region_bytes:].view(torch.float32).copy_(k_scale)

                q = (
                    torch.randn(
                        batch_size,
                        self.NUM_INDEX_HEADS,
                        INDEX_HEAD_DIM,
                        device="cuda",
                    )
                    * 2
                ).to(fp8_dtype)
                weights = torch.randn(
                    batch_size,
                    self.NUM_INDEX_HEADS,
                    dtype=torch.float32,
                    device="cuda",
                )
                if batch_size == 1:
                    seq_lens = torch.tensor([53], dtype=torch.int32, device="cuda")
                else:
                    seq_lens = torch.tensor(
                        [1, 7, 16, 31, 32, 47, 63, 64],
                        dtype=torch.int32,
                        device="cuda",
                    )
                block_tables = torch.arange(
                    batch_size, dtype=torch.int32, device="cuda"
                ).view(batch_size, 1)

                paged_logits = aiter_paged_mqa_logits(
                    q,
                    cache.view(batch_size, self.SLOTS_PER_PAGE, 1, INDEX_HEAD_DIM + 4),
                    weights,
                    seq_lens,
                    block_tables,
                    self.SLOTS_PER_PAGE,
                    preshuffle=True,
                    kv_block_size=self.SLOTS_PER_PAGE,
                )

                gathered_k = []
                gathered_scale = []
                for batch_idx, seq_len in enumerate(seq_lens.tolist()):
                    k_out = torch.empty(
                        (seq_len, INDEX_HEAD_DIM), dtype=torch.uint8, device="cuda"
                    )
                    scale_out = torch.empty(
                        (seq_len,), dtype=torch.float32, device="cuda"
                    )
                    gather_index_k_scale_prefix_into(
                        pool=pool,
                        buf=cache,
                        page_indices=torch.tensor(
                            [batch_idx], dtype=torch.int32, device="cuda"
                        ),
                        seq_len=seq_len,
                        k_out=k_out,
                        scale_out=scale_out,
                    )
                    gathered_k.append(k_out.view(fp8_dtype))
                    gathered_scale.append(scale_out)

                flat_k = torch.cat(gathered_k)
                flat_scale = torch.cat(gathered_scale)
                ends = seq_lens.cumsum(0)
                starts = ends - seq_lens
                gathered_logits = fp8_mqa_logits(
                    q,
                    flat_k,
                    flat_scale,
                    weights,
                    starts,
                    ends,
                    clean_logits=True,
                )

                for batch_idx, seq_len in enumerate(seq_lens.tolist()):
                    start = starts[batch_idx].item()
                    torch.testing.assert_close(
                        paged_logits[batch_idx, :seq_len],
                        gathered_logits[batch_idx, start : start + seq_len],
                        atol=2e-2,
                        rtol=2e-2,
                    )


if __name__ == "__main__":
    unittest.main()
