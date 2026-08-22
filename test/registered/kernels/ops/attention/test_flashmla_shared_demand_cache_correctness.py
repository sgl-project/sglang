import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=60,
    stage="base-b-kernel-unit",
    runner_config="1-gpu-large",
)


class TestFlashMLASharedDemandCacheCorrectness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("FlashMLA Shared-KV requires CUDA")
        if torch.cuda.get_device_capability()[0] != 9:
            raise unittest.SkipTest("FlashMLA Shared-KV requires SM90")

        from sgl_kernel.flash_mla import (
            flash_mla_with_kvcache,
            get_mla_metadata,
        )

        cls.flash_mla_with_kvcache = staticmethod(flash_mla_with_kvcache)
        cls.get_mla_metadata = staticmethod(get_mla_metadata)

    def _fixture(self, *, seq_q: int):
        device = torch.device("cuda")
        torch.manual_seed(20260811)
        num_rows = 2048
        page_size = 64
        num_heads = 64

        rows = torch.empty((num_rows, 656), dtype=torch.uint8, device=device)
        rows[:, :512].view(torch.float8_e4m3fn).copy_(
            torch.randn((num_rows, 512), device=device).clamp_(-4, 4)
        )
        rows[:, 512:528].view(torch.float32).copy_(
            torch.rand((num_rows, 4), dtype=torch.float32, device=device) + 0.25
        )
        rows[:, 528:].view(torch.bfloat16).copy_(
            torch.randn((num_rows, 64), dtype=torch.bfloat16, device=device)
        )
        k_cache = rows.view(num_rows // page_size, page_size, 1, 656)
        q = torch.randn((1, seq_q, num_heads, 576), dtype=torch.bfloat16, device=device)
        indices = (
            torch.arange(num_rows, dtype=torch.int32, device=device)
            .view(1, 1, num_rows)
            .expand(1, seq_q, num_rows)
            .contiguous()
        )
        cache_seqlens = torch.full((1,), num_rows, dtype=torch.int32, device=device)
        metadata, num_splits = self.get_mla_metadata(
            cache_seqlens=cache_seqlens,
            num_q_tokens_per_head_k=seq_q * num_heads,
            num_heads_k=1,
            num_heads_q=num_heads,
            is_fp8_kvcache=True,
            topk=num_rows,
        )
        return q, k_cache, rows, indices, cache_seqlens, metadata, num_splits

    def _run_direct(self, q, k_cache, indices, cache_seqlens, metadata, num_splits):
        return self.flash_mla_with_kvcache(
            q=q,
            k_cache=k_cache,
            block_table=torch.empty((1, 0), dtype=torch.int32, device=q.device),
            cache_seqlens=cache_seqlens,
            head_dim_v=512,
            tile_scheduler_metadata=metadata,
            num_splits=num_splits,
            is_fp8_kvcache=True,
            indices=indices,
        )

    def _shared_kwargs(self, *, q, cache_rows, local_begin, local_end):
        return dict(
            shared_kv_row_cache=torch.empty(
                (cache_rows, 656), dtype=torch.uint8, device=q.device
            ),
            shared_kv_cache_tags=torch.zeros(
                (1, cache_rows), dtype=torch.int64, device=q.device
            ),
            shared_kv_cache_rows_per_request=cache_rows,
            shared_kv_num_request_slots=1,
            shared_kv_cache_epoch=1,
            shared_kv_cache_generation_tensor=torch.ones(
                (), dtype=torch.int32, device=q.device
            ),
            shared_kv_local_row_begin=local_begin,
            shared_kv_local_row_end=local_end,
        )

    def test_request_slot_reuse_refills_after_lifecycle_invalidation(self):
        q, k_cache, rows, indices, cache_seqlens, metadata, num_splits = self._fixture(
            seq_q=1
        )
        kwargs = self._shared_kwargs(
            q=q, cache_rows=4096, local_begin=2047, local_end=2048
        )

        self.flash_mla_with_kvcache(
            q=q,
            k_cache=k_cache,
            block_table=torch.empty((1, 0), dtype=torch.int32, device=q.device),
            cache_seqlens=cache_seqlens,
            head_dim_v=512,
            tile_scheduler_metadata=metadata,
            num_splits=num_splits,
            is_fp8_kvcache=True,
            indices=indices,
            **kwargs,
        )

        rows[17].copy_(rows[23])
        # Production only rewrites a physical row after its old request slot is
        # released.  The request-generation lifecycle clears that slot's tags
        # before the replacement request can execute attention.
        kwargs["shared_kv_cache_tags"].zero_()
        kwargs["shared_kv_cache_generation_tensor"].fill_(2)
        expected_out, expected_lse = self._run_direct(
            q, k_cache, indices, cache_seqlens, metadata, num_splits
        )
        actual_out, actual_lse = self.flash_mla_with_kvcache(
            q=q,
            k_cache=k_cache,
            block_table=torch.empty((1, 0), dtype=torch.int32, device=q.device),
            cache_seqlens=cache_seqlens,
            head_dim_v=512,
            tile_scheduler_metadata=metadata,
            num_splits=num_splits,
            is_fp8_kvcache=True,
            indices=indices,
            **kwargs,
        )

        torch.testing.assert_close(actual_out, expected_out, rtol=0, atol=0)
        torch.testing.assert_close(actual_lse, expected_lse, rtol=0, atol=0)

    def test_current_row_widths_one_to_four_match_authoritative_rows(self):
        for width in range(1, 5):
            with self.subTest(width=width):
                q, k_cache, rows, indices, cache_seqlens, metadata, num_splits = (
                    self._fixture(seq_q=width)
                )
                expected_out, expected_lse = self._run_direct(
                    q, k_cache, indices, cache_seqlens, metadata, num_splits
                )
                marked_indices = indices.clone()
                current_rows = torch.empty(
                    (width, width, 656), dtype=torch.uint8, device=q.device
                )
                current_row_ids = torch.full(
                    (width, width), -1, dtype=torch.int32, device=q.device
                )
                current_row_counts = torch.arange(
                    1, width + 1, dtype=torch.int32, device=q.device
                )
                for query_row in range(width):
                    count = query_row + 1
                    marked_indices[0, query_row, :count] = torch.arange(
                        -2, -2 - count, -1, dtype=torch.int32, device=q.device
                    )
                    current_rows[query_row, :count].copy_(rows[:count])
                    current_row_ids[query_row, :count] = torch.arange(
                        count, dtype=torch.int32, device=q.device
                    )

                kwargs = self._shared_kwargs(
                    q=q, cache_rows=4096, local_begin=0, local_end=2048
                )
                actual_out, actual_lse = self.flash_mla_with_kvcache(
                    q=q,
                    k_cache=k_cache,
                    block_table=torch.empty((1, 0), dtype=torch.int32, device=q.device),
                    cache_seqlens=cache_seqlens,
                    head_dim_v=512,
                    tile_scheduler_metadata=metadata,
                    num_splits=num_splits,
                    is_fp8_kvcache=True,
                    indices=marked_indices,
                    shared_kv_current_rows=current_rows,
                    shared_kv_current_row_ids=current_row_ids,
                    shared_kv_current_row_counts=current_row_counts,
                    **kwargs,
                )

                torch.testing.assert_close(actual_out, expected_out, rtol=0, atol=0)
                torch.testing.assert_close(actual_lse, expected_lse, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
