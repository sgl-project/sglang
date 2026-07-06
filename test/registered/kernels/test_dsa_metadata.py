import unittest

import torch

from sglang.srt.layers.attention.triton_ops.dsa_metadata import (
    fused_dsa_decode_metadata,
    fused_dsa_draft_extend_metadata,
    fused_dsa_target_verify_metadata,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=8, stage="base-b", runner_config="1-gpu-large")


def _cu_seqlens(seqlens: torch.Tensor) -> torch.Tensor:
    out = torch.empty(seqlens.numel() + 1, dtype=torch.int32, device=seqlens.device)
    out[:1].zero_()
    out[1:].copy_(torch.cumsum(seqlens.to(torch.int32), dim=0, dtype=torch.int32))
    return out


def _dsa_seqlens(seqlens: torch.Tensor, topk: int) -> torch.Tensor:
    return torch.minimum(
        seqlens.to(torch.int32), torch.tensor(topk, device=seqlens.device)
    )


def _real_page_table(page_table_1: torch.Tensor, real_page_size: int) -> torch.Tensor:
    if real_page_size == 1:
        return page_table_1
    return page_table_1[:, ::real_page_size] // real_page_size


def _make_req_to_token(
    pool_size: int, max_len: int, device: torch.device
) -> torch.Tensor:
    # Row-dependent values catch accidental row reuse, while monotonic columns make
    # real-page-table checks easy to reason about.
    cols = torch.arange(max_len, dtype=torch.int32, device=device)
    rows = torch.arange(pool_size, dtype=torch.int32, device=device).view(-1, 1)
    return rows * (max_len + 17) + cols


def _assert_equal(actual: torch.Tensor, expected: torch.Tensor, name: str) -> None:
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, msg=name)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for this test.")
class TestDSAMetadataKernels(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.device = torch.device("cuda")

    def _check_decode(
        self,
        seq_lens_values,
        *,
        max_len: int,
        dsa_index_topk: int,
        real_page_size: int,
    ):
        bs = len(seq_lens_values)
        pool_size = max(bs + 3, 8)
        seq_lens = torch.tensor(seq_lens_values, dtype=torch.int64, device=self.device)
        req_pool_indices = torch.arange(bs, dtype=torch.int64, device=self.device) * 2
        req_to_token = _make_req_to_token(pool_size * 2, max_len, self.device)

        cache_seqlens = torch.empty(bs, dtype=torch.int32, device=self.device)
        cu_seqlens_k = torch.empty(bs + 1, dtype=torch.int32, device=self.device)
        page_table_1 = torch.empty((bs, max_len), dtype=torch.int32, device=self.device)
        dsa_cache_seqlens = torch.empty(bs, dtype=torch.int32, device=self.device)
        dsa_cu_seqlens_k = torch.empty(bs + 1, dtype=torch.int32, device=self.device)
        real_page_table = (
            torch.empty(
                (bs, (max_len + real_page_size - 1) // real_page_size),
                dtype=torch.int32,
                device=self.device,
            )
            if real_page_size > 1
            else page_table_1
        )

        fused_dsa_decode_metadata(
            seq_lens=seq_lens,
            req_pool_indices=req_pool_indices,
            req_to_token=req_to_token,
            cache_seqlens=cache_seqlens,
            cu_seqlens_k=cu_seqlens_k,
            page_table_1=page_table_1,
            dsa_cache_seqlens=dsa_cache_seqlens,
            dsa_cu_seqlens_k=dsa_cu_seqlens_k,
            real_page_table=real_page_table,
            bs=bs,
            max_len=max_len,
            dsa_index_topk=dsa_index_topk,
            real_page_size=real_page_size,
        )

        expected_cache = seq_lens.to(torch.int32)
        expected_page_table = req_to_token[req_pool_indices, :max_len].contiguous()
        expected_dsa = _dsa_seqlens(expected_cache, dsa_index_topk)

        _assert_equal(cache_seqlens, expected_cache, "decode cache_seqlens")
        _assert_equal(cu_seqlens_k, _cu_seqlens(expected_cache), "decode cu_seqlens_k")
        _assert_equal(page_table_1, expected_page_table, "decode page_table_1")
        _assert_equal(dsa_cache_seqlens, expected_dsa, "decode dsa_cache_seqlens")
        _assert_equal(
            dsa_cu_seqlens_k, _cu_seqlens(expected_dsa), "decode dsa_cu_seqlens_k"
        )
        if real_page_size > 1:
            _assert_equal(
                real_page_table,
                _real_page_table(expected_page_table, real_page_size),
                "decode real_page_table",
            )

    def _check_target_verify(
        self,
        seq_lens_values,
        *,
        max_seqlen_k: int,
        dsa_index_topk: int,
        real_page_size: int,
        next_n: int,
        fill_ctx_lens: bool,
    ):
        bs = len(seq_lens_values)
        expanded_size = bs * next_n
        pool_size = max(bs + 5, 8)
        seq_lens = torch.tensor(seq_lens_values, dtype=torch.int64, device=self.device)
        req_pool_indices = torch.arange(bs, dtype=torch.int64, device=self.device) + 1
        req_to_token = _make_req_to_token(pool_size + 2, max_seqlen_k, self.device)

        cache_seqlens = torch.empty(bs, dtype=torch.int32, device=self.device)
        cu_seqlens_k = torch.empty(bs + 1, dtype=torch.int32, device=self.device)
        page_table_1 = torch.empty(
            (expanded_size, max_seqlen_k), dtype=torch.int32, device=self.device
        )
        seqlens_expanded = torch.empty(
            expanded_size, dtype=torch.int32, device=self.device
        )
        dsa_cache_seqlens = torch.empty(
            expanded_size, dtype=torch.int32, device=self.device
        )
        dsa_cu_seqlens_k = torch.empty(
            expanded_size + 1, dtype=torch.int32, device=self.device
        )
        real_page_table = (
            torch.empty(
                (
                    expanded_size,
                    (max_seqlen_k + real_page_size - 1) // real_page_size,
                ),
                dtype=torch.int32,
                device=self.device,
            )
            if real_page_size > 1
            else page_table_1
        )
        paged_mqa_ctx_lens_2d = (
            torch.empty((bs, next_n), dtype=torch.int32, device=self.device)
            if fill_ctx_lens
            else None
        )

        fused_dsa_target_verify_metadata(
            seq_lens=seq_lens,
            req_pool_indices=req_pool_indices,
            req_to_token=req_to_token,
            cache_seqlens=cache_seqlens,
            cu_seqlens_k=cu_seqlens_k,
            page_table_1=page_table_1,
            seqlens_expanded=seqlens_expanded,
            dsa_cache_seqlens=dsa_cache_seqlens,
            dsa_cu_seqlens_k=dsa_cu_seqlens_k,
            real_page_table=real_page_table,
            bs=bs,
            max_seqlen_k=max_seqlen_k,
            dsa_index_topk=dsa_index_topk,
            real_page_size=real_page_size,
            next_n=next_n,
            paged_mqa_ctx_lens_2d=paged_mqa_ctx_lens_2d,
        )

        expected_cache = (seq_lens + next_n).to(torch.int32)
        base_page_table = req_to_token[req_pool_indices, :max_seqlen_k].contiguous()
        expected_page_table = torch.repeat_interleave(
            base_page_table, repeats=next_n, dim=0
        ).contiguous()
        draft_offsets = torch.arange(next_n, dtype=torch.int32, device=self.device)
        expected_expanded = seq_lens.to(torch.int32).view(-1, 1) + draft_offsets + 1
        expected_expanded = expected_expanded.reshape(-1).contiguous()
        expected_dsa = _dsa_seqlens(expected_expanded, dsa_index_topk)

        _assert_equal(cache_seqlens, expected_cache, "target cache_seqlens")
        _assert_equal(cu_seqlens_k, _cu_seqlens(expected_cache), "target cu_seqlens_k")
        _assert_equal(page_table_1, expected_page_table, "target page_table_1")
        _assert_equal(seqlens_expanded, expected_expanded, "target seqlens_expanded")
        _assert_equal(dsa_cache_seqlens, expected_dsa, "target dsa_cache_seqlens")
        _assert_equal(
            dsa_cu_seqlens_k, _cu_seqlens(expected_dsa), "target dsa_cu_seqlens_k"
        )
        if real_page_size > 1:
            _assert_equal(
                real_page_table,
                _real_page_table(expected_page_table, real_page_size),
                "target real_page_table",
            )
        if fill_ctx_lens:
            expected_ctx = expected_cache.view(bs, 1).expand(bs, next_n).contiguous()
            _assert_equal(
                paged_mqa_ctx_lens_2d, expected_ctx, "target paged_mqa_ctx_lens_2d"
            )

    def _check_draft_extend(
        self,
        seq_lens_values,
        extend_seq_lens_values,
        *,
        max_seqlen_k: int,
        dsa_index_topk: int,
        real_page_size: int,
        max_extend_len: int,
        max_total_len: int,
        static_extend_len: bool,
    ):
        bs = len(seq_lens_values)
        total_len = sum(extend_seq_lens_values)
        pool_size = max(bs + 4, 8)
        seq_lens = torch.tensor(seq_lens_values, dtype=torch.int64, device=self.device)
        extend_seq_lens = torch.tensor(
            extend_seq_lens_values, dtype=torch.int32, device=self.device
        )
        req_pool_indices = torch.arange(bs, dtype=torch.int64, device=self.device) + 2
        req_to_token = _make_req_to_token(pool_size + 4, max_seqlen_k, self.device)

        cache_seqlens = torch.empty(bs, dtype=torch.int32, device=self.device)
        cu_seqlens_k = torch.empty(bs + 1, dtype=torch.int32, device=self.device)
        page_table_1 = torch.empty(
            (max_total_len, max_seqlen_k), dtype=torch.int32, device=self.device
        )
        seqlens_expanded = torch.empty(
            max_total_len, dtype=torch.int32, device=self.device
        )
        dsa_cache_seqlens = torch.empty(
            max_total_len, dtype=torch.int32, device=self.device
        )
        dsa_cu_seqlens_k = torch.empty(
            max_total_len + 1, dtype=torch.int32, device=self.device
        )
        real_page_table = (
            torch.empty(
                (max_total_len, (max_seqlen_k + real_page_size - 1) // real_page_size),
                dtype=torch.int32,
                device=self.device,
            )
            if real_page_size > 1
            else page_table_1
        )

        fused_dsa_draft_extend_metadata(
            seq_lens=seq_lens,
            extend_seq_lens=extend_seq_lens,
            req_pool_indices=req_pool_indices,
            req_to_token=req_to_token,
            cache_seqlens=cache_seqlens,
            cu_seqlens_k=cu_seqlens_k,
            page_table_1=page_table_1,
            seqlens_expanded=seqlens_expanded,
            dsa_cache_seqlens=dsa_cache_seqlens,
            dsa_cu_seqlens_k=dsa_cu_seqlens_k,
            real_page_table=real_page_table,
            bs=bs,
            total_len=total_len,
            max_seqlen_k=max_seqlen_k,
            dsa_index_topk=dsa_index_topk,
            real_page_size=real_page_size,
            max_extend_len=max_extend_len,
            max_total_len=max_total_len,
            static_extend_len=static_extend_len,
        )

        expected_cache = seq_lens.to(torch.int32)
        base_page_table = req_to_token[req_pool_indices, :max_seqlen_k].contiguous()
        expected_page_table = torch.repeat_interleave(
            base_page_table, repeats=extend_seq_lens, dim=0
        ).contiguous()
        expanded_parts = []
        for seq_len, qo_len in zip(seq_lens, extend_seq_lens, strict=True):
            expanded_parts.append(
                torch.arange(
                    seq_len.item() - qo_len.item() + 1,
                    seq_len.item() + 1,
                    dtype=torch.int32,
                    device=self.device,
                )
            )
        expected_expanded = (
            torch.cat(expanded_parts, dim=0)
            if expanded_parts
            else torch.empty(0, dtype=torch.int32, device=self.device)
        )
        expected_dsa = _dsa_seqlens(expected_expanded, dsa_index_topk)

        _assert_equal(cache_seqlens, expected_cache, "draft cache_seqlens")
        _assert_equal(cu_seqlens_k, _cu_seqlens(expected_cache), "draft cu_seqlens_k")
        _assert_equal(
            page_table_1[:total_len], expected_page_table, "draft page_table_1"
        )
        _assert_equal(
            seqlens_expanded[:total_len], expected_expanded, "draft seqlens_expanded"
        )
        _assert_equal(
            dsa_cache_seqlens[:total_len], expected_dsa, "draft dsa_cache_seqlens"
        )
        _assert_equal(
            dsa_cu_seqlens_k[: total_len + 1],
            _cu_seqlens(expected_dsa),
            "draft dsa_cu_seqlens_k",
        )
        if real_page_size > 1:
            _assert_equal(
                real_page_table[:total_len],
                _real_page_table(expected_page_table, real_page_size),
                "draft real_page_table",
            )

    def test_decode_matches_eager_reference(self):
        for real_page_size in (1, 64):
            with self.subTest(real_page_size=real_page_size):
                self._check_decode(
                    [1, 7, 65, 513],
                    max_len=769,
                    dsa_index_topk=64,
                    real_page_size=real_page_size,
                )

    def test_target_verify_matches_eager_reference(self):
        for real_page_size, fill_ctx_lens in ((1, False), (64, True)):
            with self.subTest(
                real_page_size=real_page_size, fill_ctx_lens=fill_ctx_lens
            ):
                self._check_target_verify(
                    [5, 63, 128],
                    max_seqlen_k=257,
                    dsa_index_topk=64,
                    real_page_size=real_page_size,
                    next_n=4,
                    fill_ctx_lens=fill_ctx_lens,
                )

    def test_draft_extend_static_width_matches_eager_reference(self):
        self._check_draft_extend(
            [16, 31, 80],
            [4, 4, 4],
            max_seqlen_k=193,
            dsa_index_topk=64,
            real_page_size=1,
            max_extend_len=4,
            max_total_len=12,
            static_extend_len=True,
        )

    def test_draft_extend_variable_width_defensive_path(self):
        # The production draft-extend-v2 replay path uses static_extend_len=True.
        # Keep this case to guard the generic variable-width kernel branch.
        self._check_draft_extend(
            [12, 31, 80],
            [3, 5, 2],
            max_seqlen_k=193,
            dsa_index_topk=64,
            real_page_size=64,
            max_extend_len=5,
            max_total_len=10,
            static_extend_len=False,
        )

    def test_draft_extend_partial_fill(self):
        self._check_draft_extend(
            [12, 31, 80],
            [3, 5, 2],
            max_seqlen_k=193,
            dsa_index_topk=64,
            real_page_size=64,
            max_extend_len=5,
            max_total_len=16,
            static_extend_len=False,
        )

    def test_empty_batch(self):
        self._check_decode(
            [],
            max_len=8,
            dsa_index_topk=64,
            real_page_size=64,
        )
        self._check_target_verify(
            [],
            max_seqlen_k=8,
            dsa_index_topk=64,
            real_page_size=64,
            next_n=4,
            fill_ctx_lens=True,
        )
        self._check_draft_extend(
            [],
            [],
            max_seqlen_k=8,
            dsa_index_topk=64,
            real_page_size=64,
            max_extend_len=1,
            max_total_len=0,
            static_extend_len=True,
        )

    def test_large_shape_coverage(self):
        max_len = 1_000_003
        self._check_decode(
            [1_000_000, 999_983],
            max_len=max_len,
            dsa_index_topk=4096,
            real_page_size=64,
        )
        self._check_target_verify(
            [1_000_000],
            max_seqlen_k=max_len,
            dsa_index_topk=4096,
            real_page_size=64,
            next_n=2,
            fill_ctx_lens=True,
        )
        self._check_draft_extend(
            [1_000_000],
            [4],
            max_seqlen_k=max_len,
            dsa_index_topk=4096,
            real_page_size=64,
            max_extend_len=4,
            max_total_len=4,
            static_extend_len=True,
        )

    def test_large_batch_coverage(self):
        bs = 16 * 1024
        seq_lens = (torch.arange(bs, dtype=torch.int64) % 257 + 1).tolist()
        self._check_decode(
            seq_lens,
            max_len=1,
            dsa_index_topk=64,
            real_page_size=1,
        )
        self._check_target_verify(
            seq_lens,
            max_seqlen_k=1,
            dsa_index_topk=64,
            real_page_size=1,
            next_n=1,
            fill_ctx_lens=False,
        )
        self._check_draft_extend(
            seq_lens,
            [1] * bs,
            max_seqlen_k=1,
            dsa_index_topk=64,
            real_page_size=1,
            max_extend_len=1,
            max_total_len=bs,
            static_extend_len=True,
        )


if __name__ == "__main__":
    unittest.main()
