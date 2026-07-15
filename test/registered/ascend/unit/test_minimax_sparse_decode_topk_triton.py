import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers.attention.minimax_sparse_ops.npu_triton import (
    flash_block_score_decode as score_decode,
)
from sglang.srt.layers.attention.minimax_sparse_ops.npu_triton import (
    topk_sparse_decode as sparse_decode,
)
from sglang.srt.layers.attention.minimax_sparse_ops.npu_triton import (
    prefill_block_score,
)
from sglang.srt.layers.attention.minimax_sparse_backend import (
    MiniMaxSparseAttnBackend,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=180, suite="nightly-1-npu-a3", nightly=True)

_DEVICE = "npu"
_BLOCK_SIZE = 128
_TOPK = 16
_NUM_Q_HEADS = 4
_NUM_KV_HEADS = 1
_HEAD_DIM = 128


def _build_inputs(seq_lens: list[int]) -> tuple[torch.Tensor, ...]:
    batch_size = len(seq_lens)
    max_blocks = max((seq_len + _BLOCK_SIZE - 1) // _BLOCK_SIZE for seq_len in seq_lens)
    blocks_per_request = [
        (seq_len + _BLOCK_SIZE - 1) // _BLOCK_SIZE for seq_len in seq_lens
    ]
    num_pages = sum(blocks_per_request)

    q = torch.randn(
        batch_size,
        _NUM_Q_HEADS,
        _HEAD_DIM,
        dtype=torch.bfloat16,
        device=_DEVICE,
    )
    k_cache = torch.randn(
        num_pages,
        _BLOCK_SIZE,
        _NUM_KV_HEADS,
        _HEAD_DIM,
        dtype=torch.bfloat16,
        device=_DEVICE,
    )
    block_table = torch.zeros(
        batch_size, max_blocks, dtype=torch.int32, device=_DEVICE
    )

    page_offset = 0
    for batch_idx, num_blocks in enumerate(blocks_per_request):
        block_table[batch_idx, :num_blocks] = torch.arange(
            page_offset, page_offset + num_blocks, dtype=torch.int32, device=_DEVICE
        )
        page_offset += num_blocks

    return q, k_cache, block_table, torch.tensor(
        seq_lens, dtype=torch.int32, device=_DEVICE
    )


def _reference_topk(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    score_type: str,
) -> torch.Tensor:
    batch_size = q.shape[0]
    max_blocks = block_table.shape[1]
    result = torch.full(
        (_NUM_Q_HEADS, batch_size, _TOPK),
        -1,
        dtype=torch.int32,
        device=q.device,
    )

    for batch_idx in range(batch_size):
        seq_len = int(seq_lens[batch_idx].item())
        num_blocks = (seq_len + _BLOCK_SIZE - 1) // _BLOCK_SIZE
        pages = block_table[batch_idx, :num_blocks].to(torch.long)
        keys = k_cache[pages, :, 0, :].reshape(
            num_blocks * _BLOCK_SIZE, _HEAD_DIM
        )
        qk = q[batch_idx].float() @ keys.float().transpose(0, 1)
        qk *= _HEAD_DIM**-0.5

        padded = torch.full(
            (_NUM_Q_HEADS, num_blocks * _BLOCK_SIZE),
            float("-inf"),
            dtype=torch.float32,
            device=q.device,
        )
        padded[:, :seq_len] = qk[:, :seq_len]
        block_scores = padded.view(_NUM_Q_HEADS, num_blocks, _BLOCK_SIZE)
        if score_type == "max":
            block_scores = block_scores.max(dim=-1).values
        else:
            block_max = block_scores.max(dim=-1, keepdim=True).values
            block_scores = block_max.squeeze(-1) + torch.log(
                torch.exp(block_scores - block_max).sum(dim=-1)
            )

        actual_topk = min(_TOPK, num_blocks)
        result[:, batch_idx, :actual_topk] = torch.topk(
            block_scores, k=actual_topk, dim=-1
        ).indices.to(torch.int32)

    return result


def _assert_topk_sets_equal(
    test_case: unittest.TestCase,
    actual: torch.Tensor,
    expected: torch.Tensor,
    seq_lens: torch.Tensor,
) -> None:
    for batch_idx in range(seq_lens.numel()):
        num_blocks = (int(seq_lens[batch_idx].item()) + _BLOCK_SIZE - 1) // _BLOCK_SIZE
        actual_topk = min(_TOPK, num_blocks)
        for head_idx in range(_NUM_Q_HEADS):
            test_case.assertEqual(
                set(actual[head_idx, batch_idx, :actual_topk].cpu().tolist()),
                set(expected[head_idx, batch_idx, :actual_topk].cpu().tolist()),
            )
            test_case.assertTrue(
                (actual[head_idx, batch_idx, actual_topk:] == -1).all().item()
            )


def _legacy_topk_reached(*args, **kwargs):
    raise AssertionError("legacy topk path reached")


def _reference_append_local_block(
    topk_idx: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    num_blocks: int,
) -> torch.Tensor:
    """Reference for the MiniMax decode init=0/local=1 fast path."""
    num_kv_heads, batch_size, topk = topk_idx.shape
    query_positions = (seq_lens.to(torch.long) - 1).clamp(min=0)
    local = (query_positions // block_size).clamp(min=0, max=num_blocks - 1).to(
        topk_idx.dtype
    )
    out = torch.full(
        (num_kv_heads, batch_size, topk + 1),
        -1,
        dtype=topk_idx.dtype,
        device=topk_idx.device,
    )
    for head_idx in range(num_kv_heads):
        for batch_idx in range(batch_size):
            candidates = topk_idx[head_idx, batch_idx]
            valid = (candidates >= 0) & (candidates < num_blocks)
            valid = valid & (candidates * block_size <= query_positions[batch_idx])
            out[head_idx, batch_idx, :topk] = torch.where(
                valid, candidates, torch.full_like(candidates, -1)
            )
            if not ((candidates == local[batch_idx]) & valid).any():
                out[head_idx, batch_idx, topk] = local[batch_idx]
    return out


def _direct_page_map_from_block_table(
    block_table: torch.Tensor,
    block_size: int,
    num_request_rows: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a fragmented req-to-token map for direct-page lookup tests."""
    batch_size, max_blocks = block_table.shape
    assert num_request_rows > batch_size

    req_to_token = torch.full(
        (num_request_rows, max_blocks * block_size),
        -1,
        dtype=torch.int32,
        device=block_table.device,
    )
    # Deliberately non-monotonic rows prove the kernel consumes request ids rather
    # than assuming the query batch index is the pool row.
    req_pool_indices = torch.arange(
        0, batch_size, dtype=torch.int32, device=block_table.device
    )
    req_pool_indices = (num_request_rows - 1) - req_pool_indices * 2

    for batch_idx in range(batch_size):
        req_row = int(req_pool_indices[batch_idx].item())
        for block_idx in range(max_blocks):
            req_to_token[req_row, block_idx * block_size] = (
                block_table[batch_idx, block_idx] * block_size
            )

    return req_to_token, req_pool_indices


class TestMiniMaxSparseDecodeTopKTriton(CustomTestCase):
    def test_prefill_metadata_keeps_direct_request_map_without_block_table(self):
        backend = object.__new__(MiniMaxSparseAttnBackend)
        backend._max_seqlen_k = 384
        backend._get_safe_block_size_q = lambda *_args: 1
        backend.req_to_token = torch.arange(
            4 * 512, dtype=torch.int32, device=_DEVICE
        ).view(4, 512)

        forward_batch = SimpleNamespace(
            req_pool_indices=torch.tensor([3, 1], dtype=torch.int32, device=_DEVICE),
            extend_seq_lens_cpu=[128, 128],
        )
        cu_seqlens = torch.tensor([0, 128, 256], dtype=torch.int32, device=_DEVICE)
        seq_lens = torch.tensor([384, 256], dtype=torch.int32, device=_DEVICE)
        prefix_lens = torch.tensor([256, 128], dtype=torch.int32, device=_DEVICE)
        cached_qblock_mappings = object()

        with patch.object(
            prefill_block_score,
            "_build_qblock_mappings",
            return_value=cached_qblock_mappings,
        ) as build_qblock_mappings:
            meta = backend._build_prefill_meta(
                forward_batch,
                cu_seqlens,
                seq_lens,
                prefix_lens,
                torch.device(_DEVICE),
                _BLOCK_SIZE,
                num_pages=16,
                total_q=256,
            )

        expected_req_map = torch.cat(
            (
                torch.full((128,), 3, dtype=torch.long, device=_DEVICE),
                torch.full((128,), 1, dtype=torch.long, device=_DEVICE),
            )
        )
        torch.testing.assert_close(meta.per_query_req, expected_req_map)
        self.assertFalse(hasattr(meta, "block_table_f"))
        self.assertIs(meta.qblock_mappings, cached_qblock_mappings)
        build_qblock_mappings.assert_called_once()

    def test_backend_uses_fused_local_append_only_for_m3_fast_path(self):
        backend = object.__new__(MiniMaxSparseAttnBackend)
        backend.topk_blocks = _TOPK
        backend.block_size_k = _BLOCK_SIZE
        topk_idx = torch.zeros((1, 2, _TOPK), dtype=torch.int32, device=_DEVICE)
        seq_lens = torch.tensor([512, 513], dtype=torch.int32, device=_DEVICE)
        fused = torch.full(
            (1, 2, _TOPK + 1), -1, dtype=torch.int32, device=_DEVICE
        )

        backend.init_blocks = 0
        backend.local_blocks = 1
        with patch.object(
            sparse_decode, "append_local_block_to_topk_idx", return_value=fused
        ) as append_local:
            actual = backend._prepare_npu_triton_topk_idx(
                topk_idx, seq_lens, num_idx_heads=1, num_kv_heads=1, max_blocks=5
            )

        self.assertIs(actual, fused)
        append_local.assert_called_once_with(topk_idx, seq_lens, _BLOCK_SIZE, 5)

        backend.init_blocks = 1
        backend.local_blocks = 1
        with patch.object(
            sparse_decode, "append_local_block_to_topk_idx"
        ) as append_local, patch.object(
            backend, "_merge_sparse_blocks", side_effect=lambda blocks, *_: blocks
        ):
            fallback = backend._prepare_npu_triton_topk_idx(
                topk_idx, seq_lens, num_idx_heads=1, num_kv_heads=1, max_blocks=5
            )

        append_local.assert_not_called()
        torch.testing.assert_close(fallback, topk_idx)

    def test_append_local_block_fused_matches_reference(self):
        topk_idx = torch.full(
            (2, 2, _TOPK), -1, dtype=torch.int32, device=_DEVICE
        )
        topk_idx[0, 0, :5] = torch.tensor(
            [1, 3, 4, -1, 0], dtype=torch.int32, device=_DEVICE
        )
        topk_idx[1, 0, :4] = torch.tensor(
            [1, 4, 0, -1], dtype=torch.int32, device=_DEVICE
        )
        topk_idx[0, 1, :5] = torch.tensor(
            [0, 2, 5, -1, 1], dtype=torch.int32, device=_DEVICE
        )
        topk_idx[1, 1, :5] = torch.tensor(
            [4, 0, 6, -1, 2], dtype=torch.int32, device=_DEVICE
        )
        seq_lens = torch.tensor([512, 513], dtype=torch.int32, device=_DEVICE)

        actual = sparse_decode.append_local_block_to_topk_idx(
            topk_idx, seq_lens, block_size=_BLOCK_SIZE, num_blocks=5
        )
        expected = _reference_append_local_block(
            topk_idx, seq_lens, block_size=_BLOCK_SIZE, num_blocks=5
        )

        self.assertEqual(actual.shape, (2, 2, _TOPK + 1))
        self.assertTrue(actual.is_contiguous())
        torch.testing.assert_close(actual, expected)

    def test_unified_triton_topk_matches_reference_for_serving_contexts(self):
        torch.manual_seed(20260714)
        cases = (
            ("max", [16384, 16257]),
            ("lse", [32768, 32641]),
            ("max", [131072]),
        )

        with patch.object(
            score_decode, "_streaming_topk_from_score", _legacy_topk_reached
        ), patch.object(score_decode, "_torch_topk_from_score", _legacy_topk_reached):
            for score_type, lengths in cases:
                with self.subTest(score_type=score_type, lengths=lengths):
                    q, k_cache, block_table, seq_lens = _build_inputs(lengths)
                    _, actual = score_decode.flash_decode_bnsd_with_topk_idx(
                        q=q,
                        sink=None,
                        k_cache_bnsd=k_cache,
                        v_cache_bnsd=None,
                        block_table=block_table,
                        seq_lens=seq_lens,
                        max_seqlen=max(lengths),
                        block_size=_BLOCK_SIZE,
                        topk=_TOPK,
                        init_blocks=0,
                        local_blocks=0,
                        score_type=score_type,
                        disable_index_value=True,
                    )
                    expected = _reference_topk(
                        q, k_cache, block_table, seq_lens, score_type
                    )
                    _assert_topk_sets_equal(self, actual, expected, seq_lens)

    def test_unified_triton_topk_for_index_value_path(self):
        torch.manual_seed(20260715)
        lengths = [16384, 16257]
        q, k_cache, block_table, seq_lens = _build_inputs(lengths)
        v_cache = torch.randn(
            k_cache.shape,
            dtype=k_cache.dtype,
            device=k_cache.device,
        )

        with patch.object(
            score_decode, "_streaming_topk_from_score", _legacy_topk_reached
        ), patch.object(score_decode, "_torch_topk_from_score", _legacy_topk_reached):
            _, actual = score_decode.flash_decode_bnsd_with_topk_idx(
                q=q,
                sink=None,
                k_cache_bnsd=k_cache,
                v_cache_bnsd=v_cache,
                block_table=block_table,
                seq_lens=seq_lens,
                max_seqlen=max(lengths),
                block_size=_BLOCK_SIZE,
                topk=_TOPK,
                init_blocks=0,
                local_blocks=0,
                score_type="max",
                disable_index_value=False,
            )

        expected = _reference_topk(q, k_cache, block_table, seq_lens, "max")
        _assert_topk_sets_equal(self, actual, expected, seq_lens)

    def test_direct_page_lookup_topk_matches_block_table_reference(self):
        torch.manual_seed(20260715)
        lengths = [16384, 16257]
        q, k_cache, block_table, seq_lens = _build_inputs(lengths)
        req_to_token, req_pool_indices = _direct_page_map_from_block_table(
            block_table, _BLOCK_SIZE, num_request_rows=5
        )

        _, actual = score_decode.flash_decode_bnsd_with_topk_idx(
            q=q,
            sink=None,
            k_cache_bnsd=k_cache,
            v_cache_bnsd=None,
            block_table=None,
            req_to_token=req_to_token,
            req_pool_indices=req_pool_indices,
            max_num_blocks=block_table.shape[1],
            num_pages=k_cache.shape[0],
            sanitize_page_ids=True,
            seq_lens=seq_lens,
            max_seqlen=max(lengths),
            block_size=_BLOCK_SIZE,
            topk=_TOPK,
            init_blocks=0,
            local_blocks=0,
            score_type="max",
            disable_index_value=True,
        )

        expected = _reference_topk(q, k_cache, block_table, seq_lens, "max")
        _assert_topk_sets_equal(self, actual, expected, seq_lens)

    def test_direct_page_lookup_gqa_follows_current_request_indices(self):
        torch.manual_seed(20260716)
        q, k_cache, block_table, seq_lens = _build_inputs([512, 512])
        v_cache = torch.randn_like(k_cache)
        req_to_token, req_pool_indices = _direct_page_map_from_block_table(
            block_table, _BLOCK_SIZE, num_request_rows=5
        )
        topk_idx = torch.tensor(
            [[[0, 1, 2, 3], [0, 1, 2, 3]]],
            dtype=torch.int32,
            device=_DEVICE,
        )

        expected = sparse_decode.flash_decode_bnsd_with_gqa_share_sparse(
            q=q,
            sink=None,
            k_cache_bnsd=k_cache,
            v_cache_bnsd=v_cache,
            block_table=block_table,
            seq_lens=seq_lens,
            block_size=_BLOCK_SIZE,
            topk_idx=topk_idx,
        )
        actual = sparse_decode.flash_decode_bnsd_with_gqa_share_sparse(
            q=q,
            sink=None,
            k_cache_bnsd=k_cache,
            v_cache_bnsd=v_cache,
            block_table=None,
            req_to_token=req_to_token,
            req_pool_indices=req_pool_indices,
            max_num_blocks=block_table.shape[1],
            num_pages=k_cache.shape[0],
            sanitize_page_ids=True,
            seq_lens=seq_lens,
            block_size=_BLOCK_SIZE,
            topk_idx=topk_idx,
        )
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)

        swapped_indices = req_pool_indices.flip(0).contiguous()
        expected_swapped = sparse_decode.flash_decode_bnsd_with_gqa_share_sparse(
            q=q,
            sink=None,
            k_cache_bnsd=k_cache,
            v_cache_bnsd=v_cache,
            block_table=block_table.flip(0).contiguous(),
            seq_lens=seq_lens,
            block_size=_BLOCK_SIZE,
            topk_idx=topk_idx,
        )
        actual_swapped = sparse_decode.flash_decode_bnsd_with_gqa_share_sparse(
            q=q,
            sink=None,
            k_cache_bnsd=k_cache,
            v_cache_bnsd=v_cache,
            block_table=None,
            req_to_token=req_to_token,
            req_pool_indices=swapped_indices,
            max_num_blocks=block_table.shape[1],
            num_pages=k_cache.shape[0],
            sanitize_page_ids=True,
            seq_lens=seq_lens,
            block_size=_BLOCK_SIZE,
            topk_idx=topk_idx,
        )
        torch.testing.assert_close(
            actual_swapped, expected_swapped, rtol=2e-2, atol=2e-2
        )

    def test_backend_decode_and_verify_route_direct_page_metadata(self):
        backend = object.__new__(MiniMaxSparseAttnBackend)
        backend.page_size = _BLOCK_SIZE
        backend._max_seqlen_k = 512
        backend.topk_blocks = _TOPK
        backend.score_type = "max"
        backend.req_to_token = torch.arange(
            3 * 512, dtype=torch.int32, device=_DEVICE
        ).view(3, 512)
        backend._prepare_npu_triton_topk_idx = MagicMock(
            side_effect=lambda topk_idx, *_: topk_idx
        )

        q = torch.randn(1, _NUM_Q_HEADS, _HEAD_DIM, dtype=torch.bfloat16, device=_DEVICE)
        k_cache = torch.randn(
            4, _BLOCK_SIZE, _NUM_KV_HEADS, _HEAD_DIM,
            dtype=torch.bfloat16, device=_DEVICE,
        )
        v_cache = torch.randn_like(k_cache)
        idx_q = torch.randn_like(q)
        idx_k_cache = torch.randn_like(k_cache)
        forward_batch = SimpleNamespace(
            seq_lens=torch.tensor([512], dtype=torch.int32, device=_DEVICE),
            req_pool_indices=torch.tensor([2], dtype=torch.int32, device=_DEVICE),
        )
        topk_decode = torch.zeros((1, 1, _TOPK), dtype=torch.int32, device=_DEVICE)

        with patch.object(
            score_decode, "flash_decode_bnsd_with_topk_idx", return_value=(None, topk_decode)
        ) as score_call, patch.object(
            sparse_decode, "flash_decode_bnsd_with_gqa_share_sparse", return_value=q
        ) as gqa_call:
            backend._forward_npu_triton_decode(
                q, k_cache, v_cache, idx_q, idx_k_cache, None, forward_batch
            )

        for call in (score_call, gqa_call):
            kwargs = call.call_args.kwargs
            self.assertIsNone(kwargs["block_table"])
            self.assertIs(kwargs["req_to_token"], backend.req_to_token)
            torch.testing.assert_close(
                kwargs["req_pool_indices"], forward_batch.req_pool_indices
            )
            self.assertEqual(kwargs["max_num_blocks"], 4)
            self.assertEqual(kwargs["num_pages"], 4)
            self.assertFalse(kwargs["sanitize_page_ids"])

        verify_q = torch.randn(
            2, _NUM_Q_HEADS, _HEAD_DIM, dtype=torch.bfloat16, device=_DEVICE
        )
        verify_idx_q = torch.randn_like(verify_q)
        topk_verify = torch.zeros((1, 2, _TOPK), dtype=torch.int32, device=_DEVICE)
        backend._prepare_npu_triton_topk_idx.reset_mock()
        with patch.object(
            score_decode, "flash_decode_bnsd_with_topk_idx", return_value=(None, topk_verify)
        ) as score_call, patch.object(
            sparse_decode,
            "flash_decode_bnsd_with_gqa_share_sparse",
            return_value=verify_q,
        ) as gqa_call:
            backend._forward_npu_triton_verify(
                verify_q,
                k_cache,
                v_cache,
                verify_idx_q,
                idx_k_cache,
                None,
                forward_batch,
                prefix_lens=torch.tensor([510], dtype=torch.int32, device=_DEVICE),
            )

        expected_verify_rows = torch.tensor([2, 2], dtype=torch.int64, device=_DEVICE)
        for call in (score_call, gqa_call):
            kwargs = call.call_args.kwargs
            self.assertIsNone(kwargs["block_table"])
            self.assertIs(kwargs["req_to_token"], backend.req_to_token)
            torch.testing.assert_close(kwargs["req_pool_indices"], expected_verify_rows)
            self.assertEqual(kwargs["max_num_blocks"], 4)
            self.assertEqual(kwargs["num_pages"], 4)
            self.assertTrue(kwargs["sanitize_page_ids"])


if __name__ == "__main__":
    unittest.main()
