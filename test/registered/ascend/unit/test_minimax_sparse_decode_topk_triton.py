import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.minimax_sparse_ops.npu_triton import (
    flash_block_score_decode as score_decode,
)
from sglang.srt.layers.attention.minimax_sparse_ops.npu_triton import (
    topk_sparse_decode as sparse_decode,
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


class TestMiniMaxSparseDecodeTopKTriton(CustomTestCase):
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


if __name__ == "__main__":
    unittest.main()
