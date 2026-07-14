import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.minimax_sparse_ops.npu_triton import (
    flash_block_score_decode as score_decode,
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


class TestMiniMaxSparseDecodeTopKTriton(CustomTestCase):
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
