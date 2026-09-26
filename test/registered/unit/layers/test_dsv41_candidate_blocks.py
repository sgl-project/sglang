import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsv4.candidate_indexer import (
    PrefillCandidateBlocks,
    PrefillIndexerBudget,
    candidate_block_mask,
    select_candidate_block_ids,
    select_candidate_blocks,
)
from sglang.srt.layers.attention.dsv4.dense_prefill_indexer import _rows_per_chunk
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestPrefillCandidateBlocks(CustomTestCase):
    def test_causal_partial_blocks_and_forced_newest_block(self):
        scores = torch.tensor([[100.0] * 8 + [50.0] * 8 + [-10.0] * 3] * 4)
        lengths = torch.tensor([[0], [1], [9], [19]])
        scores.masked_fill_(torch.arange(19)[None, :] >= lengths, -torch.inf)
        original_scores = scores.clone()
        expected = torch.tensor(
            [
                [False] * 19,
                [True] * 8 + [False] * 11,
                [True] * 16 + [False] * 3,
                [True] * 8 + [False] * 8 + [True] * 3,
            ]
        )
        blocks = select_candidate_block_ids(
            logits=scores, compress_lens=lengths, topk_blocks=2, block_size=8
        )
        self.assertEqual(blocks.dtype, torch.int32)
        self.assertEqual(tuple(blocks.shape), (4, 2))
        torch.testing.assert_close(blocks[0], torch.tensor([-1, -1], dtype=torch.int32))
        torch.testing.assert_close(
            candidate_block_mask(blocks=blocks, width=19, block_size=8), expected
        )
        torch.testing.assert_close(
            select_candidate_blocks(
                logits=scores, compress_lens=lengths, topk_blocks=2, block_size=8
            ),
            expected,
        )
        torch.testing.assert_close(scores, original_scores)

    def test_underfilled_and_empty_candidates(self):
        for width in (0, 1, 7, 8, 9):
            with self.subTest(width=width):
                scores = torch.zeros((2, width))
                blocks = select_candidate_block_ids(
                    logits=scores, compress_lens=width, topk_blocks=2048, block_size=8
                )
                torch.testing.assert_close(
                    candidate_block_mask(blocks=blocks, width=width, block_size=8),
                    torch.ones_like(scores, dtype=torch.bool),
                )
        blocks = torch.full((3, 2), -1, dtype=torch.int32)
        self.assertFalse(
            candidate_block_mask(blocks=blocks, width=19, block_size=8).any()
        )

    def test_replay_tail_keeps_request_boundaries_and_empty_tails(self):
        requests = [torch.arange(n * 2).reshape(n, 2) for n in (5, 0, 3)]
        candidates = PrefillCandidateBlocks(request_blocks=requests)
        tail = candidates.tail([2, 0, 0])
        self.assertEqual(
            [tuple(b.shape) for b in tail.request_blocks], [(2, 2), (0, 2), (0, 2)]
        )
        torch.testing.assert_close(tail.request_blocks[0], requests[0][3:])
        self.assertEqual(tail.request_blocks[0].data_ptr(), requests[0][3:].data_ptr())
        self.assertEqual([b.shape[0] for b in candidates.request_blocks], [5, 0, 3])

    def test_nonfinite_blocks_match_mask_selection(self):
        logits = torch.tensor([[float("nan")] * 8 + [1.0] * 8 + [-torch.inf] * 8])
        kwargs = dict(logits=logits, compress_lens=24, topk_blocks=3, block_size=8)
        blocks = select_candidate_block_ids(**kwargs)
        torch.testing.assert_close(
            candidate_block_mask(blocks=blocks, width=24, block_size=8),
            select_candidate_blocks(**kwargs),
        )


class TestPrefillIndexerBudget(CustomTestCase):
    def test_free_memory_is_sampled_once_and_refreshed_next_forward(self):
        maybe_stub_sgl_kernel()
        from sglang.srt.layers.attention.deepseek_v4_backend import DSV4Metadata

        first = DSV4Metadata(core_attn_metadata=None, indexer_metadata=None)
        following = DSV4Metadata(core_attn_metadata=None, indexer_metadata=None)
        with (
            get_context().override_server_args(mem_fraction_static=0.75),
            envs.SGLANG_DSA_MQA_LOGITS_FREE_MEM_FRACTION.override(0.25),
            patch(
                "sglang.srt.layers.attention.mqa_logits_utils.get_device_module",
                return_value=torch.cuda,
            ),
            patch(
                "torch.cuda.get_device_properties",
                return_value=SimpleNamespace(total_memory=64 << 30),
            ),
            patch("torch.cuda.is_current_stream_capturing", return_value=False),
            patch("torch.cuda.mem_get_info", return_value=(32 << 30, 64 << 30)) as free,
        ):
            budget = first.prefill_indexer_budget
            args = dict(
                rows=16384, width=1048576, heads=32, device=torch.device("cuda:0")
            )
            self.assertEqual(_rows_per_chunk(**args, budget=budget), 1024)
            free.return_value = (4 << 30, 64 << 30)
            self.assertEqual(_rows_per_chunk(**args, budget=budget), 1024)
            self.assertEqual(
                _rows_per_chunk(**args, budget=following.prefill_indexer_budget), 256
            )
            self.assertEqual(free.call_count, 2)

    def test_capture_uses_static_budget_without_free_memory_query(self):
        with (
            get_context().override_server_args(mem_fraction_static=0.75),
            envs.SGLANG_DSA_MQA_LOGITS_FREE_MEM_FRACTION.override(0.25),
            patch(
                "sglang.srt.layers.attention.mqa_logits_utils.get_device_module",
                return_value=torch.cuda,
            ),
            patch(
                "torch.cuda.get_device_properties",
                return_value=SimpleNamespace(total_memory=64 << 30),
            ),
            patch("torch.cuda.is_current_stream_capturing", return_value=True),
            patch(
                "torch.cuda.mem_get_info", side_effect=AssertionError("capture sync")
            ),
        ):
            self.assertEqual(
                _rows_per_chunk(
                    16384,
                    1048576,
                    heads=32,
                    device=torch.device("cuda:0"),
                    budget=PrefillIndexerBudget(),
                ),
                1024,
            )

    def test_small_inputs_do_not_query_or_fix_the_forward_budget(self):
        budget = PrefillIndexerBudget()
        with patch("torch.cuda.mem_get_info", side_effect=AssertionError("small sync")):
            self.assertEqual(
                _rows_per_chunk(
                    31, 8192, heads=32, device=torch.device("cuda:0"), budget=budget
                ),
                31,
            )
        self.assertIsNone(budget.bytes)

    def test_padded_rows_and_columns_include_scratch(self):
        for limit, expected in (
            (45056, 13),
            (45055, 12),
            (24576, 8),
            (24575, 4),
            (1, 4),
        ):
            with self.subTest(limit=limit):
                self.assertEqual(
                    _rows_per_chunk(
                        13,
                        257,
                        heads=32,
                        device=torch.device("cuda:0"),
                        budget=PrefillIndexerBudget(bytes=limit),
                        scratch_row_bytes=512,
                        scratch_bytes=4096,
                    ),
                    expected,
                )


if __name__ == "__main__":
    unittest.main()
