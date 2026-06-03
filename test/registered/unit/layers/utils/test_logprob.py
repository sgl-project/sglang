# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC

"""Unit tests for srt/layers/utils/logprob.py — CPU-only, no server."""

from sglang.test.ci.ci_register import register_cpu_ci

# Register CPU CI with estimated time and suite
register_cpu_ci(est_time=2, suite="base-a-test-cpu")

import unittest
from unittest.mock import MagicMock
import torch

from sglang.srt.layers.utils.logprob import (
    LogprobStage,
    get_top_logprobs_raw,
    get_token_ids_logprobs_raw,
    get_top_logprobs_chunk,
    get_token_ids_logprobs_chunk,
)
from sglang.test.test_utils import CustomTestCase


class TestLogprobUtils(CustomTestCase):

    def test_get_top_logprobs_raw_decode(self) -> None:
        # 2 sequences, vocab size 5
        logprobs = torch.tensor([
            [0.1, 0.5, 0.2, 0.1, 0.1],
            [0.2, 0.1, 0.3, 0.4, 0.0]
        ])
        # Sequence 1 wants top 2, Sequence 2 wants top 3
        top_logprobs_nums = [2, 3]
        
        val, idx = get_top_logprobs_raw(
            logprobs, top_logprobs_nums, stage=LogprobStage.DECODE
        )
        
        self.assertEqual(len(val), 2)
        self.assertEqual(len(idx), 2)
        
        # Test values (approximate due to float)
        self.assertAlmostEqual(val[0][0], 0.5)
        self.assertAlmostEqual(val[0][1], 0.2)
        self.assertEqual(idx[0], [1, 2])
        
        self.assertAlmostEqual(val[1][0], 0.4)
        self.assertAlmostEqual(val[1][1], 0.3)
        self.assertAlmostEqual(val[1][2], 0.2)
        self.assertEqual(idx[1], [3, 2, 0])

    def test_get_top_logprobs_raw_prefill(self) -> None:
        # Flat logprobs for prefill: 3 tokens total (Seq 0 has 2 tokens, Seq 1 has 0 tokens, Seq 2 has 1 token)
        logprobs = torch.tensor([
            [0.9, 0.1, 0.0, 0.0, 0.0], # Seq 0, token 0
            [0.2, 0.8, 0.0, 0.0, 0.0], # Seq 0, token 1
            [0.3, 0.3, 0.4, 0.0, 0.0]  # Seq 2, token 0
        ])
        # Seq 0 wants top 1, Seq 1 wants top 5 (but has 0 tokens), Seq 2 wants top 2
        top_logprobs_nums = [1, 5, 2]
        extend_logprob_pruned_lens_cpu = [2, 0, 1]
        
        val, idx = get_top_logprobs_raw(
            logprobs,
            top_logprobs_nums,
            stage=LogprobStage.PREFILL,
            extend_logprob_pruned_lens_cpu=extend_logprob_pruned_lens_cpu
        )
        
        self.assertEqual(len(val), 3)
        self.assertEqual(len(val[0]), 2) # Seq 0 has 2 tokens
        self.assertEqual(len(val[1]), 0) # Seq 1 has 0 tokens (empty list)
        self.assertEqual(len(val[2]), 1) # Seq 2 has 1 token
        
        # Seq 0
        self.assertAlmostEqual(val[0][0][0], 0.9)
        self.assertEqual(idx[0][0], [0])
        self.assertAlmostEqual(val[0][1][0], 0.8)
        self.assertEqual(idx[0][1], [1])
        
        # Seq 1
        self.assertEqual(val[1], [])
        self.assertEqual(idx[1], [])
        
        # Seq 2
        self.assertAlmostEqual(val[2][0][0], 0.4)
        self.assertEqual(idx[2][0][0], 2)
        self.assertEqual(len(idx[2][0]), 2)

    def test_get_token_ids_logprobs_raw_decode(self) -> None:
        logprobs = torch.tensor([
            [0.1, 0.5, 0.2, 0.1, 0.1],
            [0.2, 0.1, 0.3, 0.4, 0.0]
        ])
        # Seq 0 wants logprob of token 1 and 3. Seq 1 wants logprob of token 0 and 2.
        token_ids_logprobs_list = [[1, 3], [0, 2]]
        
        vals, idxs = get_token_ids_logprobs_raw(
            logprobs, token_ids_logprobs_list, stage=LogprobStage.DECODE
        )
        
        self.assertEqual(idxs, token_ids_logprobs_list)
        self.assertAlmostEqual(vals[0][0], 0.5)
        self.assertAlmostEqual(vals[0][1], 0.1)
        self.assertAlmostEqual(vals[1][0], 0.2)
        self.assertAlmostEqual(vals[1][1], 0.3)

    def test_get_token_ids_logprobs_raw_prefill(self) -> None:
        # Flat logprobs for prefill: 3 tokens total (Seq 0 has 2 tokens, Seq 1 has 0 tokens, Seq 2 has 1 token)
        logprobs = torch.tensor([
            [0.1, 0.5, 0.2, 0.1, 0.1], # Seq 0, token 0
            [0.2, 0.1, 0.3, 0.4, 0.0], # Seq 0, token 1
            [0.8, 0.1, 0.0, 0.0, 0.1]  # Seq 2, token 0
        ])
        
        # Seq 0 wants logprobs for [1, 3], Seq 1 wants [5, 6] (but has 0 tokens), Seq 2 wants [0, 4]
        token_ids_logprobs_list = [[1, 3], [5, 6], [0, 4]]
        extend_logprob_pruned_lens_cpu = [2, 0, 1]
        
        vals, idxs = get_token_ids_logprobs_raw(
            logprobs,
            token_ids_logprobs_list,
            stage=LogprobStage.PREFILL,
            extend_logprob_pruned_lens_cpu=extend_logprob_pruned_lens_cpu
        )
        
        self.assertEqual(len(vals), 3)
        self.assertEqual(len(vals[0]), 2) # 2 tokens for Seq 0
        self.assertEqual(len(vals[1]), 0) # 0 tokens for Seq 1
        self.assertEqual(len(vals[2]), 1) # 1 token for Seq 2
        
        # Seq 0
        self.assertAlmostEqual(vals[0][0][0], 0.5)
        self.assertAlmostEqual(vals[0][0][1], 0.1)
        self.assertAlmostEqual(vals[0][1][0], 0.1)
        self.assertAlmostEqual(vals[0][1][1], 0.4)
        
        # Seq 1
        self.assertEqual(vals[1], [])
        self.assertEqual(idxs[1], [])
        
        # Seq 2
        self.assertAlmostEqual(vals[2][0][0], 0.8)
        self.assertAlmostEqual(vals[2][0][1], 0.1)
        
        self.assertEqual(idxs[0], [[1, 3], [1, 3]])
        self.assertEqual(idxs[2], [[0, 4]])

    def test_get_top_logprobs_chunk(self) -> None:
        # Shape: [3, 5] (3 tokens in this chunk: Seq 0 has 2, Seq 1 has 0, Seq 2 has 1)
        logprobs = torch.tensor([
            [0.1, 0.5, 0.2, 0.1, 0.1], # Seq 0, token 0
            [0.2, 0.1, 0.3, 0.4, 0.0], # Seq 0, token 1
            [0.8, 0.1, 0.0, 0.0, 0.1]  # Seq 2, token 0
        ])
        
        # Mock logits_metadata
        logits_metadata = MagicMock()
        logits_metadata.top_logprobs_nums = [2, 5, 1]
        
        top_k_nums = [2, 5, 1]
        pruned_lens = [2, 0, 1]
        
        input_top_logprobs_val = []
        input_top_logprobs_idx = []
        
        next_split = get_top_logprobs_chunk(
            logprobs,
            logits_metadata,
            top_k_nums,
            pruned_lens,
            input_top_logprobs_val,
            input_top_logprobs_idx,
            split_pruned_len=0
        )
        
        self.assertEqual(next_split, 0)
        self.assertEqual(len(input_top_logprobs_val), 3)
        self.assertEqual(len(input_top_logprobs_val[0]), 2) # Seq 0
        self.assertEqual(len(input_top_logprobs_val[1]), 0) # Seq 1 (empty list)
        self.assertEqual(len(input_top_logprobs_val[2]), 1) # Seq 2
        
        self.assertEqual(input_top_logprobs_idx[0][0], [1, 2])
        self.assertEqual(input_top_logprobs_idx[0][1], [3, 2])
        self.assertEqual(input_top_logprobs_idx[1], [])
        self.assertEqual(input_top_logprobs_idx[2][0], [0])

    def test_get_token_ids_logprobs_chunk(self) -> None:
        # Shape: [3, 5] (3 tokens in this chunk: Seq 0 has 2, Seq 1 has 0, Seq 2 has 1)
        logprobs = torch.tensor([
            [0.1, 0.5, 0.2, 0.1, 0.1], # Seq 0, token 0
            [0.2, 0.1, 0.3, 0.4, 0.0], # Seq 0, token 1
            [0.8, 0.1, 0.0, 0.0, 0.1]  # Seq 2, token 0
        ])
        
        token_ids_logprobs = [[1, 3], [5, 6], [0, 4]]
        pruned_lens = [2, 0, 1]
        
        input_token_ids_logprobs_val = []
        input_token_ids_logprobs_idx = []
        
        next_split = get_token_ids_logprobs_chunk(
            logprobs,
            token_ids_logprobs,
            pruned_lens,
            input_token_ids_logprobs_val,
            input_token_ids_logprobs_idx,
            split_pruned_len=0
        )
        
        self.assertEqual(next_split, 0)
        self.assertEqual(len(input_token_ids_logprobs_val), 3)
        self.assertEqual(len(input_token_ids_logprobs_val[0]), 2) # Seq 0
        self.assertEqual(len(input_token_ids_logprobs_val[1]), 0) # Seq 1 (empty list)
        self.assertEqual(len(input_token_ids_logprobs_val[2]), 1) # Seq 2
        
        self.assertAlmostEqual(input_token_ids_logprobs_val[0][0][0], 0.5)
        self.assertAlmostEqual(input_token_ids_logprobs_val[0][0][1], 0.1)
        self.assertEqual(input_token_ids_logprobs_idx[0][0], [1, 3])
        self.assertEqual(input_token_ids_logprobs_idx[1], [])
        self.assertEqual(input_token_ids_logprobs_idx[2][0], [0, 4])


if __name__ == "__main__":
    unittest.main()
