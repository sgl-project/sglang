"""Correctness tests for the fused k-pool top-k candidate cache."""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase


register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@unittest.skipUnless(torch.cuda.is_available(), "Test requires CUDA")
class TestKpoolTopKTransform(CustomTestCase):
    def test_repeated_candidate_cache_overflow(self):
        from sglang.kernels.ops.moe.kpool_topk_transform import (
            fast_kpool_topk_transform_fused,
        )

        torch.manual_seed(2)
        batch_size, seq_len = 1, 16 * 1024
        pool_size, group_topk = 2, 512
        values = 1.0 + torch.linspace(
            0, 1e-4, seq_len, dtype=torch.float32, device="cuda"
        )
        score = values[torch.randperm(seq_len, device="cuda")].reshape(
            batch_size, seq_len
        )
        lengths = torch.full((batch_size,), seq_len, dtype=torch.int32, device="cuda")
        offsets = torch.zeros(batch_size, dtype=torch.int32, device="cuda")

        output = fast_kpool_topk_transform_fused(
            score=score,
            lengths=lengths,
            pool_size=pool_size,
            topk=group_topk * pool_size,
            topk_indices_offset=offsets,
        )
        # Each selected group expands to its two consecutive token positions.
        selected_groups = output[:, ::pool_size] // pool_size

        # Explicit structural checks: shape, range, no duplicates.
        assert selected_groups.shape == (batch_size, group_topk), (
            f"shape mismatch: {selected_groups.shape}"
        )
        assert (selected_groups >= 0).all(), "negative index found"
        assert (selected_groups < seq_len).all(), "index >= seq_len"
        sorted_row = selected_groups[0].sort().values
        dups = (sorted_row[1:] == sorted_row[:-1]).sum().item()
        assert dups == 0, f"{dups} duplicate indices"

        selected_values = (
            score.gather(1, selected_groups.long()).sort(dim=1, descending=True).values
        )
        reference_values = torch.topk(score, group_topk, dim=1, sorted=True).values
        torch.testing.assert_close(selected_values, reference_values, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
