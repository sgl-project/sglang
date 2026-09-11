"""CUDA correctness tests for the fused suffix-attention merge."""

import math
import unittest

import torch

from sglang.kernels.ops.attention.suffix_attention_merge import (
    merge_suffix_attention_in_place,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(
    est_time=15,
    stage="base-b-kernel-unit",
    runner_config="1-gpu-large",
)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestSuffixAttentionMerge(CustomTestCase):
    def _case(self, *, num_queries: int, head_dim: int, dtype: torch.dtype):
        torch.manual_seed(7)
        device = torch.device("cuda")
        num_q_heads = 8
        num_kv_heads = 2
        num_slots = 2 * num_queries + 11

        q = torch.randn(num_queries, num_q_heads, head_dim, device=device, dtype=dtype)
        k_cache = torch.randn(
            num_slots, num_kv_heads, head_dim, device=device, dtype=dtype
        )
        v_cache = torch.randn_like(k_cache)
        page_table = torch.stack(
            [
                torch.randperm(num_slots, device=device)[:num_queries]
                for _ in range(num_queries)
            ]
        ).to(torch.int32)
        suffix_lengths = (
            torch.arange(num_queries, device=device, dtype=torch.int32)
            .remainder(num_queries)
            .add_(1)
        )
        prefix = torch.randn_like(q)
        prefix_lse = torch.randn(
            num_q_heads, num_queries, device=device, dtype=torch.float32
        )
        scale = 1.0 / math.sqrt(head_dim)

        reference = prefix.float().clone()
        heads_per_kv = num_q_heads // num_kv_heads
        kv_heads = torch.arange(num_q_heads, device=device) // heads_per_kv
        for token in range(num_queries):
            length = int(suffix_lengths[token])
            slots = page_table[token, :length].long()
            keys = k_cache[slots][:, kv_heads].float()
            values = v_cache[slots][:, kv_heads].float()
            scores = torch.einsum("lhd,hd->lh", keys, q[token].float()) * scale
            maximum = torch.maximum(prefix_lse[:, token], scores.max(dim=0).values)
            prefix_weight = torch.exp(prefix_lse[:, token] - maximum)
            suffix_weights = torch.exp(scores - maximum)
            reference[token] = (
                reference[token] * prefix_weight[:, None]
                + torch.einsum("lh,lhd->hd", suffix_weights, values)
            ) / (prefix_weight + suffix_weights.sum(dim=0))[:, None]

        static_prefix = prefix.clone()
        merge_suffix_attention_in_place(
            q,
            k_cache,
            v_cache,
            page_table,
            suffix_lengths,
            static_prefix,
            prefix_lse,
            scale,
        )
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_prefix.copy_(prefix)
            merge_suffix_attention_in_place(
                q,
                k_cache,
                v_cache,
                page_table,
                suffix_lengths,
                static_prefix,
                prefix_lse,
                scale,
            )
        graph.replay()
        torch.cuda.synchronize()

        torch.testing.assert_close(
            static_prefix.float(), reference, rtol=2e-2, atol=2e-2
        )

    def test_representative_shapes(self):
        cases = (
            (16, 64, torch.float16),
            (60, 128, torch.bfloat16),
        )
        for num_queries, head_dim, dtype in cases:
            with self.subTest(
                num_queries=num_queries,
                head_dim=head_dim,
                dtype=dtype,
            ):
                self._case(
                    num_queries=num_queries,
                    head_dim=head_dim,
                    dtype=dtype,
                )


if __name__ == "__main__":
    unittest.main()
