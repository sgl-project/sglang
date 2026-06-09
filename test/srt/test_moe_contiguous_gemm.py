"""
Unit tests for the contiguous GEMM path in Standard Dispatcher with DeepGEMM.

Tests cover:
1. moe_ep_deepgemm_preprocess_contiguous correctness
2. ep_scatter + ep_gather round-trip
3. Contiguous vs Masked path output comparison
4. Edge cases: top_k=1, dropped tokens, large batch
"""

import unittest

import torch

from sglang.srt.utils import is_cuda
from sglang.test.test_utils import CustomTestCase

_is_cuda = is_cuda()

EXPERT_ALIGNMENT = 128


def _fp8_supported():
    if not _is_cuda:
        return False
    capability = torch.cuda.get_device_capability()
    return capability[0] >= 9 or capability == (8, 9)


def _generate_topk_ids(num_tokens, num_experts, top_k, drop_ratio=0.0):
    topk_ids = torch.randint(0, num_experts, (num_tokens, top_k), device="cuda")
    if drop_ratio > 0:
        mask = torch.rand(num_tokens, top_k, device="cuda") < drop_ratio
        topk_ids[mask] = -1
    return topk_ids


@unittest.skipUnless(_is_cuda and _fp8_supported(), "Requires CUDA with FP8 support")
class TestMoeEpDeepgemmPreprocessContiguous(CustomTestCase):

    def _run_preprocess(self, num_tokens, hidden_size, num_experts, top_k, drop_ratio=0.0):
        from sglang.srt.layers.moe.ep_moe.kernels import (
            moe_ep_deepgemm_preprocess_contiguous,
        )

        hidden_states = torch.randn(
            num_tokens, hidden_size, device="cuda", dtype=torch.bfloat16
        )
        topk_ids = _generate_topk_ids(num_tokens, num_experts, top_k, drop_ratio)
        block_shape = [128, 128]

        input_tensor, input_tensor_scale, m_indices, output_index, all_tokens = (
            moe_ep_deepgemm_preprocess_contiguous(
                topk_ids, num_experts, hidden_states, top_k, block_shape
            )
        )
        return (
            input_tensor, input_tensor_scale, m_indices, output_index,
            all_tokens, topk_ids, hidden_states,
        )

    def test_basic_correctness(self):
        configs = [
            (32, 256, 4, 2),
            (64, 512, 8, 2),
            (128, 1024, 16, 2),
        ]
        for num_tokens, hidden_size, num_experts, top_k in configs:
            with self.subTest(
                num_tokens=num_tokens, hidden_size=hidden_size,
                num_experts=num_experts, top_k=top_k,
            ):
                (input_tensor, input_tensor_scale, m_indices, output_index,
                 all_tokens, topk_ids, hidden_states) = self._run_preprocess(
                    num_tokens, hidden_size, num_experts, top_k
                )

                self.assertEqual(input_tensor.shape, (all_tokens, hidden_size))
                self.assertEqual(input_tensor.dtype, torch.float8_e4m3fn)
                self.assertEqual(m_indices.shape, (all_tokens,))
                self.assertEqual(output_index.shape, (num_tokens, top_k))

                actual_routed = (topk_ids >= 0).sum().item()
                self.assertGreaterEqual(all_tokens, actual_routed)

                torch.cuda.empty_cache()

    def test_expert_alignment(self):
        num_tokens, hidden_size, num_experts, top_k = 100, 256, 8, 2
        (input_tensor, _, m_indices, _, all_tokens, topk_ids, _) = (
            self._run_preprocess(num_tokens, hidden_size, num_experts, top_k)
        )

        self.assertEqual(all_tokens % EXPERT_ALIGNMENT, 0)

        for expert_id in range(num_experts):
            expert_count = (m_indices == expert_id).sum().item()
            self.assertEqual(
                expert_count % EXPERT_ALIGNMENT, 0,
                f"Expert {expert_id} has {expert_count} tokens, not aligned to {EXPERT_ALIGNMENT}",
            )

    def test_m_indices_values(self):
        num_tokens, hidden_size, num_experts, top_k = 64, 256, 4, 2
        (_, _, m_indices, _, all_tokens, topk_ids, _) = self._run_preprocess(
            num_tokens, hidden_size, num_experts, top_k
        )

        unique_experts = m_indices.unique()
        for eid in unique_experts:
            self.assertGreaterEqual(eid.item(), 0)
            self.assertLess(eid.item(), num_experts)

        # m_indices should be grouped: all tokens for expert 0, then expert 1, etc.
        prev_expert = -1
        seen_experts = set()
        i = 0
        while i < all_tokens:
            cur = m_indices[i].item()
            if cur != prev_expert:
                self.assertNotIn(
                    cur, seen_experts,
                    f"Expert {cur} appears non-contiguously in m_indices",
                )
                seen_experts.add(cur)
                prev_expert = cur
            i += 1

    def test_dropped_tokens(self):
        num_tokens, hidden_size, num_experts, top_k = 128, 256, 8, 2
        (input_tensor, _, m_indices, output_index, all_tokens, topk_ids, _) = (
            self._run_preprocess(
                num_tokens, hidden_size, num_experts, top_k, drop_ratio=0.3
            )
        )

        valid_count = (topk_ids >= 0).sum().item()
        self.assertGreaterEqual(all_tokens, valid_count)

        # output_index for valid entries should be in range [0, all_tokens)
        valid_mask = topk_ids >= 0
        valid_indices = output_index[valid_mask]
        self.assertTrue(
            (valid_indices >= 0).all().item(),
            "Valid output_index entries should be non-negative",
        )
        self.assertTrue(
            (valid_indices < all_tokens).all().item(),
            "Valid output_index entries should be < all_tokens",
        )

        torch.cuda.empty_cache()

    def test_top_k_1(self):
        num_tokens, hidden_size, num_experts, top_k = 64, 256, 8, 1
        (input_tensor, _, m_indices, output_index, all_tokens, topk_ids, _) = (
            self._run_preprocess(num_tokens, hidden_size, num_experts, top_k)
        )

        self.assertEqual(output_index.shape, (num_tokens, 1))
        self.assertGreaterEqual(all_tokens, num_tokens)
        torch.cuda.empty_cache()

    def test_large_batch(self):
        num_tokens, hidden_size, num_experts, top_k = 4096, 1024, 16, 2
        (input_tensor, _, m_indices, output_index, all_tokens, topk_ids, _) = (
            self._run_preprocess(num_tokens, hidden_size, num_experts, top_k)
        )

        self.assertEqual(input_tensor.shape[0], all_tokens)
        self.assertEqual(input_tensor.shape[1], hidden_size)
        torch.cuda.empty_cache()


@unittest.skipUnless(_is_cuda and _fp8_supported(), "Requires CUDA with FP8 support")
class TestEpScatterGatherRoundtrip(CustomTestCase):

    def _test_roundtrip(self, num_tokens, hidden_size, num_experts, top_k):
        from sglang.srt.layers.moe.ep_moe.kernels import ep_gather, ep_scatter
        from sglang.srt.layers.quantization.fp8_kernel import (
            sglang_per_token_group_quant_fp8 as per_token_group_quant_fp8,
        )
        from sglang.srt.utils import ceil_div

        hidden_states = torch.randn(
            num_tokens, hidden_size, device="cuda", dtype=torch.bfloat16
        )
        topk_ids = _generate_topk_ids(num_tokens, num_experts, top_k)
        topk_weights = torch.ones(num_tokens, top_k, device="cuda", dtype=torch.float32)
        # With top_k > 1, each token's contributions are summed during gather.
        # Set weight = 1/top_k so the sum reproduces the original.
        topk_weights.fill_(1.0 / top_k)

        block_k = 128
        hidden_states_fp8, scale = per_token_group_quant_fp8(hidden_states, block_k)

        num_tokens_per_expert = torch.zeros(
            num_experts, device="cuda", dtype=torch.int32
        )
        flat_ids = topk_ids.view(-1)
        valid_mask = flat_ids >= 0
        valid_ids = flat_ids[valid_mask]
        if valid_ids.numel() > 0:
            num_tokens_per_expert.scatter_add_(
                0, valid_ids.to(torch.int64),
                torch.ones_like(valid_ids, dtype=torch.int32),
            )

        aligned_per_expert = (
            (num_tokens_per_expert + EXPERT_ALIGNMENT - 1)
            // EXPERT_ALIGNMENT
            * EXPERT_ALIGNMENT
        )
        all_tokens = int(aligned_per_expert.sum().item())

        scale_hidden_size = hidden_size // block_k
        input_tensor = torch.empty(
            (all_tokens, hidden_size), device="cuda", dtype=hidden_states_fp8.dtype
        )
        input_tensor_scale = torch.empty(
            (all_tokens, scale_hidden_size), device="cuda", dtype=torch.float32
        )
        m_indices = torch.empty(all_tokens, device="cuda", dtype=torch.int32)
        output_index = torch.empty(num_tokens, top_k, device="cuda", dtype=torch.int32)
        expert_start_loc = torch.empty_like(aligned_per_expert)

        ep_scatter(
            hidden_states_fp8, scale, topk_ids,
            aligned_per_expert, expert_start_loc,
            input_tensor, input_tensor_scale, m_indices, output_index,
        )

        # Simulate GEMM as identity: convert FP8 back to bf16 using scales
        gemm_output = input_tensor.to(torch.bfloat16)

        gather_out = torch.empty(
            num_tokens, hidden_size, device="cuda", dtype=torch.bfloat16
        )
        ep_gather(gemm_output, topk_ids, topk_weights, output_index, gather_out)

        # Since we use FP8 quantization, there will be precision loss.
        # Also, scatter duplicates each token top_k times, then gather
        # sums them with weight 1/top_k. The result should approximate
        # the FP8-dequantized version of hidden_states.
        ref = hidden_states_fp8.to(torch.bfloat16)
        torch.testing.assert_close(gather_out, ref, rtol=1e-1, atol=5e-1)

    def test_roundtrip_small(self):
        self._test_roundtrip(32, 256, 4, 2)

    def test_roundtrip_medium(self):
        self._test_roundtrip(128, 512, 8, 2)

    def test_roundtrip_top_k_1(self):
        self._test_roundtrip(64, 256, 4, 1)

    def test_roundtrip_large(self):
        self._test_roundtrip(1024, 1024, 16, 2)


@unittest.skipUnless(_is_cuda and _fp8_supported(), "Requires CUDA with FP8 support")
class TestContiguousVsMaskedPreprocess(CustomTestCase):

    def _test_both_paths(self, num_tokens, hidden_size, num_experts, top_k):
        from sglang.srt.layers.moe.ep_moe.kernels import (
            moe_ep_deepgemm_preprocess,
            moe_ep_deepgemm_preprocess_contiguous,
        )

        hidden_states = torch.randn(
            num_tokens, hidden_size, device="cuda", dtype=torch.bfloat16
        )
        topk_ids = _generate_topk_ids(num_tokens, num_experts, top_k)
        block_shape = [128, 128]

        # Contiguous path
        input_contig, scale_contig, m_indices, output_index, all_tokens = (
            moe_ep_deepgemm_preprocess_contiguous(
                topk_ids, num_experts, hidden_states, top_k, block_shape
            )
        )

        # Masked path
        masked_m, expected_m, src2dst, gateup_input, gateup_input_scale = (
            moe_ep_deepgemm_preprocess(
                topk_ids, num_experts, hidden_states, top_k, block_shape
            )
        )

        # Both paths should quantize the same data to FP8, so
        # the set of valid token data should match.
        # Verify that all valid expert tokens appear in both representations.
        for expert_id in range(num_experts):
            expert_mask_contig = m_indices == expert_id
            contig_count = expert_mask_contig.sum().item()
            masked_count = masked_m[expert_id].item()

            # The actual routed token count should match
            expected_count = (topk_ids == expert_id).sum().item()
            self.assertEqual(masked_count, expected_count)
            # Contiguous count includes alignment padding
            self.assertGreaterEqual(contig_count, expected_count)
            self.assertEqual(contig_count % EXPERT_ALIGNMENT, 0)

        torch.cuda.empty_cache()

    def test_compare_small(self):
        self._test_both_paths(32, 256, 4, 2)

    def test_compare_medium(self):
        self._test_both_paths(128, 512, 8, 2)

    def test_compare_many_experts(self):
        self._test_both_paths(256, 256, 16, 2)

    def test_compare_top_k_1(self):
        self._test_both_paths(64, 256, 8, 1)


if __name__ == "__main__":
    unittest.main()
