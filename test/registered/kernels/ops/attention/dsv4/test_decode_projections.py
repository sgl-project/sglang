"""Numerical and input ownership checks for decode projections."""

import unittest

import torch
import torch.nn.functional as F

from sglang.kernels.ops.attention.dsv4.wo_a_bf16_gemv import wo_a_bf16_gemv
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=35, stage="base-b", runner_config="1-gpu-large")


class TestDecodeProjections(CustomTestCase):
    def test_grouped_gemv(self):
        for seed in (0, 17, 20260908):
            with self.subTest(seed=seed):
                torch.manual_seed(seed)
                x = torch.randn(1, 2, 4096, device="cuda", dtype=torch.bfloat16)
                w = torch.randn(2, 1024, 4096, device="cuda", dtype=torch.bfloat16)
                original_x, original_w = x.clone(), w.clone()
                result = wo_a_bf16_gemv(x, w)
                reference = torch.einsum("tgd,grd->tgr", x, w)
                self.assertEqual(result.dtype, torch.bfloat16)
                self.assertTrue(result.is_contiguous())
                torch.testing.assert_close(result, reference, atol=1e-3, rtol=8e-3)
                # Sample both groups against CPU FP64 accumulation independently
                # of the cuBLAS algorithm selected by the serving reference.
                exact = torch.einsum(
                    "tgd,grd->tgr", x.cpu().double(), w[:, ::31].cpu().double()
                ).bfloat16()
                torch.testing.assert_close(
                    result[:, :, ::31].cpu(), exact, atol=1e-3, rtol=8e-3
                )
                self.assertTrue(torch.equal(x, original_x))
                self.assertTrue(torch.equal(w, original_w))

    def test_fp4_indexer_direct_mapping_matches_explicit_slots(self):
        from sglang.kernels.ops.attention.dsv4.sm90_fp4_indexer import (
            fp4_index_logits_decode,
            fp4_index_logits_req_to_token,
        )

        torch.manual_seed(17)
        rows, heads, width, page_size = 3, 32, 257, 64
        q = torch.randn(rows, heads, 128, device="cuda", dtype=torch.bfloat16)
        weights = torch.randn(rows, heads, device="cuda", dtype=torch.bfloat16)
        req = torch.tensor([2, 0, 1], device="cuda", dtype=torch.int64)
        lens = torch.tensor([257, 130, 0], device="cuda", dtype=torch.int64)
        num_pages = 16
        table = torch.randint(
            0,
            256,
            (num_pages, page_size * 68),
            device="cuda",
            dtype=torch.uint8,
        )
        table[:, page_size * 64 :] = 127

        for ratio in (1, 2):
            with self.subTest(ratio=ratio):
                logical_slots = torch.stack(
                    [
                        torch.randperm(num_pages * page_size, device="cuda")[:width]
                        for _ in range(3)
                    ]
                ).to(torch.int32)
                req_to_token = torch.zeros(
                    3, width * ratio, device="cuda", dtype=torch.int32
                )
                req_to_token[:, ::ratio] = logical_slots * ratio
                explicit_slots = (
                    req_to_token[
                        req[:, None], torch.arange(width, device="cuda") * ratio
                    ]
                    // ratio
                )

                expected = fp4_index_logits_decode(
                    q, weights, explicit_slots, lens, table, page_size
                )
                actual = fp4_index_logits_req_to_token(
                    q,
                    weights,
                    req_to_token,
                    req,
                    lens,
                    table,
                    page_size,
                    ratio,
                    width,
                )
                torch.testing.assert_close(actual, expected, equal_nan=True)

    def test_candidate_block_indices_matches_torch(self):
        from sglang.kernels.ops.attention.dsv4.candidate_blocks import (
            candidate_block_indices,
        )

        torch.manual_seed(29)
        rows, width, block_size, topk_blocks = 3, 259, 8, 7
        logits = torch.randn(rows, width, device="cuda")
        lens = torch.tensor([259, 130, 0], device="cuda", dtype=torch.int32)

        actual_indices, actual_valid = candidate_block_indices(
            logits,
            lens,
            topk_blocks=topk_blocks,
            block_size=block_size,
        )

        padded = F.pad(logits, (0, -width % block_size), value=-torch.inf)
        scores = padded.unflatten(-1, (-1, block_size)).amax(dim=-1)
        last = (lens - 1) // block_size
        scores = scores.masked_fill(
            torch.arange(scores.shape[1], device="cuda") == last[:, None],
            torch.inf,
        )
        expected = scores.topk(topk_blocks, dim=-1)

        torch.testing.assert_close(
            actual_indices.sort(dim=-1).values,
            expected.indices.sort(dim=-1).values,
        )
        torch.testing.assert_close(actual_valid, expected.values > -torch.inf)


if __name__ == "__main__":
    unittest.main()
