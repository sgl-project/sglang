# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import patch

import torch
import triton.language as tl

from sglang.kernels.ops.moe.fused_moe_triton_kernels import (
    invoke_fused_moe_kernel,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=90, stage="base-c", runner_config="4-gpu-gb300")


class TestLfm25FusedMoeSm103(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        if torch.cuda.get_device_capability() != (10, 3):
            raise unittest.SkipTest("The LFM2.5 fast path requires SM103")

    @staticmethod
    def _routing_metadata(topk_ids: torch.Tensor, block_size: int):
        flat_ids = topk_ids.flatten()
        num_experts = 32
        if flat_ids.numel() < num_experts + 1:
            max_num_tokens_padded = flat_ids.numel() * block_size
        else:
            max_num_tokens_padded = flat_ids.numel() + (num_experts + 1) * (
                block_size - 1
            )

        sorted_ids = torch.full(
            (max_num_tokens_padded,),
            flat_ids.numel(),
            device="cuda",
            dtype=torch.int32,
        )
        expert_ids = torch.zeros(
            ((max_num_tokens_padded + block_size - 1) // block_size,),
            device="cuda",
            dtype=torch.int32,
        )
        cursor = 0
        for expert in range(num_experts):
            token_ids = (flat_ids == expert).nonzero().flatten().to(torch.int32)
            if not token_ids.numel():
                continue
            padded = ((token_ids.numel() + block_size - 1) // block_size) * block_size
            sorted_ids[cursor : cursor + token_ids.numel()] = token_ids
            expert_ids[cursor // block_size : (cursor + padded) // block_size] = expert
            cursor += padded

        num_tokens_post_pad = torch.tensor([cursor], device="cuda", dtype=torch.int32)
        return sorted_ids, expert_ids, num_tokens_post_pad

    def _run_case(
        self,
        num_tokens: int,
        n: int,
        k: int,
        top_k: int,
        mul_routed_weight: bool,
    ):
        generator = torch.Generator(device="cuda").manual_seed(num_tokens + n + k)
        topk_ids = torch.randint(
            0,
            32,
            (num_tokens, 4),
            device="cuda",
            dtype=torch.int32,
            generator=generator,
        )
        topk_weights = torch.softmax(
            torch.randn(
                num_tokens, 4, device="cuda", generator=generator, dtype=torch.float32
            ),
            dim=-1,
        )
        a_rows = num_tokens if top_k == 4 else num_tokens * 4
        a = torch.randn(
            a_rows,
            k,
            device="cuda",
            generator=generator,
            dtype=torch.bfloat16,
        )
        b = (
            torch.randn(
                32,
                n,
                k,
                device="cuda",
                generator=generator,
                dtype=torch.bfloat16,
            )
            * 0.02
        )
        reference = torch.full(
            (num_tokens * 4, n),
            float("nan"),
            device="cuda",
            dtype=torch.bfloat16,
        )
        actual = torch.full_like(reference, float("nan"))
        sorted_ids, expert_ids, num_tokens_post_pad = self._routing_metadata(
            topk_ids, 16
        )
        args = (
            a,
            b,
            None,
            reference,
            None,
            None,
            None,
            topk_weights,
            topk_ids,
            sorted_ids,
            expert_ids,
            num_tokens_post_pad,
            mul_routed_weight,
            top_k,
            {
                "BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 1,
                "num_warps": 4,
                "num_stages": 3,
            },
            tl.bfloat16,
            False,
            False,
            False,
            False,
            False,
            None,
            False,
            False,
            False,
            False,
            False,
            False,
            1,
            False,
            None,
            False,
            False,
            False,
        )

        with patch(
            "sglang.kernels.ops.moe.lfm25_fused_moe_sm103."
            "can_use_lfm25_fused_moe_sm103",
            return_value=False,
        ):
            invoke_fused_moe_kernel(*args)
        candidate_args = list(args)
        candidate_args[3] = actual
        invoke_fused_moe_kernel(*candidate_args)
        torch.cuda.synchronize()

        torch.testing.assert_close(actual, reference, rtol=0, atol=0, equal_nan=True)

    def test_matches_generic_triton_for_lfm25_shapes(self):
        cases = (
            (1, 3584, 2048, 4, False),
            (32, 3584, 2048, 4, False),
            (32, 2048, 1792, 1, True),
            (2048, 3584, 2048, 4, False),
        )
        for case in cases:
            with self.subTest(case=case):
                self._run_case(*case)


if __name__ == "__main__":
    unittest.main()
