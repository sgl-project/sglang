import unittest

import torch
import triton.language as tl

from sglang.kernels.ops.moe.fused_moe_triton_kernels import (
    invoke_fused_moe_kernel,
)
from sglang.srt.layers.moe.shared_ep.pull_cache_prefill import (
    PullCache,
    allocate_pull_cache,
    invoke_pull_cache_w13,
    make_pull_cache_prefill_plan,
    pull_cache_rows,
)


class TestPullCachePrefillPlan(unittest.TestCase):
    def test_glm_prefill_plan(self):
        plan = make_pull_cache_prefill_plan(
            owners=8,
            source_tokens_per_owner=1024,
            hidden_size=6144,
            top_k=8,
            num_local_experts=32,
            expert_alignment=128,
        )

        self.assertEqual(plan.source_rows, 8192)
        self.assertEqual(plan.source_route_capacity, 65536)
        self.assertEqual(plan.cache_rows, 69600)
        self.assertEqual(plan.scale_groups, 48)
        self.assertEqual(plan.activation_bytes, 427622400)
        self.assertEqual(plan.scale_bytes, 13363200)
        self.assertEqual(plan.total_cache_bytes, 440985600)

    def test_invalid_dimensions_are_rejected(self):
        valid = {
            "owners": 8,
            "source_tokens_per_owner": 1024,
            "hidden_size": 6144,
            "top_k": 8,
            "num_local_experts": 32,
            "expert_alignment": 128,
        }
        for name in valid:
            with self.subTest(name=name):
                invalid = {**valid, name: 0}
                with self.assertRaisesRegex(ValueError, "positive"):
                    make_pull_cache_prefill_plan(**invalid)

    def test_hidden_size_must_be_divisible_by_scale_group(self):
        with self.assertRaisesRegex(ValueError, "divisible by 128"):
            make_pull_cache_prefill_plan(
                owners=8,
                source_tokens_per_owner=1024,
                hidden_size=6145,
                top_k=8,
                num_local_experts=32,
                expert_alignment=128,
            )

    def test_expert_alignment_must_be_a_power_of_two(self):
        with self.assertRaisesRegex(ValueError, "power of two"):
            make_pull_cache_prefill_plan(
                owners=8,
                source_tokens_per_owner=1024,
                hidden_size=6144,
                top_k=8,
                num_local_experts=32,
                expert_alignment=96,
            )

    def test_plan_is_immutable(self):
        plan = make_pull_cache_prefill_plan(
            owners=8,
            source_tokens_per_owner=1024,
            hidden_size=6144,
            top_k=8,
            num_local_experts=32,
            expert_alignment=128,
        )
        with self.assertRaises(AttributeError):
            plan.cache_rows = 1


class TestPullCachePrefillCuda(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_inactive_capacity_tail_is_not_rewritten(self):
        plan = make_pull_cache_prefill_plan(
            owners=1,
            source_tokens_per_owner=4,
            hidden_size=256,
            top_k=2,
            num_local_experts=2,
            expert_alignment=8,
        )
        cache = allocate_pull_cache(
            plan,
            active_rows=16,
            device=torch.device("cuda"),
        )
        cache.activations.fill_(7)
        cache.scales.fill_(7)
        sorted_token_ids = torch.full(
            (16,),
            plan.source_route_capacity,
            dtype=torch.int32,
            device="cuda",
        )
        sorted_token_ids[:2] = torch.tensor(
            [0, 2],
            dtype=torch.int32,
            device="cuda",
        )

        pull_cache_rows(
            source_activations=torch.ones(
                (plan.source_rows, plan.hidden_size),
                dtype=torch.float8_e4m3fn,
                device="cuda",
            ),
            source_scales=torch.ones(
                (plan.source_rows, plan.scale_groups),
                dtype=torch.float32,
                device="cuda",
            ),
            sorted_token_ids=sorted_token_ids,
            num_tokens_post_padded=torch.tensor(
                [8],
                dtype=torch.int32,
                device="cuda",
            ),
            cache=cache,
            top_k=plan.top_k,
            source_route_capacity=plan.source_route_capacity,
        )

        self.assertTrue(torch.all(cache.activations[8:] == 7))
        self.assertTrue(torch.all(cache.scales[8:] == 7))

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_permuted_rows_padding_and_reuse(self):
        plan = make_pull_cache_prefill_plan(
            owners=3,
            source_tokens_per_owner=4,
            hidden_size=256,
            top_k=2,
            num_local_experts=2,
            expert_alignment=8,
        )
        active_rows = 24
        cache = allocate_pull_cache(
            plan,
            active_rows=active_rows,
            device=torch.device("cuda"),
        )
        cache.activations.fill_(7)
        cache.scales.fill_(7)
        source_values = (
            torch.arange(plan.source_rows * plan.hidden_size, device="cuda")
            .view(plan.source_rows, plan.hidden_size)
            .remainder(31)
            .sub(15)
            .to(torch.float8_e4m3fn)
        )
        source_scales = (
            torch.arange(plan.source_rows * plan.scale_groups, device="cuda")
            .view(plan.source_rows, plan.scale_groups)
            .to(torch.float32)
        )
        route_capacity = source_values.shape[0] * plan.top_k
        sorted_token_ids = torch.full(
            (active_rows,),
            route_capacity,
            dtype=torch.int32,
            device="cuda",
        )
        sorted_token_ids[:3] = torch.tensor([7, 0, 22], dtype=torch.int32)
        sorted_token_ids[8:10] = torch.tensor([2, 15], dtype=torch.int32)
        expert_ids = torch.tensor([1, 0, -1], dtype=torch.int32, device="cuda")
        num_tokens_post_padded = torch.tensor(
            [16],
            dtype=torch.int32,
            device="cuda",
        )

        def expected():
            activations = torch.zeros(
                (active_rows, plan.hidden_size),
                dtype=torch.float8_e4m3fn,
                device="cuda",
            )
            scales = torch.zeros(
                (active_rows, plan.scale_groups),
                dtype=torch.float32,
                device="cuda",
            )
            activations[16:].fill_(7)
            scales[16:].fill_(7)
            for target_row, route in ((0, 7), (1, 0), (2, 22), (8, 2), (9, 15)):
                source_row = route // plan.top_k
                activations[target_row].copy_(source_values[source_row])
                scales[target_row].copy_(source_scales[source_row])
            return activations, scales

        for delta in (0, 1):
            if delta:
                source_values.copy_((source_values.float() + 1).to(source_values.dtype))
                source_scales.add_(1)
            pull_cache_rows(
                source_activations=source_values,
                source_scales=source_scales,
                sorted_token_ids=sorted_token_ids,
                num_tokens_post_padded=num_tokens_post_padded,
                cache=cache,
                top_k=plan.top_k,
                source_route_capacity=route_capacity,
            )
            expected_activations, expected_scales = expected()
            self.assertTrue(torch.equal(cache.activations, expected_activations))
            self.assertTrue(torch.equal(cache.scales, expected_scales))

        self.assertEqual(cache.scales.stride(0), 1)
        self.assertGreaterEqual(cache.scales.stride(1), cache.active_rows)
        self.assertTrue(
            torch.equal(
                cache.row_ids,
                torch.arange(active_rows, dtype=torch.int32, device="cuda"),
            )
        )
        self.assertEqual(cache.row_weights.shape, (active_rows, 1))
        self.assertTrue(torch.all(cache.row_weights == 1))
        self.assertTrue(cache.row_weights.is_contiguous())
        self.assertTrue(cache.activations.is_contiguous())

        config = {
            "BLOCK_SIZE_M": 8,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 1,
            "num_warps": 4,
            "num_stages": 3,
        }
        weights = torch.randint(
            -4,
            5,
            (2, 256, 256),
            dtype=torch.int8,
            device="cuda",
        ).to(torch.float8_e4m3fn)
        weight_scales = torch.ones((2, 2, 2), dtype=torch.float32, device="cuda")
        actual_w13 = torch.zeros(
            (active_rows, 256),
            dtype=torch.bfloat16,
            device="cuda",
        )
        reference_w13 = torch.zeros_like(actual_w13)
        invoke_pull_cache_w13(
            cache=cache,
            weight=weights,
            weight_scale=weight_scales,
            output=actual_w13,
            expert_ids=expert_ids,
            num_tokens_post_padded=num_tokens_post_padded,
            config=config,
            block_shape=(128, 128),
        )
        invoke_fused_moe_kernel(
            A=source_values,
            B=weights,
            bias=None,
            C=reference_w13,
            A_scale=source_scales,
            B_scale=weight_scales,
            B_zp=None,
            topk_weights=torch.ones(
                (plan.source_rows, plan.top_k),
                dtype=torch.float32,
                device="cuda",
            ),
            topk_ids=torch.zeros(
                (plan.source_rows, plan.top_k),
                dtype=torch.int32,
                device="cuda",
            ),
            sorted_token_ids=sorted_token_ids,
            expert_ids=expert_ids,
            num_tokens_post_padded=num_tokens_post_padded,
            mul_routed_weight=False,
            top_k=plan.top_k,
            config=config,
            compute_type=tl.bfloat16,
            use_fp8_w8a8=True,
            use_int8_w8a8=False,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            per_channel_quant=False,
            block_shape=[128, 128],
            filter_expert=True,
            c_sorted=True,
            a_is_prequantized=True,
        )
        valid_positions = sorted_token_ids < route_capacity
        self.assertTrue(
            torch.allclose(
                actual_w13[valid_positions].float(),
                reference_w13[valid_positions].float(),
                rtol=1e-2,
                atol=2e-2,
            )
        )

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_forged_row_major_scale_cache_is_rejected(self):
        plan = make_pull_cache_prefill_plan(
            owners=1,
            source_tokens_per_owner=2,
            hidden_size=256,
            top_k=2,
            num_local_experts=2,
            expert_alignment=8,
        )
        allocated = allocate_pull_cache(
            plan,
            active_rows=8,
            device=torch.device("cuda"),
        )
        forged = PullCache(
            plan=allocated.plan,
            active_rows=allocated.active_rows,
            activations=allocated.activations,
            scales=torch.empty(
                (allocated.active_rows, allocated.plan.scale_groups),
                dtype=torch.float32,
                device="cuda",
            ),
            row_ids=allocated.row_ids,
            row_weights=allocated.row_weights,
            scale_backing=allocated.scale_backing,
        )

        with self.assertRaisesRegex(ValueError, "TMA-aligned"):
            pull_cache_rows(
                source_activations=torch.ones(
                    (2, 256),
                    dtype=torch.float8_e4m3fn,
                    device="cuda",
                ),
                source_scales=torch.ones(
                    (2, 2),
                    dtype=torch.float32,
                    device="cuda",
                ),
                sorted_token_ids=torch.tensor(
                    [0, 1, 2, 3, 4, 4, 4, 4],
                    dtype=torch.int32,
                    device="cuda",
                ),
                num_tokens_post_padded=torch.tensor(
                    [4],
                    dtype=torch.int32,
                    device="cuda",
                ),
                cache=forged,
                top_k=2,
                source_route_capacity=4,
            )

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_empty_expert_set_does_not_touch_reused_cache(self):
        plan = make_pull_cache_prefill_plan(
            owners=1,
            source_tokens_per_owner=2,
            hidden_size=256,
            top_k=2,
            num_local_experts=2,
            expert_alignment=8,
        )
        cache = allocate_pull_cache(
            plan,
            active_rows=8,
            device=torch.device("cuda"),
        )
        cache.activations.fill_(1)
        cache.scales.fill_(1)
        pull_cache_rows(
            source_activations=torch.ones(
                (2, 256),
                dtype=torch.float8_e4m3fn,
                device="cuda",
            ),
            source_scales=torch.ones((2, 2), dtype=torch.float32, device="cuda"),
            sorted_token_ids=torch.full(
                (8,),
                4,
                dtype=torch.int32,
                device="cuda",
            ),
            num_tokens_post_padded=torch.zeros(
                (1,),
                dtype=torch.int32,
                device="cuda",
            ),
            cache=cache,
            top_k=2,
            source_route_capacity=4,
        )

        self.assertTrue(torch.all(cache.activations == 1))
        self.assertTrue(torch.all(cache.scales == 1))

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_invalid_pull_contract_is_rejected_before_launch(self):
        plan = make_pull_cache_prefill_plan(
            owners=1,
            source_tokens_per_owner=2,
            hidden_size=256,
            top_k=2,
            num_local_experts=2,
            expert_alignment=8,
        )
        cache = allocate_pull_cache(
            plan,
            active_rows=8,
            device=torch.device("cuda"),
        )
        source_activations = torch.zeros(
            (2, 256),
            dtype=torch.float8_e4m3fn,
            device="cuda",
        )
        source_scales = torch.ones((2, 2), dtype=torch.float32, device="cuda")
        sorted_token_ids = torch.full(
            (16,),
            4,
            dtype=torch.int32,
            device="cuda",
        )
        padded = torch.tensor([8], dtype=torch.int32, device="cuda")
        valid = {
            "source_activations": source_activations,
            "source_scales": source_scales,
            "sorted_token_ids": sorted_token_ids,
            "num_tokens_post_padded": padded,
            "cache": cache,
            "top_k": 2,
            "source_route_capacity": 4,
        }

        cases = (
            (
                {"source_activations": source_activations.float()},
                TypeError,
                "float8_e4m3fn",
            ),
            (
                {"source_scales": source_scales.to(torch.bfloat16)},
                TypeError,
                "float32",
            ),
            (
                {
                    "sorted_token_ids": torch.empty(
                        (16,),
                        dtype=torch.int32,
                        device="cuda",
                    )[::2]
                },
                ValueError,
                "contiguous",
            ),
            ({"source_route_capacity": 5}, ValueError, "cannot exceed"),
        )
        for mutation, error, message in cases:
            with self.subTest(mutation=next(iter(mutation))):
                with self.assertRaisesRegex(error, message):
                    pull_cache_rows(**{**valid, **mutation})


if __name__ == "__main__":
    unittest.main()
