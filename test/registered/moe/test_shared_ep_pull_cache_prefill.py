import unittest

import torch
import triton.language as tl

from sglang.kernels.ops.moe.fused_moe_triton_kernels import (
    invoke_fused_moe_kernel,
)
from sglang.srt.layers.moe.shared_ep.layout import shared_ep_fp8_dtype
from sglang.srt.layers.moe.shared_ep.pull_cache_prefill import (
    PullCache,
    allocate_pull_cache,
    invoke_pull_cache_w13,
    make_pull_cache_prefill_plan,
    pull_cache_rows,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-large")
register_amd_ci(est_time=20, stage="stage-b", runner_config="1-gpu-small-amd")


class TestPullCachePrefillPlan(unittest.TestCase):
    def test_release_profiles_have_exact_route_bounds(self):
        glm = make_pull_cache_prefill_plan(
            owners=8,
            source_tokens_per_owner=1024,
            hidden_size=6144,
            top_k=8,
            num_local_experts=32,
            expert_alignment=64,
        )
        dsv4 = make_pull_cache_prefill_plan(
            owners=8,
            source_tokens_per_owner=1024,
            hidden_size=4096,
            top_k=6,
            num_local_experts=32,
            expert_alignment=64,
        )
        self.assertEqual(glm.source_route_capacity, 65536)
        self.assertEqual(glm.cache_rows, 67552)
        self.assertEqual(glm.scale_groups, 48)
        self.assertEqual(dsv4.source_route_capacity, 49152)
        self.assertEqual(dsv4.cache_rows, 51168)
        self.assertEqual(dsv4.scale_groups, 32)

    def test_invalid_dimensions_are_rejected(self):
        valid = {
            "owners": 8,
            "source_tokens_per_owner": 1024,
            "hidden_size": 4096,
            "top_k": 6,
            "num_local_experts": 32,
            "expert_alignment": 64,
        }
        for name in valid:
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "positive"):
                    make_pull_cache_prefill_plan(**{**valid, name: 0})
        with self.assertRaisesRegex(ValueError, "divisible by 128"):
            make_pull_cache_prefill_plan(**{**valid, "hidden_size": 4097})
        with self.assertRaisesRegex(ValueError, "power of two"):
            make_pull_cache_prefill_plan(**{**valid, "expert_alignment": 96})


@unittest.skipUnless(torch.cuda.is_available(), "requires a GPU")
class TestPullCachePrefillGpu(unittest.TestCase):
    def _plan(self):
        return make_pull_cache_prefill_plan(
            owners=3,
            source_tokens_per_owner=4,
            hidden_size=256,
            top_k=2,
            num_local_experts=2,
            expert_alignment=8,
        )

    def test_permuted_rows_padding_tail_and_reuse(self):
        plan = self._plan()
        cache = allocate_pull_cache(
            plan,
            active_rows=plan.cache_rows,
            device=torch.device("cuda"),
        )
        cache.activations.fill_(7)
        cache.scales.fill_(7)
        source_values = (
            torch.arange(plan.source_rows * plan.hidden_size, device="cuda")
            .view(plan.source_rows, plan.hidden_size)
            .remainder(31)
            .sub(15)
            .to(shared_ep_fp8_dtype())
        )
        source_scales = (
            torch.arange(plan.source_rows * plan.scale_groups, device="cuda")
            .view(plan.source_rows, plan.scale_groups)
            .float()
        )
        sorted_ids = torch.full(
            (plan.cache_rows,),
            plan.source_route_capacity,
            dtype=torch.int32,
            device="cuda",
        )
        sorted_ids[:3] = torch.tensor([7, 0, 22], dtype=torch.int32, device="cuda")
        sorted_ids[8:10] = torch.tensor([2, 15], dtype=torch.int32, device="cuda")
        padded = torch.tensor([16], dtype=torch.int32, device="cuda")

        for delta in (0, 1):
            if delta:
                source_values.copy_((source_values.float() + 1).to(source_values.dtype))
                source_scales.add_(1)
            pull_cache_rows(
                source_activations=source_values,
                source_scales=source_scales,
                sorted_token_ids=sorted_ids,
                num_tokens_post_padded=padded,
                cache=cache,
                top_k=plan.top_k,
                source_route_capacity=plan.source_route_capacity,
            )
            expected_activations = torch.zeros_like(cache.activations)
            expected_scales = torch.zeros_like(cache.scales)
            expected_activations[16:].fill_(7)
            expected_scales[16:].fill_(7)
            for target_row, route in ((0, 7), (1, 0), (2, 22), (8, 2), (9, 15)):
                source_row = route // plan.top_k
                expected_activations[target_row].copy_(source_values[source_row])
                expected_scales[target_row].copy_(source_scales[source_row])
            self.assertTrue(torch.equal(cache.activations, expected_activations))
            self.assertTrue(torch.equal(cache.scales, expected_scales))

    def test_pull_w13_matches_direct_indexing(self):
        plan = self._plan()
        cache = allocate_pull_cache(
            plan,
            active_rows=plan.cache_rows,
            device=torch.device("cuda"),
        )
        source_values = (
            torch.arange(plan.source_rows * plan.hidden_size, device="cuda")
            .view(plan.source_rows, plan.hidden_size)
            .remainder(17)
            .sub(8)
            .to(shared_ep_fp8_dtype())
        )
        source_scales = torch.ones(
            (plan.source_rows, plan.scale_groups),
            dtype=torch.float32,
            device="cuda",
        )
        sorted_ids = torch.full(
            (plan.cache_rows,),
            plan.source_route_capacity,
            dtype=torch.int32,
            device="cuda",
        )
        sorted_ids[:3] = torch.tensor([7, 0, 22], dtype=torch.int32, device="cuda")
        sorted_ids[8:10] = torch.tensor([2, 15], dtype=torch.int32, device="cuda")
        expert_ids = torch.full(
            ((plan.cache_rows + 7) // 8,),
            -1,
            dtype=torch.int32,
            device="cuda",
        )
        expert_ids[:2] = torch.tensor([1, 0], dtype=torch.int32, device="cuda")
        padded = torch.tensor([16], dtype=torch.int32, device="cuda")
        pull_cache_rows(
            source_activations=source_values,
            source_scales=source_scales,
            sorted_token_ids=sorted_ids,
            num_tokens_post_padded=padded,
            cache=cache,
            top_k=plan.top_k,
            source_route_capacity=plan.source_route_capacity,
        )

        config = {
            "BLOCK_SIZE_M": 8,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 1,
            "num_warps": 4,
            "num_stages": 1,
        }
        weights = torch.randint(
            -4,
            5,
            (2, 512, 256),
            dtype=torch.int8,
            device="cuda",
        ).to(shared_ep_fp8_dtype())
        weight_scales = torch.ones((2, 4, 2), dtype=torch.float32, device="cuda")
        actual = torch.full(
            (plan.cache_rows, 512),
            7,
            dtype=torch.bfloat16,
            device="cuda",
        )
        reference = torch.full_like(actual, 7)
        invoke_pull_cache_w13(
            cache=cache,
            weight=weights,
            weight_scale=weight_scales,
            output=actual,
            expert_ids=expert_ids,
            num_tokens_post_padded=padded,
            config=config,
            block_shape=(128, 128),
        )
        invoke_fused_moe_kernel(
            A=source_values,
            B=weights,
            bias=None,
            C=reference,
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
            sorted_token_ids=sorted_ids,
            expert_ids=expert_ids,
            num_tokens_post_padded=padded,
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
        valid_positions = sorted_ids[:16] < plan.source_route_capacity
        torch.testing.assert_close(
            actual[:16][valid_positions],
            reference[:16][valid_positions],
            rtol=1e-2,
            atol=2e-2,
        )
        self.assertTrue(torch.all(actual[:16][~valid_positions] == 0))
        self.assertTrue(torch.all(actual[16:] == 7))

    def test_invalid_cache_contract_is_rejected(self):
        plan = self._plan()
        allocated = allocate_pull_cache(
            plan,
            active_rows=plan.cache_rows,
            device=torch.device("cuda"),
        )
        forged = PullCache(
            plan=allocated.plan,
            active_rows=allocated.active_rows,
            activations=allocated.activations,
            scales=torch.empty(
                (allocated.active_rows, allocated.plan.scale_groups * 2),
                dtype=torch.float32,
                device="cuda",
            )[:, ::2],
            row_ids=allocated.row_ids,
            row_weights=allocated.row_weights,
        )
        with self.assertRaisesRegex(ValueError, "contiguous row-major"):
            pull_cache_rows(
                source_activations=torch.zeros(
                    (plan.source_rows, plan.hidden_size),
                    dtype=shared_ep_fp8_dtype(),
                    device="cuda",
                ),
                source_scales=torch.ones(
                    (plan.source_rows, plan.scale_groups),
                    dtype=torch.float32,
                    device="cuda",
                ),
                sorted_token_ids=torch.full(
                    (plan.cache_rows,),
                    plan.source_route_capacity,
                    dtype=torch.int32,
                    device="cuda",
                ),
                num_tokens_post_padded=torch.zeros(
                    1,
                    dtype=torch.int32,
                    device="cuda",
                ),
                cache=forged,
                top_k=plan.top_k,
                source_route_capacity=plan.source_route_capacity,
            )


if __name__ == "__main__":
    unittest.main()
