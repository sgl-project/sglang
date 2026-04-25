from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=4, suite="stage-b-test-1-gpu-small")

import importlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

fused_moe_mod = importlib.import_module(
    "sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe"
)


class TestFusedMoeDeterministicPath(unittest.TestCase):
    def _run_small_token_case(self, *, backend: str, deterministic: bool):
        num_tokens = 4
        hidden_dim = 2
        topk = 3
        intermediate_dim = 4
        num_experts = 5
        routed_scaling_factor = 0.5

        hidden_states = torch.zeros((num_tokens, hidden_dim), dtype=torch.float32)
        w1 = torch.zeros(
            (num_experts, intermediate_dim, hidden_dim), dtype=torch.float32
        )
        w2 = torch.zeros(
            (num_experts, hidden_dim, intermediate_dim // 2), dtype=torch.float32
        )
        topk_weights = torch.full((num_tokens, topk), 1.0 / topk, dtype=torch.float32)
        topk_ids = torch.zeros((num_tokens, topk), dtype=torch.int32)
        sorted_token_ids = torch.zeros((num_tokens * topk,), dtype=torch.int32)
        expert_ids = torch.zeros((num_tokens * topk,), dtype=torch.int32)
        num_tokens_post_padded = torch.tensor(num_tokens * topk, dtype=torch.int32)

        server_args = SimpleNamespace(
            enable_fused_moe_sum_all_reduce=False,
            enable_deterministic_inference=deterministic,
        )

        def fake_invoke_fused_moe_kernel(*args, **kwargs):
            output = args[3]
            output.fill_(1.0)

        def fake_silu_and_mul(_x, out):
            out.fill_(1.0)

        def fake_reduce(x, out, routed_scaling_factor):
            out.copy_(x.sum(dim=1) * routed_scaling_factor)

        reduce_fn_name = (
            "moe_sum_reduce" if backend == "cuda" else "moe_sum_reduce_triton"
        )

        with (
            patch.object(fused_moe_mod, "_is_cuda", backend == "cuda"),
            patch.object(fused_moe_mod, "_is_hip", backend == "hip"),
            patch.object(fused_moe_mod, "_is_xpu", False),
            patch.object(fused_moe_mod, "_use_aiter", False),
            patch.object(fused_moe_mod, "_has_vllm_ops", False),
            patch.object(
                fused_moe_mod,
                "get_global_server_args",
                return_value=server_args,
            ),
            patch.object(
                fused_moe_mod,
                "invoke_fused_moe_kernel",
                side_effect=fake_invoke_fused_moe_kernel,
            ),
            patch.object(
                fused_moe_mod,
                "silu_and_mul",
                side_effect=fake_silu_and_mul,
            ),
            patch.object(
                fused_moe_mod,
                "moe_sum_reduce_torch_compile",
                side_effect=fake_reduce,
            ) as mock_compile_reduce,
            patch.object(
                fused_moe_mod,
                reduce_fn_name,
                side_effect=fake_reduce,
            ) as mock_regular_reduce,
        ):
            out = fused_moe_mod._fused_moe_kernel_sequence(
                hidden_states,
                w1,
                w2,
                topk_weights,
                topk_ids,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                {"BLOCK_SIZE_M": 16},
                None,
                False,
                b1=None,
                b2=None,
                use_fp8_w8a8=False,
                use_int8_w8a8=False,
                use_int8_w8a16=False,
                use_int4_w4a16=False,
                per_channel_quant=False,
                w1_scale=None,
                w2_scale=None,
                w1_zp=None,
                w2_zp=None,
                a1_scale=None,
                a2_scale=None,
                block_shape=None,
                activation="silu",
                is_gated=True,
                no_combine=False,
                inplace=False,
                apply_router_weight_on_input=False,
                routed_scaling_factor=routed_scaling_factor,
                gemm1_alpha=None,
                gemm1_limit=None,
                filter_expert=False,
                hooks=None,
            )

        expected = torch.full_like(hidden_states, topk * routed_scaling_factor)
        torch.testing.assert_close(out, expected)

        if deterministic:
            mock_compile_reduce.assert_not_called()
            mock_regular_reduce.assert_called_once()
        else:
            mock_compile_reduce.assert_called_once()
            mock_regular_reduce.assert_not_called()

    def test_cuda_small_token_path_respects_deterministic_flag(self):
        for deterministic in [False, True]:
            with self.subTest(deterministic=deterministic):
                self._run_small_token_case(backend="cuda", deterministic=deterministic)

    def test_hip_small_token_path_respects_deterministic_flag(self):
        for deterministic in [False, True]:
            with self.subTest(deterministic=deterministic):
                self._run_small_token_case(backend="hip", deterministic=deterministic)


if __name__ == "__main__":
    unittest.main()
