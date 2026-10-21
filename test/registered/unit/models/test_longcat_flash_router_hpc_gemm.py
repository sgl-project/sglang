"""Unit tests for LongCat-Flash router GEMM dispatch to the HPC-Ops bf16xfp32 kernel."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.models.longcat_flash import (  # noqa: E402
    LongcatFlashMoE,
    LongcatFlashRouter,
)
from sglang.srt.runtime_context import get_context, get_exec, get_parallel  # noqa: E402

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


def _longcat_config(hidden_size, n_routed_experts, *, router_bias=False):
    return SimpleNamespace(
        hidden_size=hidden_size,
        n_routed_experts=n_routed_experts,
        router_bias=router_bias,
    )


class TestLongcatFlashRouterHpcGemm(CustomTestCase):
    def test_zero_experts_reject_placement_before_loading_expert_weights(self):
        """Identity expert IDs cannot index routed-only placement tables."""
        config = SimpleNamespace(
            routed_scaling_factor=1.0,
            n_routed_experts=4,
            moe_topk=2,
            zero_expert_num=2,
            zero_expert_type="identity",
            rounter_params_dtype="float32",
            hidden_act="silu",
        )
        with get_context().override_server_args(model_path="dummy", device="cpu"):
            for algorithm in ("static", "dynamic", "fake", "lp"):
                with (
                    self.subTest(algorithm=algorithm),
                    get_exec().moe.override(ep_dispatch_algorithm=algorithm),
                    self.assertRaisesRegex(ValueError, "LongCat zero experts"),
                ):
                    LongcatFlashMoE(config=config, layer_id=0)

            with (
                get_exec().moe.override(ep_dispatch_algorithm=None),
                get_parallel().override(tp_size=1),
                patch(
                    "sglang.srt.models.longcat_flash.LongcatFlashRouter",
                    side_effect=RuntimeError("placement admitted"),
                ),
                self.assertRaisesRegex(RuntimeError, "placement admitted"),
            ):
                LongcatFlashMoE(config=config, layer_id=0)

    def _assert_dispatches_to_hpc_gemm(
        self,
        *,
        hidden_size,
        n_routed_experts,
        zero_expert_num,
        expected_min_m,
    ):
        router = LongcatFlashRouter(
            _longcat_config(hidden_size, n_routed_experts),
            zero_expert_num=zero_expert_num,
            rounter_params_dtype=torch.float32,
        )
        hidden_states = torch.randn((4, hidden_size), dtype=torch.bfloat16)
        expected = torch.randn(
            (4, n_routed_experts + zero_expert_num), dtype=torch.float32
        )

        with patch(
            "sglang.srt.models.longcat_flash.linear_bf16_fp32",
            return_value=expected,
        ) as mock_linear:
            out = router(hidden_states)

        self.assertIs(out, expected)
        mock_linear.assert_called_once()
        args, kwargs = mock_linear.call_args
        self.assertIs(args[0], hidden_states)
        self.assertIs(args[1], router.classifier.weight)
        self.assertEqual(kwargs["hpc_kernel_min_m"], expected_min_m)

    def _assert_uses_classifier(self, router, hidden_size):
        hidden_states = torch.randn((4, hidden_size), dtype=torch.bfloat16)
        expected = torch.randn((4, router.n_routed_experts), dtype=torch.float32)

        with (
            patch(
                "sglang.srt.models.longcat_flash.linear_bf16_fp32",
                side_effect=AssertionError("unexpected hpc kernel dispatch"),
            ),
            patch.object(
                router.classifier,
                "forward",
                return_value=(expected, None),
            ) as mock_classifier,
        ):
            out = router(hidden_states)

        self.assertIs(out, expected)
        mock_classifier.assert_called_once()
        self.assertEqual(mock_classifier.call_args.args[0].dtype, torch.float32)

    def test_chat_shape_dispatches_with_benchmark_guard(self):
        self._assert_dispatches_to_hpc_gemm(
            hidden_size=6144,
            n_routed_experts=512,
            zero_expert_num=256,
            expected_min_m=64,
        )

    def test_lite_shape_dispatches_with_benchmark_guard(self):
        self._assert_dispatches_to_hpc_gemm(
            hidden_size=3072,
            n_routed_experts=256,
            zero_expert_num=128,
            expected_min_m=128,
        )

    def test_unbenchmarked_shape_uses_classifier(self):
        router = LongcatFlashRouter(
            _longcat_config(4096, 256),
            zero_expert_num=128,
            rounter_params_dtype=torch.float32,
        )

        self._assert_uses_classifier(router, hidden_size=4096)

    def test_router_bias_uses_classifier(self):
        router = LongcatFlashRouter(
            _longcat_config(6144, 512, router_bias=True),
            zero_expert_num=256,
            rounter_params_dtype=torch.float32,
        )

        self._assert_uses_classifier(router, hidden_size=6144)


if __name__ == "__main__":
    unittest.main()
